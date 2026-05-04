"""Inject a splat-rendered subject into FLUX.2-generated scenes.

For each prompt: render the splat at a chosen camera, hand the render to
FLUX.2 as conditioning, generate.

Three conditioning modes (named honestly):

  flux2-ref (default, implemented today):
    splat-render is passed to FLUX.2 as the only reference image via the
    pipeline's native multimodal encoder (Mistral3 + Pixtral). Same
    cross-attention path as the flux2-ref command, just with a single
    splat-rendered reference. Cheap, leverages existing infrastructure.
    Limitation: refs are NON-SPATIAL — identity features (face, body
    shape) propagate well, but pixel-level spatial layout (exact vitiligo
    patch positions, scar locations) may not survive cross-attention.
    Good first test.

  img2img-strength (pending, ~1 day):
    Custom denoising loop. VAE-encode the splat-render to a clean latent,
    noise it at strength × σ_max, then run the FLUX.2 scheduler from that
    intermediate noise level. Spatial layout is preserved by construction
    (the latent starts as the splat-render and is only partially
    re-denoised). This is the right tool for vitiligo topology.

  depth-controlnet (pending, requires FLUX.2 ControlNet availability):
    Render depth from the splat, condition FLUX.2 generation on it via a
    depth ControlNet. Frees framing more than img2img while preserving
    3D structure.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


SPLAT_INJECT_VERSION = "0.1"


def _load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"prompt entry missing 'text' field: {obj}")
            prompts.append(obj)
    return prompts


def inject_splat(
    splat_dir: str,
    pipeline_id: str,
    prompts_path: str,
    output: str,
    *,
    mode: str = "flux2-ref",
    strength: float = 0.7,
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    dtype: str = "bfloat16",
    offload: str = "sequential",
    view_strategy: str = "canonical_frontal",
    identity_tag: str | None = None,
    num_ref_views: int = 1,
    init_view: str = "training_first",
) -> dict:
    """Inject the splat through FLUX.2 to generate scenes.

    Args:
        splat_dir: path to a .splat directory (output of splat-build)
        pipeline_id: HF model id for the diffusion model
        prompts_path: jsonl with one {"text": "..."} per line
        output: directory to write generated images + metadata + splat_renders/
        mode: conditioning mode (see module docstring)
        strength: img2img denoising strength (only used for mode=img2img-strength)
        view_strategy:
            - "canonical_frontal": same front-facing view for every prompt
            - "training_first": use the first training camera (stable but limited)
            - "training_random": sample a random training camera per prompt
            - "training_diverse": farthest-point sampled training cameras
              (used as the multi-view ref stack; only meaningful for
              flux2-ref / flux2-inpaint-img2img with num_ref_views > 1).
        identity_tag: optional text appended to every prompt, same as flux2-ref
        num_ref_views: number of distinct splat views to feed FLUX.2's
            multimodal encoder as identity references. 1 = current single-ref
            behaviour, up to 10 (FLUX.2 cap). Picked via farthest-point
            sampling on training-camera positions for spatial diversity.
        init_view: which splat view becomes the img2img init latent (only
            used by flux2-inpaint-img2img). One of training_first /
            canonical_frontal / training_random.
    """
    if mode == "flux2-ref":
        return _inject_via_flux2_ref(
            splat_dir=splat_dir, pipeline_id=pipeline_id,
            prompts_path=prompts_path, output=output,
            seed=seed, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, width=width, height=height,
            dtype=dtype, offload=offload, view_strategy=view_strategy,
            identity_tag=identity_tag, num_ref_views=num_ref_views,
        )
    elif mode == "flux2-inpaint-img2img":
        return _inject_via_flux2_inpaint_img2img(
            splat_dir=splat_dir, pipeline_id=pipeline_id,
            prompts_path=prompts_path, output=output,
            seed=seed, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, width=width, height=height,
            dtype=dtype, offload=offload, identity_tag=identity_tag,
            strength=strength, num_ref_views=num_ref_views,
            init_view=init_view,
        )
    elif mode == "img2img-strength":
        raise NotImplementedError(
            "mode=img2img-strength superseded by flux2-inpaint-img2img "
            "(diffusers Flux2KleinInpaintPipeline with mask=ones gives the "
            "same semantics + multi-view ref support)."
        )
    elif mode == "depth-controlnet":
        raise NotImplementedError(
            "mode=depth-controlnet pending (requires FLUX.2 depth ControlNet)."
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}")


def _inject_via_flux2_ref(
    splat_dir: str,
    pipeline_id: str,
    prompts_path: str,
    output: str,
    *,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    dtype: str,
    offload: str,
    view_strategy: str,
    identity_tag: str | None,
    num_ref_views: int = 1,
) -> dict:
    """Render splat → save PNG → call generate_with_flux2_ref.

    With num_ref_views>1, picks farthest-point-sampled training cameras and
    feeds all renders as the multi-image ref stack to FLUX.2's multimodal
    encoder.
    """
    from .splat_render import (
        CameraView, render_splat, render_at_training_camera,
        select_diverse_training_views,
    )
    from ..profile.flux2_ref import generate_with_flux2_ref

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    splat_path = Path(splat_dir)

    # Validate the splat
    for required in ("gaussians.ply", "cameras.json", "metadata.json", "bbox.json"):
        if not (splat_path / required).exists():
            raise ValueError(f"{splat_path} is missing {required} — not a valid .splat")

    with open(splat_path / "metadata.json") as fh:
        splat_meta = json.load(fh)
    print(f"splat: impl={splat_meta.get('impl')}, "
          f"{splat_meta.get('n_registered_cameras')} cameras, "
          f"{splat_meta.get('n_sparse_points')} init points")

    prompts = _load_prompts(prompts_path)
    print(f"prompts: {len(prompts)}")

    # Render splat once (canonical view) or per-prompt depending on strategy.
    splat_render_dir = out_dir / "splat_renders"
    splat_render_dir.mkdir(exist_ok=True)
    print(f"\nrendering splats (view_strategy={view_strategy})")
    rendered_paths: list[str] = []  # list-of-list when num_ref_views>1
    if num_ref_views > 1:
        # Multi-view ref stack — render N diverse training cameras once, share
        # the same stack across every prompt.
        indices = select_diverse_training_views(str(splat_path), n=num_ref_views)
        view_paths: list[str] = []
        for idx in indices:
            im = render_at_training_camera(str(splat_path), image_index=idx)
            p = splat_render_dir / f"view_{idx:03d}.png"
            im.save(p)
            view_paths.append(str(p))
        print(f"  multi-view stack: {len(view_paths)} renders → {view_paths}")
        rendered_paths = [view_paths] * len(prompts)  # type: ignore[list-item]
    elif view_strategy == "canonical_frontal":
        view = CameraView.canonical_frontal(str(splat_path), width=width, height=height)
        img = render_splat(str(splat_path), view)
        path = splat_render_dir / "canonical.png"
        img.save(path)
        rendered_paths = [str(path)] * len(prompts)  # same canonical for every prompt
    elif view_strategy == "training_first":
        img = render_at_training_camera(str(splat_path), image_index=0)
        path = splat_render_dir / "training_first.png"
        img.save(path)
        rendered_paths = [str(path)] * len(prompts)
    elif view_strategy == "training_random":
        rng = np.random.default_rng(seed)
        with open(splat_path / "cameras.json") as fh:
            cam_data = json.load(fh)
        n_cams = len(cam_data["images"])
        for pi in range(len(prompts)):
            idx = int(rng.integers(0, n_cams))
            img = render_at_training_camera(str(splat_path), image_index=idx)
            path = splat_render_dir / f"prompt_{pi:03d}_view{idx}.png"
            img.save(path)
            rendered_paths.append(str(path))
    else:
        raise ValueError(f"unknown view_strategy: {view_strategy}")

    # Determine if every prompt sees the same ref set (single call) or
    # not (per-prompt invocation).
    def _to_list(item):
        return item if isinstance(item, list) else [item]
    canonical_refs = _to_list(rendered_paths[0])
    same_for_all = all(_to_list(rp) == canonical_refs for rp in rendered_paths)
    print(f"rendered {len(canonical_refs)} ref view(s); "
          f"{'shared across all prompts' if same_for_all else 'per-prompt views'}")

    # Hand the splat-render to FLUX.2 as the only reference. Per-prompt-view
    # support requires invoking generate_with_flux2_ref multiple times if
    # rendered_paths varies. For canonical strategy, one call suffices.
    if same_for_all:
        # Single call with the shared ref stack
        meta = generate_with_flux2_ref(
            pipeline_id=pipeline_id,
            prompts_path=prompts_path,
            output=str(out_dir),
            ref_images=canonical_refs,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width, height=height,
            dtype=dtype, offload=offload,
            identity_tag=identity_tag,
        )
    else:
        # Per-prompt views — need per-prompt invocation. Fall back to
        # writing a single-prompt jsonl per call. Slower because pipe
        # reloads each time; acceptable for training_random sanity tests.
        elapsed_total = 0
        for pi, (prompt_obj, ref_paths) in enumerate(zip(prompts, rendered_paths)):
            ref_paths = _to_list(ref_paths)
            sub_prompts = out_dir / f"_prompt_{pi:03d}.jsonl"
            with open(sub_prompts, "w") as fh:
                fh.write(json.dumps(prompt_obj) + "\n")
            sub_out = out_dir / f"_run_{pi:03d}"
            t0 = time.time()
            generate_with_flux2_ref(
                pipeline_id=pipeline_id,
                prompts_path=str(sub_prompts),
                output=str(sub_out),
                ref_images=ref_paths,
                seed=seed + pi,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width, height=height,
                dtype=dtype, offload=offload,
                identity_tag=identity_tag,
            )
            elapsed_total += time.time() - t0
            # Move the single output up
            src = sub_out / "prompt_000.png"
            if src.exists():
                src.rename(out_dir / f"prompt_{pi:03d}.png")
        meta = {"per_prompt_run_total_s": round(elapsed_total, 1)}

    # Stamp our own metadata on top
    inject_meta = {
        "version": SPLAT_INJECT_VERSION,
        "splat_dir": str(splat_path),
        "splat_impl": splat_meta.get("impl"),
        "pipeline_id": pipeline_id,
        "mode": "flux2-ref",
        "view_strategy": view_strategy,
        "identity_tag": identity_tag,
        "n_prompts": len(prompts),
        "n_ref_views": len(canonical_refs),
        "ref_views_shared": same_for_all,
        "underlying_call": meta,
    }
    with open(out_dir / "splat_inject_metadata.json", "w") as fh:
        json.dump(inject_meta, fh, indent=2)
    print(f"\ninject done. wrote {out_dir}")
    return inject_meta


def _inject_via_flux2_inpaint_img2img(
    splat_dir: str,
    pipeline_id: str,
    prompts_path: str,
    output: str,
    *,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    dtype: str,
    offload: str,
    identity_tag: str | None,
    strength: float,
    num_ref_views: int,
    init_view: str,
) -> dict:
    """img2img + multi-view ref via Flux2KleinInpaintPipeline (mask=ones).

    Uses an inpaint pipeline as the actual primitive: with a fully-white
    mask the pipeline behaves as proper img2img (init = VAE-encoded
    splat-render, schedule truncated by `strength`). The separate
    `image_reference` channel takes our multi-view stack so identity
    cross-attention isn't paying for spatial preservation. This is the
    "geometry from img2img + identity from refs" combination.
    """
    from PIL import Image
    import torch
    from .splat_render import (
        CameraView, render_splat, render_at_training_camera,
        select_diverse_training_views,
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    splat_path = Path(splat_dir)
    for required in ("gaussians.ply", "cameras.json", "metadata.json", "bbox.json"):
        if not (splat_path / required).exists():
            raise ValueError(f"{splat_path} is missing {required} — not a valid .splat")
    with open(splat_path / "metadata.json") as fh:
        splat_meta = json.load(fh)
    print(f"splat: impl={splat_meta.get('impl')}, "
          f"{splat_meta.get('n_registered_cameras')} cameras, "
          f"{splat_meta.get('n_sparse_points')} init points")

    prompts = _load_prompts(prompts_path)
    print(f"prompts: {len(prompts)}, strength={strength}, "
          f"num_ref_views={num_ref_views}, init_view={init_view}")

    splat_render_dir = out_dir / "splat_renders"
    splat_render_dir.mkdir(exist_ok=True)

    # 1. Pick the init view (the splat-render that becomes the img2img latent).
    if init_view == "training_first":
        init_img = render_at_training_camera(str(splat_path), image_index=0)
        init_label = "training_first"
    elif init_view == "canonical_frontal":
        view = CameraView.canonical_frontal(str(splat_path), width=width, height=height)
        init_img = render_splat(str(splat_path), view)
        init_label = "canonical"
    elif init_view == "training_random":
        rng = np.random.default_rng(seed)
        with open(splat_path / "cameras.json") as fh:
            cam_data = json.load(fh)
        n_cams = len(cam_data["images"])
        idx = int(rng.integers(0, n_cams))
        init_img = render_at_training_camera(str(splat_path), image_index=idx)
        init_label = f"training_random{idx}"
    else:
        raise ValueError(f"unknown init_view: {init_view!r}")
    init_img = init_img.convert("RGB").resize((width, height), Image.LANCZOS)
    init_path = splat_render_dir / f"init_{init_label}.png"
    init_img.save(init_path)
    print(f"init view: {init_path}")

    # 2. Render the multi-view ref stack (diverse training cameras).
    ref_pils: list[Image.Image] = []
    if num_ref_views > 0:
        indices = select_diverse_training_views(str(splat_path), n=num_ref_views)
        for idx in indices:
            im = render_at_training_camera(str(splat_path), image_index=idx)
            im = im.convert("RGB")
            p = splat_render_dir / f"ref_view_{idx:03d}.png"
            im.save(p)
            ref_pils.append(im)
        print(f"ref stack: {len(ref_pils)} views (training indices {indices})")

    # 3. Build the mask = full white (inpaint everywhere = pure img2img).
    mask_img = Image.new("L", (width, height), 255)

    # 4. Load the inpaint pipeline.
    from diffusers import Flux2KleinInpaintPipeline
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print(f"loading {pipeline_id} ({dtype}, offload={offload})...")
    pipe = Flux2KleinInpaintPipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe = pipe.to("cuda")
    else:
        raise ValueError(f"unknown offload: {offload}")

    # 5. Generate per prompt.
    t0 = time.time()
    elapsed = []
    all_meta = []
    for pi, p in enumerate(prompts):
        text = p["text"]
        gen_prompt = f"{text}, {identity_tag}" if identity_tag else text
        gen = torch.Generator(device="cuda").manual_seed(seed + pi)
        ti = time.time()
        try:
            with torch.no_grad():
                result = pipe(
                    prompt=gen_prompt,
                    image=init_img,
                    mask_image=mask_img,
                    image_reference=ref_pils if ref_pils else None,
                    strength=strength,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
            img = result.images[0] if hasattr(result, "images") else None
        except Exception as e:
            print(f"  prompt {pi}: ERROR {type(e).__name__}: {str(e)[:200]}")
            raise
        if img is not None:
            img.save(out_dir / f"prompt_{pi:03d}.png")
        dt = time.time() - ti
        elapsed.append(dt)
        all_meta.append({"idx": pi, "text": text,
                          **{k: v for k, v in p.items() if k != "text"},
                          "elapsed_s": round(dt, 2)})
        print(f"  [{pi+1}/{len(prompts)}] {dt:.1f}s — {text[:70]!r}")

    inject_meta = {
        "version": SPLAT_INJECT_VERSION,
        "splat_dir": str(splat_path),
        "splat_impl": splat_meta.get("impl"),
        "pipeline_id": pipeline_id,
        "mode": "flux2-inpaint-img2img",
        "init_view": init_label,
        "n_ref_views": len(ref_pils),
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "identity_tag": identity_tag,
        "n_prompts": len(prompts),
        "elapsed_total_s": round(time.time() - t0, 1),
        "elapsed_per_prompt": elapsed,
        "prompts": all_meta,
    }
    with open(out_dir / "splat_inject_metadata.json", "w") as fh:
        json.dump(inject_meta, fh, indent=2)
    print(f"\ninject done in {inject_meta['elapsed_total_s']}s. wrote {out_dir}")
    return inject_meta
