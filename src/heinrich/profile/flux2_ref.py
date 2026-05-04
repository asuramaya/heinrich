"""FLUX.2 multi-reference generation: native zero-shot character via reference images.

FLUX.2 (Black Forest Labs, Nov 2025) ships with a multimodal text encoder
(Mistral3 + Pixtral processor) that natively accepts up to 10 reference images
alongside the text prompt. Reference images are tokenized and processed by the
multimodal LLM, embedded into the conditioning that flows into the DiT — no
adapter, no LoRA, no per-character training.

Two modes:

  Global: pass a fixed list of reference images, used for every prompt.
    generate_with_flux2_ref(ref_images=[...], ...)

  Paired pool: pass a pool of (image, caption) pairs and a top_k. For each
  generation prompt, the top_k captions most similar to the prompt are picked
  via CLIP-text cosine, and only those reference images are sent to FLUX.2.
  This matches the LOD of the chosen prompt — a "full body on beach" prompt
  pulls full-body refs, a "close-up portrait" prompt pulls headshot refs.
    generate_with_flux2_ref(ref_pool=[(path, caption), ...], top_k=5, ...)

This module is the production-grade counterpart to the dit-replace /
dit-steer-face experiments — those measured why naive activation
transplantation fails for identity; this is what actually works.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


FLUX2_REF_VERSION = "0.4"

FLUX2_MAX_REFS = 10  # FLUX.2 multimodal encoder cap

# Per-prompt ranking blend: how much weight to put on image-vs-prompt vs caption-vs-prompt.
# Image score catches LOD/composition/visual content (does this ref *look* like what was asked?).
# Caption score catches Subject-specific descriptors that aren't visually obvious (vitiligo,
# wardrobe brand, named pose). Default 0.5/0.5.
RANK_IMG_WEIGHT_DEFAULT = 0.5


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


def _load_reference_images(ref_paths: list[str], max_size: int = 1024):
    """Load reference images as PIL.Image objects."""
    from PIL import Image
    images = []
    for p in ref_paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(img)
    return images


def _clip_text_embed(text_out, model):
    """Extract text projection features, robust across transformers versions."""
    if hasattr(text_out, "text_embeds"):
        return text_out.text_embeds
    pooled = text_out.pooler_output
    return model.text_projection(pooled)


def _clip_image_embed(vision_out, model):
    """Extract image projection features, robust across transformers versions."""
    if hasattr(vision_out, "image_embeds"):
        return vision_out.image_embeds
    pooled = vision_out.pooler_output
    return model.visual_projection(pooled)


def _clip_embed_text(texts: list[str], *, processor, model, device) -> np.ndarray:
    """CLIP text tower → L2-normalized embeddings in the joint image-text space, [N, D]."""
    import torch
    with torch.no_grad():
        tok = processor(text=texts, padding=True, truncation=True, max_length=77,
                          return_tensors="pt").to(device)
        text_out = model.text_model(input_ids=tok["input_ids"],
                                       attention_mask=tok.get("attention_mask"))
        emb = _clip_text_embed(text_out, model)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return emb.float().cpu().numpy()


def _clip_embed_images(pil_images, *, processor, model, device, batch_size: int = 16) -> np.ndarray:
    """CLIP vision tower → L2-normalized embeddings in the joint image-text space, [N, D]."""
    import torch
    all_emb = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        with torch.no_grad():
            pix = processor(images=batch, return_tensors="pt").to(device)
            vision_out = model.vision_model(pixel_values=pix["pixel_values"])
            emb = _clip_image_embed(vision_out, model)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        all_emb.append(emb.float().cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def _rank_top_k(
    prompt_emb: np.ndarray,
    pool_img_emb: np.ndarray,
    pool_cap_emb: np.ndarray,
    k: int,
    img_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combined image+caption ranking. Returns (top_k_idx, top_k_score, img_score, cap_score)."""
    sim_img = pool_img_emb @ prompt_emb
    sim_cap = pool_cap_emb @ prompt_emb
    sims = img_weight * sim_img + (1.0 - img_weight) * sim_cap
    order = np.argsort(-sims)
    sel = order[:k]
    return sel, sims[sel], sim_img[sel], sim_cap[sel]


def generate_with_flux2_ref(
    pipeline_id: str,
    prompts_path: str,
    output: str,
    *,
    ref_images: list[str] | None = None,
    ref_pool: list[tuple[str, str]] | None = None,
    top_k: int | None = None,
    ranker_model_id: str = "openai/clip-vit-base-patch32",
    rank_img_weight: float = RANK_IMG_WEIGHT_DEFAULT,
    identity_tag: str | None = None,
    feature_anchors: int = 0,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    dtype: str = "bfloat16",
    offload: str = "sequential",
    max_ref_size: int = 1024,
) -> dict:
    """Generate images conditioned on reference photo(s) + text prompts.

    Exactly one of `ref_images` (global) or `ref_pool` (paired pool with
    per-prompt ranking) must be provided. With `ref_pool`, `top_k` selects
    how many refs to send per prompt (default 5, capped at 10).
    """
    import torch
    # Force-resolve transformers lazy imports BEFORE diffusers tries to load them.
    from transformers import (
        Mistral3ForConditionalGeneration,  # noqa: F401
        PixtralProcessor,  # noqa: F401
        PixtralImageProcessor,  # noqa: F401
        PixtralImageProcessorPil,  # noqa: F401
        PixtralVisionModel,  # noqa: F401
    )
    from diffusers import DiffusionPipeline  # auto-routes to Flux2Pipeline or Flux2KleinPipeline

    if (ref_images is None) == (ref_pool is None):
        raise ValueError("pass exactly one of ref_images (global) or ref_pool (paired pool)")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(prompts_path)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]

    # ----- mode: global -----
    if ref_images is not None:
        if len(ref_images) > FLUX2_MAX_REFS:
            print(f"warning: {len(ref_images)} refs given, FLUX.2 caps at {FLUX2_MAX_REFS} — "
                  f"using first {FLUX2_MAX_REFS}")
            ref_images = ref_images[:FLUX2_MAX_REFS]
        print(f"loading {len(ref_images)} reference image(s)")
        global_pil = _load_reference_images(ref_images, max_size=max_ref_size)
        for r, p in zip(global_pil, ref_images):
            print(f"  {p}  ({r.size[0]}×{r.size[1]})")
        pool_pil = None
        pool_paths = list(ref_images)
        pool_captions = None
        ranker = None
        pool_emb = None
        effective_top_k = None
    # ----- mode: paired pool -----
    else:
        if not ref_pool:
            raise ValueError("ref_pool is empty")
        if top_k is None:
            top_k = 5
        effective_top_k = min(top_k, FLUX2_MAX_REFS, len(ref_pool))
        print(f"paired pool: {len(ref_pool)} refs, selecting top {effective_top_k} per prompt "
              f"by caption-prompt CLIP similarity")
        pool_paths = [str(p) for p, _ in ref_pool]
        pool_captions = [c for _, c in ref_pool]
        print(f"loading {len(pool_paths)} pool image(s)")
        pool_pil = _load_reference_images(pool_paths, max_size=max_ref_size)
        for p, c in zip(pool_paths, pool_captions):
            print(f"  {p}  caption={c[:80]!r}")

        if not (0.0 <= rank_img_weight <= 1.0):
            raise ValueError(f"rank_img_weight must be in [0, 1], got {rank_img_weight}")
        print(f"loading CLIP ranker: {ranker_model_id} "
              f"(image weight={rank_img_weight:.2f}, caption weight={1 - rank_img_weight:.2f})")
        from transformers import CLIPModel, CLIPProcessor
        rank_proc = CLIPProcessor.from_pretrained(ranker_model_id)
        rank_model = CLIPModel.from_pretrained(ranker_model_id, torch_dtype=torch.float32).to("cuda")
        rank_model.eval()
        print("  embedding pool images...")
        pool_img_emb = _clip_embed_images(pool_pil, processor=rank_proc, model=rank_model, device="cuda")
        print("  embedding pool captions...")
        pool_cap_emb = _clip_embed_text(pool_captions, processor=rank_proc, model=rank_model, device="cuda")
        # Identity feature-evidence scoring (independent of per-prompt selection)
        identity_img_sims = None
        if identity_tag and feature_anchors > 0:
            if feature_anchors >= effective_top_k:
                raise ValueError(f"feature_anchors ({feature_anchors}) must be < top_k ({effective_top_k})")
            print(f"  identity tag: {identity_tag!r}  (reserving {feature_anchors} anchor slot(s))")
            id_emb = _clip_embed_text([identity_tag], processor=rank_proc, model=rank_model, device="cuda")[0]
            identity_img_sims = pool_img_emb @ id_emb  # [N]
        elif identity_tag:
            print(f"  identity tag: {identity_tag!r}  (augments prompt only, no anchors)")
        # Free GPU before pipe load — ranker only needed for prompt embedding next
        rank_model.to("cpu")
        torch.cuda.empty_cache()
        ranker = (rank_proc, rank_model)
        global_pil = None

    print(f"loading {pipeline_id} ...")
    pipe = DiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)
    print(f"  pipeline class: {type(pipe).__name__}")

    if lora_path:
        print(f"loading LoRA: {lora_path}  (scale={lora_scale})")
        pipe.load_lora_weights(lora_path)
        if lora_scale != 1.0 and hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(["default_0"], adapter_weights=[lora_scale])
            except Exception as e:
                print(f"  warning: could not set adapter scale ({e}); using default 1.0")

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe = pipe.to("cuda")
    else:
        raise ValueError(f"offload must be 'model'|'sequential'|'none', got {offload!r}")

    print(f"FLUX.2 pipe ready. {len(prompts)} prompts × {num_inference_steps} steps × "
          f"resolution {width}×{height}")
    all_meta = []
    t0 = time.time()

    for pi, p in enumerate(prompts):
        text = p["text"]
        print(f"\n[{pi + 1}/{len(prompts)}] {text!r}")

        # Build the prompt that goes to BOTH the ranker and FLUX.2.
        # Identity tag injection ensures the generation pipe gets explicit feature
        # language (e.g. "vitiligo skin patches") that lifts the model from
        # default-pretty-skin toward training-data-faithful rendering.
        gen_prompt = f"{text}, {identity_tag}" if identity_tag else text
        if identity_tag:
            print(f"  augmented prompt: {gen_prompt!r}")

        if ref_pool is not None:
            rank_proc, rank_model = ranker
            rank_model.to("cuda")
            prompt_emb = _clip_embed_text([gen_prompt], processor=rank_proc, model=rank_model,
                                             device="cuda")[0]
            rank_model.to("cpu")
            torch.cuda.empty_cache()
            scene_k = effective_top_k - feature_anchors
            sel_idx_scene, sc_blend, sc_img, sc_cap = _rank_top_k(
                prompt_emb, pool_img_emb, pool_cap_emb, scene_k, rank_img_weight)
            sel_idx_list = list(sel_idx_scene)
            anchor_idx_list: list[int] = []
            if feature_anchors > 0 and identity_img_sims is not None:
                taken = set(int(i) for i in sel_idx_scene)
                order = np.argsort(-identity_img_sims)
                for i in order:
                    i = int(i)
                    if i in taken:
                        continue
                    anchor_idx_list.append(i)
                    taken.add(i)
                    if len(anchor_idx_list) == feature_anchors:
                        break
            sel_idx_full = sel_idx_list + anchor_idx_list
            refs_pil = [pool_pil[i] for i in sel_idx_full]
            sel_paths = [pool_paths[i] for i in sel_idx_full]
            sel_captions = [pool_captions[i] for i in sel_idx_full]
            print(f"  selected {len(sel_idx_full)} ref(s):  (img|cap → blend)  "
                  f"[{scene_k} scene + {len(anchor_idx_list)} anchor]")
            sel_meta = []
            for j, (path, cap) in enumerate(zip(sel_paths, sel_captions)):
                if j < scene_k:
                    simg = float(sc_img[j]); scap = float(sc_cap[j]); blend = float(sc_blend[j])
                    role = "scene"
                else:
                    pi_idx = sel_idx_full[j]
                    simg = float(identity_img_sims[pi_idx])
                    scap = float("nan")
                    blend = simg
                    role = "anchor"
                marker = "*" if role == "anchor" else " "
                print(f"   {marker}[{simg:+.3f}|{scap:+.3f} → {blend:+.3f}] {Path(path).name}  "
                      f"caption={cap[:60]!r}")
                sel_meta.append({"path": str(path), "caption": cap,
                                   "score": blend, "img_score": simg, "cap_score": scap, "role": role})
            prompt_meta_extra = {"selected_refs": sel_meta}
        else:
            refs_pil = global_pil
            prompt_meta_extra = {}

        gen = torch.Generator(device="cuda").manual_seed(seed + pi)
        try:
            with torch.no_grad():
                result = pipe(
                    image=refs_pil if len(refs_pil) > 1 else refs_pil[0],
                    prompt=gen_prompt,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
            image = result.images[0] if hasattr(result, 'images') else None
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
            raise

        if image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")
        all_meta.append({"idx": pi, "text": text,
                          **{k: v for k, v in p.items() if k != "text"},
                          **prompt_meta_extra})

    elapsed = time.time() - t0
    metadata = {
        "version": FLUX2_REF_VERSION,
        "pipeline_id": pipeline_id,
        "mode": "paired_pool" if ref_pool is not None else "global",
        "n_prompts": len(prompts),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width, "height": height,
        "elapsed_s": round(elapsed, 1),
    }
    if ref_pool is not None:
        metadata["pool_size"] = len(ref_pool)
        metadata["top_k"] = effective_top_k
        metadata["ranker_model"] = ranker_model_id
        metadata["rank_img_weight"] = rank_img_weight
        metadata["identity_tag"] = identity_tag
        metadata["feature_anchors"] = feature_anchors
    else:
        metadata["ref_images"] = [str(p) for p in pool_paths]

    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for m in all_meta:
            fh.write(json.dumps(m) + "\n")
    print(f"\ndone. {elapsed:.1f}s. wrote {out_dir}")
    return metadata


def load_paired_pool(refs_dir: str) -> list[tuple[str, str]]:
    """Find (image_path, caption) pairs in a directory.

    Convention: NAME.png + NAME.txt (matching ai-toolkit / kohya format).
    Files without a matching .txt are skipped — per-prompt ranking requires captions.
    """
    pairs: list[tuple[str, str]] = []
    skipped = []
    for img in sorted(Path(refs_dir).iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        cap = img.with_suffix(".txt")
        if not cap.exists():
            skipped.append(img.name)
            continue
        text = cap.read_text(encoding="utf-8").lstrip("﻿").strip()
        if not text:
            skipped.append(img.name)
            continue
        pairs.append((str(img), text))
    if skipped:
        print(f"  skipped {len(skipped)} file(s) without caption: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return pairs
