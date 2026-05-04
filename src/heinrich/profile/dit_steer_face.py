"""Test-time latent optimization with face-encoder identity loss — image as north star.

Mechanism (classifier-style guidance, no LoRA, no training):
  For each denoising step in a chosen window:
    1. Pipe runs DiT forward to update the latent (normal denoising)
    2. callback_on_step_end fires with the updated latent
    3. Decode latent → image (VAE)
    4. Encode image → embedding (CLIP vision)
    5. loss = 1 − cos(current_emb, subject_emb)
    6. ∂L/∂latent via autograd through CLIP + VAE
    7. Update latent: latent = latent − α · ∂L/∂latent
    8. Pipe continues with the modified latent

The reference image acts as a north star — a force pulling each step's trajectory
toward Subject's identity manifold while the prompt continues to govern composition,
pose, and scene. Unlike PAFF-style mask paste, identity is rendered (not collaged):
the model integrates identity progressively through cross-attention to its own
features, guided by gradient pressure rather than spatial overwrite.

Memory: the DiT (12 GB BF16) cannot coexist on a 12 GB GPU with VAE+CLIP+autograd.
Between denoising steps we explicitly evict the DiT to CPU, run VAE+CLIP gradient
work on GPU, restore DiT before the next step.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np


DIT_STEER_FACE_VERSION = "0.1"


def _embed_reference_face(clip_model, clip_processor, ref_image_path: Path, *,
                            device, dtype):
    """Compute the CLIP embedding for the reference image. Returns [1, hidden] tensor."""
    import torch
    from PIL import Image
    img = Image.open(ref_image_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = clip_model(pixel_values=pixel_values)
    emb = out.last_hidden_state[:, 0]  # CLS token: [1, H]
    return emb / emb.norm(dim=-1, keepdim=True)


def _decode_latent_to_clip_input(latent, vae, clip_image_size: int = 224, *,
                                   vae_scaling, vae_shift, clip_mean, clip_std):
    """Decode a latent through VAE and resize to CLIP input. Stays differentiable."""
    import torch
    import torch.nn.functional as F
    # Reverse the VAE preprocessing applied during encoding
    if vae_shift is not None:
        latent_for_decode = latent / vae_scaling + vae_shift
    else:
        latent_for_decode = latent / vae_scaling
    decoded = vae.decode(latent_for_decode).sample  # [B, 3, H, W] in [-1, 1]
    decoded = (decoded.clamp(-1, 1) + 1) / 2  # [0, 1]
    # Resize to CLIP input
    resized = F.interpolate(decoded, size=(clip_image_size, clip_image_size),
                              mode='bilinear', align_corners=False)
    # CLIP normalization
    mean = torch.tensor(clip_mean, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor(clip_std, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    return (resized - mean) / std


def steer_face_during_generation(
    pipeline_id: str,
    ref_image: str,
    prompts_path: str,
    output: str,
    *,
    start_step: int = 22,
    guidance_steps: int = 4,
    learning_rate: float = 0.05,
    n_grad_steps: int = 1,
    cfg_branch_only: bool = True,
    seed: int = 42,
    num_inference_steps: int = 28,
    width: int = 512,
    height: int = 512,
    dtype: str = "bfloat16",
    clip_model_id: str = "openai/clip-vit-large-patch14",
    diagnostic_replace_with_noise: bool = False,
) -> dict:
    """Generate with face-encoder gradient guidance at chosen denoising steps."""
    import torch
    import torch.nn.functional as F
    from diffusers import ZImagePipeline, AutoencoderKL
    from transformers import CLIPVisionModel, CLIPImageProcessor

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_path) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]

    # Phase 1: load CLIP encoder + processor on GPU briefly to embed Subject, then to CPU
    print(f"loading CLIP: {clip_model_id}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    clip_model = CLIPVisionModel.from_pretrained(clip_model_id, torch_dtype=torch_dtype)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
    clip_image_size = clip_model.config.image_size
    clip_mean = clip_processor.image_mean
    clip_std = clip_processor.image_std

    # Phase 2: embed Subject reference (CLIP on GPU briefly, then off)
    print(f"embedding reference: {ref_image}")
    clip_model.to("cuda")
    subject_emb = _embed_reference_face(
        clip_model, clip_processor, Path(ref_image),
        device="cuda", dtype=torch_dtype,
    )
    subject_emb_cpu = subject_emb.cpu()
    clip_model.to("cpu")
    torch.cuda.empty_cache()
    print(f"  subject_emb shape: {tuple(subject_emb_cpu.shape)}, norm: {float(subject_emb_cpu.norm()):.3f}")

    # Phase 3a: load a SEPARATE VAE for gradient work (not under pipe's offload).
    # This is the same weights, but a distinct instance we can move freely between
    # CPU and GPU during the callback. The pipe's own VAE (under sequential offload)
    # is used only for the final pipe-managed decode at the end of generation.
    print(f"loading separate VAE for gradient work")
    vae_for_grad = AutoencoderKL.from_pretrained(pipeline_id, subfolder='vae',
                                                   torch_dtype=torch_dtype)
    vae_for_grad.eval()
    vae_scaling = getattr(vae_for_grad.config, "scaling_factor", 1.0)
    vae_shift = getattr(vae_for_grad.config, "shift_factor", None)

    # Phase 3b: load Z-Image pipe with sequential offload
    print(f"loading {pipeline_id}")
    pipe = ZImagePipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)
    pipe.enable_sequential_cpu_offload()

    # Cache CLIP on GPU permanently, DiT/VAE move via offload hooks during pipe()
    end_step = start_step + guidance_steps
    print(f"face-guidance: steps {start_step}..{end_step - 1}, lr={learning_rate}, "
          f"n_grad_steps={n_grad_steps}")

    all_meta = []
    t0 = time.time()

    for pi, p in enumerate(prompts):
        text = p["text"]
        print(f"\n[{pi + 1}/{len(prompts)}] {text!r}")

        def make_callback():
            def callback(pipe_obj, step_index, timestep, callback_kwargs):
                if not (start_step <= step_index < end_step):
                    return callback_kwargs
                latents = callback_kwargs.get('latents')
                if latents is None:
                    return callback_kwargs

                # DIAGNOSTIC: completely replace latents with noise. If pipe ignores
                # this write, output is unchanged → confirms callback writes are
                # silently discarded by ZImagePipeline.
                if diagnostic_replace_with_noise:
                    g = torch.Generator(device=latents.device).manual_seed(step_index * 1009 + 17)
                    callback_kwargs['latents'] = torch.randn(latents.shape,
                                                              generator=g,
                                                              device=latents.device,
                                                              dtype=latents.dtype)
                    print(f"  step {step_index}: replaced latents with noise "
                          f"(norm before={float(latents.norm()):.2f} "
                          f"after={float(callback_kwargs['latents'].norm()):.2f})")
                    return callback_kwargs

                # Sequential offload streams the DiT layer-by-layer; at this point
                # in the loop most layers are on CPU. Move our separate VAE and CLIP
                # onto GPU for the gradient step. Subject embedding moves with them.
                vae_for_grad.to("cuda")
                clip_model.to("cuda")
                torch.cuda.empty_cache()
                subject_emb_gpu = subject_emb_cpu.to("cuda")

                vae_dtype = next(vae_for_grad.parameters()).dtype
                target = latents[0:1] if cfg_branch_only and latents.shape[0] >= 1 else latents
                target_clone = target.detach().clone().to(device="cuda", dtype=vae_dtype)

                # Pipe runs under torch.no_grad() — re-enable autograd locally for the
                # gradient step. We discard the autograd graph each iteration.
                with torch.enable_grad():
                    for gs in range(n_grad_steps):
                        target_clone = target_clone.detach().requires_grad_(True)

                        clip_input = _decode_latent_to_clip_input(
                            target_clone, vae=vae_for_grad,
                            clip_image_size=clip_image_size,
                            vae_scaling=vae_scaling, vae_shift=vae_shift,
                            clip_mean=clip_mean, clip_std=clip_std,
                        )
                        out = clip_model(pixel_values=clip_input)
                        cur_emb = out.last_hidden_state[:, 0]
                        cur_emb = cur_emb / cur_emb.norm(dim=-1, keepdim=True)
                        sim = (cur_emb * subject_emb_gpu).sum(dim=-1).mean()
                        loss = 1.0 - sim
                        grad = torch.autograd.grad(loss, target_clone)[0]

                        target_clone = (target_clone - learning_rate * grad).detach()

                        if gs == 0:
                            print(f"  step {step_index} grad{gs}: loss={float(loss):.4f} "
                                  f"sim={float(sim):.4f} ||grad||={float(grad.norm()):.4f}")

                with torch.no_grad():
                    if cfg_branch_only and latents.shape[0] > 1:
                        latents = torch.cat(
                            [target_clone.to(latents.dtype).to(latents.device), latents[1:]], dim=0)
                    else:
                        latents = target_clone.to(latents.dtype).to(latents.device)
                    callback_kwargs['latents'] = latents

                # Move our separate VAE + CLIP off GPU before next DiT step
                vae_for_grad.to("cpu")
                clip_model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                return callback_kwargs
            return callback

        try:
            with torch.no_grad():
                result = pipe(
                    text,
                    generator=torch.Generator(device="cuda").manual_seed(seed + pi),
                    num_inference_steps=num_inference_steps,
                    width=width, height=height,
                    callback_on_step_end=make_callback(),
                    callback_on_step_end_tensor_inputs=['latents'],
                )
            image = result.images[0] if hasattr(result, 'images') else None
        except Exception as e:
            print(f"  ERROR during generation: {type(e).__name__}: {e}")
            raise

        if image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")
        all_meta.append({"idx": pi, "text": text,
                          **{k: v for k, v in p.items() if k != "text"}})

    elapsed = time.time() - t0

    metadata = {
        "version": DIT_STEER_FACE_VERSION,
        "pipeline_id": pipeline_id,
        "ref_image": str(ref_image),
        "n_prompts": len(prompts),
        "start_step": start_step,
        "guidance_steps": guidance_steps,
        "learning_rate": learning_rate,
        "n_grad_steps": n_grad_steps,
        "num_inference_steps": num_inference_steps,
        "clip_model": clip_model_id,
        "cfg_branch_only": cfg_branch_only,
        "seed": seed,
        "width": width, "height": height,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for m in all_meta:
            fh.write(json.dumps(m) + "\n")
    print(f"\ndone. {elapsed:.1f}s. wrote {out_dir}")
    return metadata
