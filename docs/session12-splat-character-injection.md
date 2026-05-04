# Session 12 — 1:1 Character Reproduction via 3DGS → FLUX.2 Injection

**Status: frozen.** The pipeline is plumbed end-to-end, the splat is built and converged, the inject CLI accepts the new mode. The strength sweep that decides whether the architecture works was blocked at the moment of freeze by a transient CUDA driver issue (stale `/dev/nvidiactl` after kernel module reload). All commits land *before* the empirical verdict.

## What this line of research was

Goal: from an arbitrary set of images of a subject (here: Subject, who has distinctive vitiligo patches), generate the subject in arbitrary scenes/poses with **fingerprint-level identity fidelity** — including the topology of body markings, not just face.

Two prior architectures hit ceilings:

1. **FLUX.2 multi-reference paired generation** (`flux2-ref` mode). CLIP-ranked top-K refs from a labelled pool. Captured *face*, lost vitiligo. The Mistral3+Pixtral encoder treats refs as non-spatial identity bags — pixel-level body marking topology doesn't survive cross-attention.
2. **Heinrich-native contrastive gradient SVD LoRA** (`profile-discover-character`). Captured the *concept* of vitiligo, lost Subject's specific patch positions. The information-theoretic ceiling: averaging over per-image gradients destroys per-image spatial detail.

The 3DGS thesis (the user's framing): **average into the manifold-that-produced-the-data.** A 3D scene of the subject is the only consistent low-dimensional manifold all training images project from. Reconstruct that manifold once; render it at any view; the variance is the geometry, not the noise. Then condition FLUX.2 generation on the splat-render so the diffusion model fills in lighting/style/scene while spatial layout is preserved.

## What landed in this session

### Pipeline

```
images → MASt3R SfM → gsplat 3DGS optimisation → .splat dir → render → FLUX.2 inject
        (sfm_mast3r.py)  (splat_build.py)                     (splat_render.py)
                                                              (splat_inject.py)
```

### Key decisions / fixes from the session

- **MASt3R replaced COLMAP.** COLMAP's SIFT-based SfM registered 2/48 cameras on the Subject set (appearance variation defeats SIFT). MASt3R's learned dense matcher gave 48/48 cameras + 4.75M sparse points.
- **`square_ok=True` in `dust3r.utils.image.load_images`.** Without it MASt3R center-crops 4096×4096 → 512×384, making downstream native-resolution rescale impossible.
- **MASt3R checkpoint via HF Hub.** Direct Naver host crawls at ~3 MB/s (unfinished after 25 min). HF mirror: 2.75 GB in 110 s.
- **`AsymmetricMASt3R.from_pretrained(directory)` not `(file)`.** Needs both `model.safetensors` and `config.json` co-located.
- **gsplat ninja PATH fix.** `Path(sys.executable).resolve().parent` resolves the venv symlink to `/usr/bin`, missing ninja. Use `sys.prefix/bin` instead. Wired into `_ensure_ninja_on_path()`.
- **pycolmap 4.x quat is `[x,y,z,w]`**, must reorder to `[w,x,y,z]` for our pipeline.
- **gsplat MLX-graph break.** Was building a forward graph across all layers; OOM'd on anything larger than SmolLM2-135M. Per-layer `_to_np()` evaluation freed graph nodes.
- **`train_image_scale` correctness.** Conditional now: `1.0` if (mast3r OR native_resolution OR training_max_side OR reuse_sfm_from), else `0.5`. Prior bug applied 0.5 on top of capped 2048 cameras → trained at 1024 instead of 2048.
- **Native-resolution OOM.** 4096×4096 with 4.75M Gaussians > 12 GB. `--training-max-side 2048` and `--max-init-points 2000000` make it fit.

### Subject splat progression

| Version | Resolution | SfM | Init pts | Iter | Final Gaussians | Loss range | PNG size |
|---------|-----------|-----|---------|------|-----------------|-----------|----------|
| v6 | 512×384 (cropped) | MASt3R square_ok=False | 1.5M | 7000 | 1.2M | 0.04–0.07 (L1) | ~120 KB (frosted) |
| v7 | 512×512 | MASt3R | 1.5M | 7000 | 1.4M | 0.04–0.09 (L1+SSIM) | ~120 KB (recognisable) |
| v9 | 2048×2048 | MASt3R reuse | 2.0M | 7000 | 1.67M | 0.025–0.04 (L1+SSIM) | ~1.4 MB |

v9 stored at `runs/subject_splat/subject_v9_2048.splat/` (gitignored).

### Inject pipeline plumbing

**Three modes** in `splat_inject.py`:

1. `flux2-ref` — splat-render → FLUX.2 multimodal encoder. Now supports `num_ref_views > 1` via farthest-point-sampled diverse training cameras (`select_diverse_training_views`). FLUX.2's documented cap is 10 refs.
2. `flux2-inpaint-img2img` — **the actual right tool, new in this session**. Uses `Flux2KleinInpaintPipeline` from diffusers 0.38 with `mask=ones` (full white), so it degenerates into proper img2img:
   - `image=` splat-render → init latent (spatial layout preserved by partial denoise)
   - `image_reference=` multi-view stack → identity refs through cross-attention
   - `strength` controls schedule truncation
   - `num_inference_steps` × `(1 - strength)` is the actual denoise depth
3. `img2img-strength` — superseded by mode 2.
4. `depth-controlnet` — pending FLUX.2 depth ControlNet availability.

CLI surface:

```bash
heinrich splat-inject \
  --splat <splat_dir> --prompts <jsonl> --output <dir> \
  --mode flux2-inpaint-img2img \
  --strength 0.6 \
  --num-ref-views 6 \
  --init-view training_first \
  --identity-tag subject1
```

### Why we expect this to work (and what's frozen unverified)

The standing two-ceiling problem from prior architectures:

- **Set-variance ceiling** (LoRA): training set has 48 images of Subject; per-image gradient SVD averages spatial detail.
- **Model-prior ceiling** (FLUX.2 ref): cross-attention treats refs as non-spatial concept embeddings.

**3DGS round-trip should escape both.** The splat is the *manifold that produced the images* — averaging there preserves geometry instead of destroying it. The img2img init latent then propagates that geometry pixel-by-pixel into the diffusion process. Ref stack on a separate channel handles identity (face shape, skin tone) without paying for spatial preservation.

**What we did not measure before freezing.** Whether the pixel-spatial preservation is strong enough at any strength sweep point to keep vitiligo topology intact through 30 denoise steps. That is the empirical question `--strength` × `--num-ref-views` was about to answer. The sweep design (queued):

```
strength ∈ {0.4, 0.6, 0.8} × num_ref_views ∈ {1, 6} × 8-prompt stress test
= 6 runs × 8 prompts ≈ 30–60 min on A3000-12GB at sequential offload
```

## What this line of research suggests next, when unfrozen

- **First**: run the queued strength × ref-views sweep against `runs/subject_splat/subject_v9_2048.splat/`. Stress prompts at `src/heinrich/eval/prompt_data/flux2_subject_stress_test.jsonl`.
- **If vitiligo survives** at one strength: declare the architecture, move to scale (more subjects, automated pipeline).
- **If vitiligo doesn't survive even at strength=0.4**: the bottleneck is splat fidelity, not the conditioning channel. Bump 3DGS to 30000 iterations + sh_degree=4, or switch to 2DGS for surface-aligned splatting.
- **If face holds but body fails**: the ref encoder is dominated by face features. Try multi-view refs that emphasise body crops, or add identity prompts that *describe* the markings ("vitiligo patches on the legs in image 1, image 3").
- **Coordinate-system bug**: `_viewmat_from_camera_view` has an OpenGL/OpenCV mismatch that makes `canonical_frontal` renders crystal-shaped. Workaround is `--init-view training_first`. Fix is straightforward but unverified.

## Files added or substantially changed

```
src/heinrich/splat/                        # new subpackage
  __init__.py
  splat_build.py        — COLMAP + MASt3R-driven 3DGS optimisation
  splat_render.py       — load .splat, render at requested camera
                          (+ select_diverse_training_views FPS)
  splat_inject.py       — three injection modes (flux2-ref,
                          flux2-inpaint-img2img, img2img-strength)
  sfm_mast3r.py         — MASt3R SfM wrapper (vendored repo at .vendor/mast3r)

src/heinrich/profile/
  flux2_ref.py          — FLUX.2 multi-ref pipeline (CLIP image+text ranker,
                          identity-tag, feature-anchor ref selection)
  character_direction.py — null-conditional contrastive gradient SVD LoRA
                          discovery (frozen ceiling-1 result)

src/heinrich/cli.py     — added: profile-discover-character, splat-build,
                          splat-render, splat-inject (with --num-ref-views,
                          --init-view, --strength, etc.)

pyproject.toml          — splat extra: gsplat>=1.5, pycolmap>=4.0,
                          pytorch_msssim, rembg

src/heinrich/eval/prompt_data/
  flux2_subject_stress_test.jsonl  — 8-prompt OOD identity stress set
```

## Reproduction (when GPU is back)

```bash
# 1. Verify GPU
.venv/bin/python -c "import torch; assert torch.cuda.is_available()"

# 2. (Re-)build splat — only if not already at runs/subject_splat/subject_v9_2048.splat
heinrich splat-build \
  --images <dataset_dir> \
  --output runs/subject_splat/subject_v9_2048.splat \
  --sfm mast3r --impl 3dgs \
  --max-init-points 2000000 --training-max-side 2048 \
  --ssim-weight 0.2 --iterations 7000

# 3. Run E2 sweep
for s in 0.4 0.6 0.8; do
  for r in 1 6; do
    heinrich splat-inject \
      --splat runs/subject_splat/subject_v9_2048.splat \
      --prompts src/heinrich/eval/prompt_data/flux2_subject_stress_test.jsonl \
      --output runs/subject_inject/v9_inpaint_s${s}_r${r} \
      --mode flux2-inpaint-img2img \
      --strength $s --num-ref-views $r \
      --init-view training_first \
      --identity-tag subject1
  done
done

# 4. Compare against existing flux2-ref baseline at runs/flux2_sofie_paired_top5/
```
