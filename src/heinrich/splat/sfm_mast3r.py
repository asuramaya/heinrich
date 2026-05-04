"""MASt3R-based SfM backend: subject-agnostic dense matching for unconstrained sets.

Wraps Naver Labs' MASt3R (https://github.com/naver/mast3r) — a learned dense
3D matcher designed for in-the-wild image collections where SIFT-based COLMAP
fails. Given a set of images, MASt3R produces:
  - per-image camera poses (cam2world)
  - per-image intrinsics (pinhole focal + principal point)
  - sparse 3D point cloud with per-point RGB

We convert all of this to the same `sparse` dict format that the COLMAP-based
backend produces, so the rest of the pipeline (gsplat init, training, save)
is shared.

Vendored repo at .vendor/mast3r; checkpoint at
.vendor/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


MAST3R_REPO = Path(__file__).resolve().parents[3] / ".vendor" / "mast3r"
# Local .pth path (Naver direct download, optional). The default is the HF
# repo ID, which uses HF's CDN and downloads the safetensors-format weights.
MAST3R_CHECKPOINT = MAST3R_REPO / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
MAST3R_HF_REPO = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def _ensure_mast3r_paths():
    """Add the vendored MASt3R + DUSt3R + CroCo paths to sys.path."""
    if not MAST3R_REPO.exists():
        raise RuntimeError(
            f"MASt3R repo not found at {MAST3R_REPO}. Clone with:\n"
            f"  cd {MAST3R_REPO.parent} && git clone --recursive https://github.com/naver/mast3r"
        )
    for p in (MAST3R_REPO, MAST3R_REPO / "dust3r", MAST3R_REPO / "dust3r" / "croco"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion [w, x, y, z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def run_mast3r_sfm(
    image_dir: Path,
    work_dir: Path,
    *,
    weights_path: Path | None = None,
    image_size: int = 512,
    device: str = "cuda",
    scene_graph: str = "complete",
) -> tuple[dict, Path]:
    """Run MASt3R sparse global alignment on a directory of images.

    Args:
        image_dir: directory of input images (all formats supported by PIL)
        work_dir: scratch directory for cached intermediate computations
        weights_path: path to MASt3R checkpoint (default: vendored)
        image_size: input resolution for MASt3R (default 512, max-side)
        device: cuda / cpu
        scene_graph: image-pair generation strategy. "complete" pairs all
            images (N(N-1) pairs, expensive but robust). "swin-N" uses a
            sliding window of size N (fast for ordered captures).

    Returns:
        (sparse_dict, staged_image_dir):
            sparse_dict has the same shape that COLMAP's _read_colmap_sparse
            returns: {cameras, images, points, rgb}.
            staged_image_dir contains the input images resized to the
            resolution recorded in `cameras` — feed this dir to _train_gsplat.
    """
    _ensure_mast3r_paths()
    import torch
    from mast3r.model import AsymmetricMASt3R

    # Resolve weights_path. Three modes:
    #   - explicit Path to existing file: load it
    #   - default: HF repo ID (HF Hub auto-download/cache via PyTorchModelHubMixin)
    if weights_path is None:
        # Prefer locally-staged HF safetensors dir (model.safetensors + config.json),
        # else legacy direct .pth, else fall back to HF repo ID for auto-download.
        local_dir = MAST3R_REPO / "checkpoints"
        if (local_dir / "model.safetensors").exists() and (local_dir / "config.json").exists():
            weights_path = str(local_dir)
        elif MAST3R_CHECKPOINT.exists():
            weights_path = str(MAST3R_CHECKPOINT)
        else:
            weights_path = MAST3R_HF_REPO
    weights_path = str(weights_path)
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    work_dir.mkdir(parents=True, exist_ok=True)

    # Discover images in image_dir
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not image_paths:
        raise ValueError(f"no images in {image_dir}")
    image_paths_str = [str(p) for p in image_paths]
    print(f"[mast3r] {len(image_paths_str)} images, image_size={image_size}, scene_graph={scene_graph}")

    # Load model
    print(f"[mast3r] loading model from {weights_path}")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)

    # Load images (returns list of dicts with img tensor + true_shape + idx).
    # square_ok=True preserves the native aspect for square inputs — without
    # it, MASt3R center-crops square images to 4:3 (e.g. 512×384 from
    # 4096×4096), which makes downstream native-resolution rescale impossible
    # because principal points and focals would be in the wrong place
    # relative to native pixels.
    print(f"[mast3r] loading images at {image_size}px (square_ok=True)")
    imgs = load_images(image_paths_str, size=image_size, square_ok=True)

    # Build image pairs
    print(f"[mast3r] building pairs (scene_graph={scene_graph})")
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"[mast3r] {len(pairs)} pairs")

    # Run sparse global alignment
    cache_path = work_dir / "mast3r_cache"
    cache_path.mkdir(exist_ok=True)
    print(f"[mast3r] sparse_global_alignment (cache={cache_path})")
    t0 = time.time()
    scene = sparse_global_alignment(image_paths_str, pairs, str(cache_path), model, device=device)
    print(f"[mast3r] alignment done in {time.time() - t0:.1f}s")

    # Extract scene state
    cam2w = scene.get_im_poses().detach().cpu().numpy()           # [N, 4, 4]
    focals = scene.get_focals().detach().cpu().numpy()             # [N]
    pps = scene.get_principal_points().detach().cpu().numpy()      # [N, 2]
    pts3d_per_view = scene.get_sparse_pts3d()                       # list of tensors
    pts3d_colors_per_view = scene.get_pts3d_colors()                # list of tensors
    image_sizes = [tuple(int(x) for x in im["true_shape"][0]) for im in imgs]  # [(H, W), ...]

    # Build cameras + images metadata
    cameras = {}
    images_meta = []
    for i, p in enumerate(image_paths):
        H, W = image_sizes[i]
        f = float(focals[i])
        cx, cy = float(pps[i, 0]), float(pps[i, 1])
        cameras[i] = {
            "model": "PINHOLE",
            "width": int(W),
            "height": int(H),
            "params": [f, f, cx, cy],
        }
        # cam_from_world = inv(cam2world)
        c2w = cam2w[i]
        # Robust inverse for an SE(3) matrix: R^T, -R^T t
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        qvec = _quat_from_rotmat(R_w2c)
        images_meta.append({
            "name": p.name,
            "qvec": qvec.tolist(),
            "tvec": t_w2c.tolist(),
            "camera_id": i,
        })

    # Aggregate sparse points + colors
    pts_arrays = []
    rgb_arrays = []
    for pts, cols in zip(pts3d_per_view, pts3d_colors_per_view):
        pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
        cols_np = cols.detach().cpu().numpy() if hasattr(cols, "detach") else np.asarray(cols)
        if pts_np.size == 0:
            continue
        # Flatten any extra dims to [P, 3]
        pts_arrays.append(pts_np.reshape(-1, 3))
        rgb_arrays.append(cols_np.reshape(-1, 3))
    if not pts_arrays:
        raise RuntimeError("MASt3R produced no 3D points")
    points = np.concatenate(pts_arrays, axis=0).astype(np.float32)
    rgb = np.concatenate(rgb_arrays, axis=0).astype(np.float32)
    if rgb.max() > 1.5:  # ints in [0, 255]
        rgb = rgb / 255.0
    print(f"[mast3r] {len(image_paths)} cameras, {len(points)} sparse points")

    # Stage images at the recorded MASt3R resolution so the trainer's loader
    # sees images that match the cameras.
    from PIL import Image
    staged_dir = work_dir / "mast3r_images"
    staged_dir.mkdir(exist_ok=True)
    for i, p in enumerate(image_paths):
        im = Image.open(p).convert("RGB")
        H, W = image_sizes[i]
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        im.save(staged_dir / p.name)
    print(f"[mast3r] staged {len(image_paths)} images at scene resolutions → {staged_dir}")

    # Free model
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    sparse = {
        "cameras": cameras,
        "images": images_meta,
        "points": points,
        "rgb": rgb,
    }
    return sparse, staged_dir
