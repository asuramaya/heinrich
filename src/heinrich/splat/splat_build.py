"""Build a .splat from a directory of images: COLMAP pose recovery + gsplat training.

Output artifact (a .splat is a directory):
    X.splat/
      gaussians.ply       gsplat checkpoint, includes positions, scales, rotations,
                          opacities, SH coefficients (per-Gaussian). Implementation
                          (3DGS vs 2DGS) is recorded in metadata.
      cameras.json        recovered camera poses (intrinsics + extrinsics) for
                          each training image, in COLMAP-compatible format
      sparse_points.ply   COLMAP sparse SfM point cloud (initialization)
      bbox.json           subject bounding box (auto from sparse points)
      metadata.json       impl, n_iterations, source images, training params,
                          timing, version

Subject-agnostic: no person/face priors. Works on any set of images that
satisfies COLMAP's standard requirements (sufficient pose variance, shared
overlap between views, reasonable lighting consistency).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


SPLAT_BUILD_VERSION = "0.1"


def _ensure_ninja_on_path() -> None:
    """gsplat JIT-compiles CUDA kernels via torch's cpp_extension on first use.
    That path subprocesses `ninja --version`, which inherits PATH from the
    parent. The pip-installed `ninja` lives in sys.prefix/bin (the venv bin)
    which isn't automatically on PATH when heinrich is launched by absolute
    path. Use sys.prefix (not Path(sys.executable).resolve(), which follows
    the venv-python symlink to /usr/bin) and prepend.
    """
    import sys
    venv_bin = str(Path(sys.prefix) / "bin")
    current = os.environ.get("PATH", "")
    if venv_bin not in current.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current}"


def _check_deps() -> None:
    """Verify required external tools and Python packages exist. Lazy-imported
    so the module loads cleanly without deps; fails clearly on first invocation.
    """
    _ensure_ninja_on_path()
    missing = []
    if shutil.which("colmap") is None:
        missing.append("colmap (system binary; apt install colmap)")
    try:
        import gsplat  # noqa: F401
    except ImportError:
        missing.append("gsplat (Python; pip install gsplat — needs nvcc on PATH)")
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        missing.append("pycolmap (Python; pip install pycolmap)")
    if missing:
        raise RuntimeError(
            "splat-build is missing dependencies:\n  - "
            + "\n  - ".join(missing)
            + "\n\nInstall them via:\n"
            "  sudo apt-get install -y nvidia-cuda-toolkit colmap\n"
            "  pip install gsplat pycolmap\n\n"
            "Note: gsplat JIT-compiles CUDA kernels on first use; nvcc must be on PATH."
        )


def _preprocess_background_mask(src_dir: Path, dest_dir: Path) -> int:
    """Run rembg over all images in src_dir, write masked versions to dest_dir.

    Background → uniform grey (preserves intensity normalization for SIFT).
    Subject silhouette is preserved with original pixels. Removes the most
    common SfM-killer for character training sets: background variation
    across studio/on-location shots that fragments SIFT matches.
    """
    from rembg import remove, new_session
    from PIL import Image
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)
    print(f"[preprocess] background-mask via rembg on {len(images)} images")
    session = new_session()  # default model: u2net
    n = 0
    for i, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        out_rgba = remove(img, session=session)  # PIL with alpha
        # Composite onto neutral grey to keep stable intensity for SIFT
        bg = Image.new("RGB", out_rgba.size, (128, 128, 128))
        if out_rgba.mode == "RGBA":
            bg.paste(out_rgba, (0, 0), out_rgba.split()[-1])
        else:
            bg = out_rgba.convert("RGB")
        bg.save(dest_dir / p.name)
        n += 1
        if (i + 1) % 10 == 0 or i == len(images) - 1:
            print(f"  [{i+1}/{len(images)}] {p.name}")
    return n


def _stage_images_only(src_dir: Path, dest_dir: Path) -> int:
    """Symlink (or copy if symlink fails) only image files from src to dest.

    COLMAP processes every file in image_path; non-image files (e.g. .txt
    captions in ai-toolkit-style training sets) raise read errors that
    fail SfM. Stage a clean images-only directory.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        target = dest_dir / p.name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(p.resolve())
        except OSError:
            import shutil as _sh
            _sh.copy2(p, target)
        n += 1
    return n


def _run_colmap_sfm(image_dir: Path, work_dir: Path, *, verbose: bool = True) -> Path:
    """Run COLMAP SfM (feature extraction + matching + sparse reconstruction).

    Stages images into work_dir/images_only/ first to filter out non-image
    files (e.g. .txt captions). COLMAP runs on the staged directory.

    Returns: path to the COLMAP sparse model directory.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    images_only = work_dir / "images_only"
    n_staged = _stage_images_only(image_dir, images_only)
    print(f"[colmap] staged {n_staged} image files into {images_only}")
    if n_staged == 0:
        raise RuntimeError(f"no image files found in {image_dir}")

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    quiet_flag = [] if verbose else ["--log_level", "0"]

    # GPU SIFT requires an OpenGL/EGL context (X11/Wayland). Default to CPU
    # SIFT for headless compatibility — slower but reliable.
    # Override with COLMAP_USE_GPU=1 if running with a display.
    use_gpu = "1" if os.environ.get("COLMAP_USE_GPU") == "1" else "0"
    print(f"[colmap] feature_extractor  (use_gpu={use_gpu})")
    # NOTE: dropped --ImageReader.single_camera 1. Heterogeneous resolutions
    # (mixed phone shots, studio shots, etc.) make COLMAP fail under that flag.
    # Letting COLMAP create one camera per unique (model, dimensions) is
    # correct for unconstrained sets and costs nothing.
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_only),
        "--SiftExtraction.use_gpu", use_gpu,
    ] + quiet_flag, check=True)

    print(f"[colmap] exhaustive_matcher  (use_gpu={use_gpu})")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", use_gpu,
    ] + quiet_flag, check=True)

    print("[colmap] mapper (sparse reconstruction; relaxed params for unconstrained sets)")
    # Default thresholds are tuned for 3D-scan datasets; LoRA training sets have
    # less overlap and lower parallax. Relax to register more views; quality
    # degrades gracefully — gsplat will denoise during training.
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_only),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.abs_pose_min_inlier_ratio", "0.15",
        "--Mapper.tri_min_angle", "0.5",
        "--Mapper.filter_min_tri_angle", "0.5",
        "--Mapper.min_num_matches", "10",
    ] + quiet_flag, check=True)

    # COLMAP outputs sparse/0/, sparse/1/, ... — we use the largest reconstruction
    candidates = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    if not candidates:
        raise RuntimeError(
            f"COLMAP produced no sparse reconstruction in {sparse_dir}. "
            "The image set may have insufficient overlap or pose variance."
        )
    # Use whichever sub-reconstruction has the most images
    best = None
    best_n = -1
    for d in candidates:
        images_bin = d / "images.bin"
        if images_bin.exists():
            n = images_bin.stat().st_size  # rough proxy for # images
            if n > best_n:
                best_n = n
                best = d
    if best is None:
        raise RuntimeError(f"No usable sparse reconstruction found in {sparse_dir}")
    print(f"[colmap] using sparse reconstruction: {best}")
    return best


def _read_colmap_sparse(sparse_dir: Path) -> dict:
    """Read COLMAP sparse model into Python structures.

    Returns dict with:
        cameras: dict[camera_id -> {model, width, height, params}]
        images:  list[{name, qvec, tvec, camera_id}]
        points:  numpy [N, 3] of 3D positions; numpy [N, 3] of RGB colors
    """
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse_dir))

    cameras = {}
    for cam_id, cam in rec.cameras.items():
        cameras[cam_id] = {
            "model": cam.model.name,
            "width": cam.width,
            "height": cam.height,
            "params": list(cam.params),
        }

    images = []
    for img_id, img in rec.images.items():
        # pycolmap 4.x: cam_from_world is a method returning Rigid3d
        cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        rotation = cam_from_world.rotation         # Rotation3d
        translation = cam_from_world.translation   # numpy [3]
        # Rotation3d.quat is [x, y, z, w] (scipy convention).
        # We standardize to [w, x, y, z] internally.
        try:
            q_xyzw = np.asarray(rotation.quat)
            qvec = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        except Exception:
            R = np.asarray(rotation.matrix())
            qvec = _rotation_matrix_to_quaternion(R).tolist()
        images.append({
            "name": img.name,
            "qvec": qvec,                           # [w, x, y, z]
            "tvec": [float(v) for v in translation],
            "camera_id": int(img.camera_id),
        })

    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0

    return {"cameras": cameras, "images": images, "points": pts, "rgb": rgb}


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
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


def _save_ply_points(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Save a sparse point cloud as a binary PLY file (sparse_points.ply)."""
    n = len(points)
    has_color = colors is not None
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header += ["end_header"]
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    pts = points.astype(np.float32)
    if has_color:
        cols = np.clip(colors * 255, 0, 255).astype(np.uint8)
        rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = pts
        rec["rgb"] = cols
        with open(path, "wb") as fh:
            fh.write(header_bytes)
            fh.write(rec.tobytes())
    else:
        with open(path, "wb") as fh:
            fh.write(header_bytes)
            fh.write(pts.tobytes())


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] quaternion → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _build_viewmat_from_colmap(qvec, tvec) -> np.ndarray:
    """COLMAP cam_from_world (qvec, tvec) → 4x4 world-to-camera viewmat (numpy)."""
    R = _quat_to_rotmat(np.asarray(qvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _build_intrinsics_from_colmap_camera(cam_meta: dict, scale: float = 1.0):
    """COLMAP camera dict → (K [3x3], width, height) at the given scale.

    Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL (radial distortion
    is ignored; the splat is trained on the rectified pinhole approximation,
    which is fine for sets without aggressive lens distortion).
    """
    model = cam_meta["model"]
    params = cam_meta["params"]
    W, H = int(cam_meta["width"] * scale), int(cam_meta["height"] * scale)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ("PINHOLE", "RADIAL", "OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"unsupported COLMAP camera model: {model}")
    K = np.array([
        [fx * scale, 0, cx * scale],
        [0, fy * scale, cy * scale],
        [0, 0, 1],
    ], dtype=np.float64)
    return K, W, H


def _knn_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to k nearest neighbors per point (rough scale init).

    Returns: [N] array of mean kNN distances.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # +1 because self is included
    return dists[:, 1:].mean(axis=1).astype(np.float32)


def _init_gaussians_from_sparse(points: np.ndarray, rgb: np.ndarray, *,
                                  sh_degree: int = 3, device: str = "cuda") -> dict:
    """Initialize gsplat-format Gaussians from a sparse (points, rgb) cloud.

    Returns dict of torch.nn.Parameters on `device`:
        means      [N, 3]
        scales     [N, 3] log-space (so torch.exp(scales) = real scale)
        quats      [N, 4] (w, x, y, z), unnormalized — gsplat expects normalized
        opacities  [N]    inverse-sigmoid (real opacity = sigmoid(opacities))
        sh0        [N, 1, 3]   SH degree 0 (DC term, captures point color)
        shN        [N, K-1, 3] higher SH bands (zero-init), K = (sh_degree+1)^2
    """
    import torch
    N = len(points)
    if N == 0:
        raise ValueError("cannot initialize Gaussians from empty sparse cloud")
    K = (sh_degree + 1) ** 2

    # Per-point scale = mean kNN distance (clamp to avoid pathological values)
    knn = _knn_distance(points, k=3)
    knn = np.clip(knn, 1e-4, None)
    init_scale = np.log(knn)[:, None].repeat(3, axis=1).astype(np.float32)

    # SH coefficients: only DC term carries the color; rest zero.
    # DC term: f_dc = (rgb - 0.5) / 0.28209479177387814 (3DGS convention,
    # converts from [0, 1] to "spherical harmonic basis coefficient" space)
    SH_C0 = 0.28209479177387814
    sh0 = ((rgb - 0.5) / SH_C0).astype(np.float32)[:, None, :]   # [N, 1, 3]
    shN = np.zeros((N, K - 1, 3), dtype=np.float32)              # [N, K-1, 3]

    # Identity rotation (w=1, xyz=0) for every Gaussian
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Initial opacity = sigmoid(opacity_param) = 0.1
    opacities = np.full(N, np.log(0.1 / 0.9), dtype=np.float32)  # logit(0.1)

    splats = {
        "means": torch.nn.Parameter(torch.from_numpy(points.astype(np.float32)).to(device)),
        "scales": torch.nn.Parameter(torch.from_numpy(init_scale).to(device)),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).to(device)),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities).to(device)),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh0).to(device)),
        "shN": torch.nn.Parameter(torch.from_numpy(shN).to(device)),
    }
    return splats


def _make_optimizers(splats: dict, *, scene_scale: float = 1.0) -> dict:
    """Standard 3DGS LR schedule. Means LR scales with scene size."""
    import torch
    return {
        "means":     torch.optim.Adam([splats["means"]],     lr=1.6e-4 * scene_scale, eps=1e-15),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=5e-3,                  eps=1e-15),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=1e-3,                  eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2,                  eps=1e-15),
        "sh0":       torch.optim.Adam([splats["sh0"]],       lr=2.5e-3,                eps=1e-15),
        "shN":       torch.optim.Adam([splats["shN"]],       lr=2.5e-3 / 20,           eps=1e-15),
    }


def _save_gaussians_ply(splats: dict, path: Path, *, sh_degree: int) -> None:
    """Save gsplat-trained Gaussians to a 3DGS-format PLY (Inria-compatible)."""
    import torch
    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy()  # [N, 1, 3]
    shN = splats["shN"].detach().cpu().numpy()  # [N, K-1, 3]

    N = len(means)
    K_minus_1 = shN.shape[1]

    # 3DGS PLY format columns (Inria convention):
    # x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 f_rest_0 ... f_rest_{3*(K-1)-1}
    # opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
    f_dc = sh0[:, 0, :]                                   # [N, 3]
    f_rest = shN.transpose(0, 2, 1).reshape(N, 3 * K_minus_1)  # [N, 3*(K-1)]
    nx = np.zeros((N, 3), dtype=np.float32)               # placeholder normals

    cols = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    for i in range(3 * K_minus_1):
        cols.append((f"f_rest_{i}", "<f4"))
    cols += [
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    rec = np.empty(N, dtype=cols)
    rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
    rec["nx"], rec["ny"], rec["nz"] = nx[:, 0], nx[:, 1], nx[:, 2]
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(3 * K_minus_1):
        rec[f"f_rest_{i}"] = f_rest[:, i]
    rec["opacity"] = opacities
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3])

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
    for name, _ in cols:
        header.append(f"property float {name}")
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")
    with open(path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(rec.tobytes())


def _load_image_torch(path: Path, scale: float = 1.0, target_size: tuple[int, int] | None = None):
    """Load image as float32 [H, W, 3] in [0, 1].

    If target_size=(W, H) is given, resize directly to those dims (avoids
    holding the full-resolution image in memory). Else apply `scale` to the
    image's native dimensions.
    """
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
    elif scale != 1.0:
        W, H = img.size
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _train_gsplat(
    sparse: dict,
    image_dir: Path,
    *,
    impl: str = "2dgs",
    iterations: int = 7000,
    densification: bool = True,
    seed: int = 42,
    sh_degree: int = 3,
    image_scale: float = 0.5,
    output_ply: Path | None = None,
    ssim_weight: float = 0.2,
) -> dict:
    """Train a Gaussian Splat from COLMAP cameras + sparse points + images.

    Implementation: standard 3DGS / 2DGS training loop using gsplat APIs.
    Memory-aware defaults for 12 GB VRAM (image_scale=0.5 halves activation
    memory; bump to 1.0 on larger GPUs).
    """
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization, rasterization_2dgs, DefaultStrategy
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # SSIM is the standard 3DGS recipe ingredient that prevents the
    # frosted-glass watercolor look from L1-only training. 0.2 weight follows
    # the original paper.
    ssim_fn = None
    if ssim_weight > 0:
        try:
            from pytorch_msssim import ssim as _ssim
            def ssim_fn(pred, gt):
                # pred, gt: [H, W, 3] in [0, 1] → [1, 3, H, W]
                p = pred.permute(2, 0, 1).unsqueeze(0)
                g = gt.permute(2, 0, 1).unsqueeze(0)
                return _ssim(p, g, data_range=1.0, size_average=True)
            print(f"  SSIM loss enabled (weight={ssim_weight})")
        except ImportError:
            print(f"  warning: pytorch_msssim not installed, falling back to L1-only")

    device = "cuda"

    # ---- Prepare per-image training metadata (paths only — lazy-load images) ----
    images_meta = sparse["images"]
    cameras_meta = sparse["cameras"]
    train_data = []
    print(f"  preparing {len(images_meta)} training views (image_scale={image_scale})")
    for img_meta in images_meta:
        cam_meta = cameras_meta[img_meta["camera_id"]]
        K, W, H = _build_intrinsics_from_colmap_camera(cam_meta, scale=image_scale)
        viewmat = _build_viewmat_from_colmap(img_meta["qvec"], img_meta["tvec"])
        img_path = image_dir / img_meta["name"]
        if not img_path.exists():
            print(f"    skipping {img_meta['name']}: file missing")
            continue
        train_data.append({
            "image_path": img_path,
            "viewmat": torch.from_numpy(viewmat).float(),
            "K": torch.from_numpy(K).float(),
            "W": W, "H": H,
        })
    if not train_data:
        raise RuntimeError("no usable training images after camera-image alignment")
    print(f"  prepared {len(train_data)} training views, sample size: {train_data[0]['W']}x{train_data[0]['H']}  (lazy-loaded per step)")

    # ---- Initialize Gaussians from sparse points ----
    print(f"  initializing {len(sparse['points'])} Gaussians from sparse cloud")
    splats = _init_gaussians_from_sparse(
        sparse["points"], sparse["rgb"], sh_degree=sh_degree, device=device)
    optimizers = _make_optimizers(splats, scene_scale=1.0)

    # ---- Densification strategy ----
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=min(15000, iterations - 1000),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    # ---- Training loop ----
    raster_fn = rasterization_2dgs if impl == "2dgs" else rasterization
    print(f"  training: {iterations} iterations, impl={impl}")
    t0 = time.time()
    # Optional small LRU cache keyed by training-view index. Bounded to
    # avoid OOM at high resolutions but speeds up training when the sample
    # rate hits the same view back-to-back.
    from collections import OrderedDict
    img_cache: OrderedDict = OrderedDict()
    img_cache_max = 8  # ~1.5 GB at 4096×4096 fp32

    def _get_view_image(idx: int):
        if idx in img_cache:
            img_cache.move_to_end(idx)
            return img_cache[idx]
        view = train_data[idx]
        gt = _load_image_torch(view["image_path"], target_size=(view["W"], view["H"]))
        if len(img_cache) >= img_cache_max:
            img_cache.popitem(last=False)
        img_cache[idx] = gt
        return gt

    for step in range(iterations):
        idx = int(rng.integers(0, len(train_data)))
        view = train_data[idx]
        gt = _get_view_image(idx).to(device)                  # [H, W, 3]
        viewmat = view["viewmat"].to(device).unsqueeze(0)     # [1, 4, 4]
        Ks = view["K"].to(device).unsqueeze(0)                # [1, 3, 3]

        # Concatenate SH coefficients [N, K, 3]
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]

        # Rasterize
        out = raster_fn(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=view["W"], height=view["H"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB",
        )
        # 2DGS returns more outputs than 3DGS — first element is the rendered
        # image either way.
        if isinstance(out, tuple):
            render = out[0]    # [1, H, W, 3]
            info = out[-1] if isinstance(out[-1], dict) else {}
        else:
            render = out
            info = {}

        render = render[0]  # [H, W, 3]

        # Photometric loss: (1 - ssim_weight) * L1 + ssim_weight * (1 - SSIM)
        l1 = (render - gt).abs().mean()
        if ssim_fn is not None:
            ssim_val = ssim_fn(render.clamp(0, 1), gt)
            loss = (1.0 - ssim_weight) * l1 + ssim_weight * (1.0 - ssim_val)
        else:
            loss = l1

        # Strategy hooks: pre-backward sets up tracking, post-backward does
        # densification/pruning at the right iterations.
        if densification and info:
            strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        loss.backward()
        if densification and info:
            strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 500 == 0 or step == iterations - 1:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{iterations}  loss={float(loss):.4f}  "
                  f"N_gaussians={splats['means'].shape[0]:6d}  elapsed={elapsed:6.1f}s")

    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s; final {splats['means'].shape[0]} Gaussians")

    if output_ply is not None:
        _save_gaussians_ply(splats, output_ply, sh_degree=sh_degree)
        print(f"  wrote {output_ply}")

    return {
        "n_gaussians": int(splats["means"].shape[0]),
        "elapsed_s": round(elapsed, 1),
        "iterations": iterations,
        "impl": impl,
        "sh_degree": sh_degree,
    }


def _rescale_sparse_uniform(sparse: dict, target_max_side: int) -> dict:
    """Uniformly downscale (or upscale) all cameras so the larger of (W, H)
    matches target_max_side. Aspect ratio preserved per camera. 3D points
    are scale-invariant (metric world coords) — only intrinsics + dims change.
    """
    new_cameras = {}
    for cam_id, cam in sparse["cameras"].items():
        W, H = int(cam["width"]), int(cam["height"])
        scale = target_max_side / max(W, H)
        new_W = int(round(W * scale))
        new_H = int(round(H * scale))
        params = cam["params"]
        if cam["model"] in ("PINHOLE", "OPENCV", "RADIAL"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            new_params = [fx * scale, fy * scale, cx * scale, cy * scale] + list(params[4:])
        elif cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = params[0], params[1], params[2]
            new_params = [f * scale, cx * scale, cy * scale] + list(params[3:])
        else:
            raise ValueError(f"unsupported camera model for uniform rescale: {cam['model']}")
        new_cameras[int(cam_id)] = {
            "model": cam["model"],
            "width": new_W,
            "height": new_H,
            "params": new_params,
        }
    return {
        "cameras": new_cameras,
        "images": sparse["images"],
        "points": sparse["points"],
        "rgb": sparse["rgb"],
    }


def _subsample_sparse_points(sparse: dict, max_points: int, seed: int = 42) -> dict:
    """Cap the number of init points by uniform random sub-sampling. Preserves
    points + rgb correspondence."""
    n = len(sparse["points"])
    if n <= max_points:
        return sparse
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return {
        "cameras": sparse["cameras"],
        "images": sparse["images"],
        "points": sparse["points"][idx],
        "rgb": sparse["rgb"][idx],
    }


def _rescale_sparse_to_native(sparse: dict, refs_dir: Path) -> dict:
    """Upscale MASt3R-recorded cameras (typically 512×384) to each image's
    native source resolution. Each input image becomes its own camera id —
    heterogeneous resolutions are handled per-image.

    Aspect-ratio mismatch between the MASt3R camera and the native image
    (which can happen when MASt3R landscape-orients via crop/pad) is
    accepted up to 5%; otherwise warned. Larger mismatches indicate the
    sparse points were triangulated against an aspect MASt3R didn't record.
    """
    from PIL import Image
    new_cameras = {}
    new_images = []
    for img_meta in sparse["images"]:
        name = img_meta["name"]
        native_path = Path(refs_dir) / name
        if not native_path.exists():
            raise ValueError(f"native image {native_path} not found for rescale")
        with Image.open(native_path) as im:
            native_W, native_H = im.size
        cam_id_old = img_meta["camera_id"]
        old_cam = sparse["cameras"][cam_id_old]
        old_W, old_H = int(old_cam["width"]), int(old_cam["height"])
        scale_w = native_W / old_W
        scale_h = native_H / old_H
        ratio = max(scale_w, scale_h) / max(min(scale_w, scale_h), 1e-9)
        if ratio > 1.05:
            print(f"  warning: aspect mismatch for {name}: "
                  f"native {native_W}×{native_H} vs MASt3R {old_W}×{old_H} "
                  f"(scale_w/scale_h={scale_w / scale_h:.3f})")
        # Use scale_w on x and scale_h on y (each axis independently — fx
        # and cx scale with width; fy and cy scale with height).
        old_params = old_cam["params"]
        if old_cam["model"] in ("PINHOLE", "OPENCV", "RADIAL"):
            fx, fy, cx, cy = old_params[0], old_params[1], old_params[2], old_params[3]
            new_params = [fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h]
            new_params += list(old_params[4:])  # any distortion coeffs unchanged
        elif old_cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = old_params[0], old_params[1], old_params[2]
            new_f = f * (scale_w + scale_h) / 2  # average if SIMPLE
            new_params = [new_f, cx * scale_w, cy * scale_h] + list(old_params[3:])
        else:
            raise ValueError(f"unsupported camera model for native rescale: {old_cam['model']}")

        new_cam_id = len(new_cameras)
        new_cameras[new_cam_id] = {
            "model": old_cam["model"],
            "width": int(native_W),
            "height": int(native_H),
            "params": new_params,
        }
        new_images.append({
            "name": name,
            "qvec": img_meta["qvec"],
            "tvec": img_meta["tvec"],
            "camera_id": new_cam_id,
        })
    return {
        "cameras": new_cameras,
        "images": new_images,
        "points": sparse["points"],
        "rgb": sparse["rgb"],
    }


def _load_sparse_from_splat(prior_splat_dir: Path) -> tuple[dict, Path]:
    """Re-use SfM output from a prior .splat for a fresh gsplat training pass.

    Loads cameras.json + sparse_points.ply. Re-discovers the staged image
    directory from the prior splat's _work/ if present, else falls back to
    the recorded refs_dir from metadata.json.

    Returns: (sparse_dict, image_dir)
    """
    cameras_path = prior_splat_dir / "cameras.json"
    sparse_pts_path = prior_splat_dir / "sparse_points.ply"
    metadata_path = prior_splat_dir / "metadata.json"
    if not cameras_path.exists():
        raise ValueError(f"prior splat is missing cameras.json: {prior_splat_dir}")
    if not sparse_pts_path.exists():
        raise ValueError(f"prior splat is missing sparse_points.ply: {prior_splat_dir}")
    with open(cameras_path) as fh:
        cam_data = json.load(fh)
    # metadata.json may be missing (e.g. if a prior build crashed mid-training);
    # the only thing we need from it is the original refs_dir as a fallback for
    # locating training images. Tolerate its absence.
    if metadata_path.exists():
        with open(metadata_path) as fh:
            prior_meta = json.load(fh)
    else:
        prior_meta = {}

    # Read sparse points (we wrote them with x/y/z + uchar rgb in our PLY format)
    # Use the same parser approach as splat_render's _load_gaussians_ply but
    # for the sparse-point columns.
    with open(sparse_pts_path, "rb") as fh:
        # Read header
        header = []
        while True:
            line = fh.readline().decode("ascii", errors="replace").rstrip()
            header.append(line)
            if line == "end_header":
                break
        n = next(int(l.split()[-1]) for l in header if l.startswith("element vertex"))
        has_rgb = any("uchar red" in l for l in header)
        if has_rgb:
            dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
        else:
            dt = np.dtype([("xyz", "<f4", 3)])
        rec = np.fromfile(fh, dtype=dt, count=n)
    points = rec["xyz"].astype(np.float32)
    if has_rgb:
        rgb = rec["rgb"].astype(np.float32) / 255.0
    else:
        rgb = np.full((n, 3), 0.5, dtype=np.float32)

    # Reconstruct the camera dict format _train_gsplat expects: keys are int
    # camera_ids. cam_data["cameras"] keys may be strings (from JSON) → coerce.
    cameras_dict = {int(k): v for k, v in cam_data["cameras"].items()}
    sparse = {
        "cameras": cameras_dict,
        "images": cam_data["images"],
        "points": points,
        "rgb": rgb,
    }

    # Locate the staged training image directory. Prior splat may still have
    # _work/mast3r_images, _work/images_only, or just point at the original refs_dir.
    candidates = [
        prior_splat_dir / "_work" / "mast3r_images",
        prior_splat_dir / "_work" / "images_only",
        Path(prior_meta.get("refs_dir", "")),
    ]
    for c in candidates:
        if c and c.exists() and c.is_dir():
            return sparse, c
    raise ValueError(
        f"could not find any image directory for prior splat {prior_splat_dir}; "
        f"tried _work/mast3r_images, _work/images_only, and recorded refs_dir"
    )


def build_splat(
    refs_dir: str,
    output: str,
    *,
    impl: str = "2dgs",
    iterations: int = 7000,
    work_dir: str | None = None,
    seed: int = 42,
    preprocess: str = "none",
    sfm: str = "colmap",
    reuse_sfm_from: str | None = None,
    ssim_weight: float = 0.2,
    native_resolution: bool = False,
    training_max_side: int | None = None,
    max_init_points: int | None = None,
) -> dict:
    """Build a .splat artifact from a directory of images.

    Args:
        refs_dir: directory of input images (any subject)
        output: path to output .splat directory (will be created)
        impl: "2dgs" (default, surface-forced Gaussians) or "3dgs" (volumetric)
        iterations: gsplat training iterations (default 7000)
        work_dir: scratch directory for COLMAP intermediate files (defaults to
            output/_work, deleted at the end unless KEEP_SPLAT_WORK env var is set)
        seed: RNG seed
    """
    _check_deps()
    if impl not in ("2dgs", "3dgs"):
        raise ValueError(f"impl must be '2dgs' or '3dgs', got {impl!r}")

    refs_path = Path(refs_dir)
    out_path = Path(output)
    if out_path.exists():
        raise ValueError(f"output {out_path} already exists; remove or choose a new path")
    out_path.mkdir(parents=True)

    images = sorted(p for p in refs_path.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not images:
        raise ValueError(f"no images found in {refs_dir}")
    print(f"input: {len(images)} images from {refs_dir}")

    work_path = Path(work_dir) if work_dir else (out_path / "_work")
    work_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    sfm_image_dir = refs_path
    if preprocess == "background-mask":
        print("\n=== Phase 0: preprocessing (background-mask) ===")
        masked_dir = work_path / "masked"
        n_masked = _preprocess_background_mask(refs_path, masked_dir)
        print(f"masked {n_masked} images → {masked_dir}")
        sfm_image_dir = masked_dir
    elif preprocess == "none":
        pass
    else:
        raise ValueError(f"unknown preprocess: {preprocess!r}")

    if reuse_sfm_from:
        print(f"\n=== Phase 1: re-using SfM from {reuse_sfm_from} ===")
        sparse, train_image_dir = _load_sparse_from_splat(Path(reuse_sfm_from))
        # Pass sfm metadata through so downstream knows what backend produced
        # this; tolerate missing metadata.json (e.g. prior build crashed before
        # writing it)
        prior_meta_path = Path(reuse_sfm_from) / "metadata.json"
        if prior_meta_path.exists():
            with open(prior_meta_path) as fh:
                sfm = json.load(fh).get("sfm", sfm)
        # Re-write the artifacts we'll preserve
        from PIL import Image as _Im
        # Just symlink them — gsplat doesn't write them, build_splat does
        for fname in ("sparse_points.ply", "cameras.json"):
            src = Path(reuse_sfm_from) / fname
            dst = out_path / fname
            if src.exists():
                import shutil as _sh
                _sh.copy2(src, dst)
    elif sfm == "colmap":
        print("\n=== Phase 1: COLMAP SfM (camera pose recovery) ===")
        sparse_model_dir = _run_colmap_sfm(sfm_image_dir, work_path)
        sparse = _read_colmap_sparse(sparse_model_dir)
        train_image_dir = sfm_image_dir
    elif sfm == "mast3r":
        print("\n=== Phase 1: MASt3R SfM (learned dense matcher) ===")
        from .sfm_mast3r import run_mast3r_sfm
        sparse, train_image_dir = run_mast3r_sfm(sfm_image_dir, work_path)
    else:
        raise ValueError(f"unknown sfm backend: {sfm!r}")
    n_recovered = len(sparse["images"])
    n_points = len(sparse["points"])
    print(f"recovered {n_recovered}/{len(images)} cameras, {n_points} sparse points")
    if n_recovered < len(images) * 0.5:
        print(f"  warning: only {n_recovered}/{len(images)} images registered. "
              "The set may have insufficient overlap or pose variance.")

    # Save sparse cloud for posterity (used as init for gsplat)
    _save_ply_points(out_path / "sparse_points.ply", sparse["points"], sparse["rgb"])

    # Save cameras.json (our serializable format, COLMAP-equivalent)
    cameras_meta = {
        "cameras": sparse["cameras"],
        "images": sparse["images"],
    }
    with open(out_path / "cameras.json", "w") as fh:
        json.dump(cameras_meta, fh, indent=2)

    # Compute subject bbox from sparse points (rough — just point cloud extents)
    if n_points > 0:
        pts = sparse["points"]
        bbox = {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "center": pts.mean(axis=0).tolist(),
        }
    else:
        bbox = {"min": [-1, -1, -1], "max": [1, 1, 1], "center": [0, 0, 0]}
    with open(out_path / "bbox.json", "w") as fh:
        json.dump(bbox, fh, indent=2)

    if max_init_points is not None and len(sparse["points"]) > max_init_points:
        n_orig = len(sparse["points"])
        sparse = _subsample_sparse_points(sparse, max_init_points, seed=seed)
        print(f"\n=== Phase 1.4: subsampled {n_orig} → {len(sparse['points'])} init points ===")

    if native_resolution:
        print(f"\n=== Phase 1.5: rescaling cameras to native source-image resolution ===")
        sparse = _rescale_sparse_to_native(sparse, refs_path)
        train_image_dir = refs_path  # original images live here
        # Re-record bbox + cameras with new dimensions (sparse_points unchanged
        # — MASt3R produced them in metric world coords, scale-invariant).
        cameras_meta_native = {
            "cameras": sparse["cameras"],
            "images": sparse["images"],
        }
        with open(out_path / "cameras.json", "w") as fh:
            json.dump(cameras_meta_native, fh, indent=2)
        # Show effective training resolution
        first_cam = next(iter(sparse["cameras"].values()))
        print(f"  native resolution: first camera {first_cam['width']}×{first_cam['height']}")

    if training_max_side is not None:
        print(f"\n=== Phase 1.6: capping training resolution at {training_max_side}px max-side ===")
        sparse = _rescale_sparse_uniform(sparse, training_max_side)
        first_cam = next(iter(sparse["cameras"].values()))
        print(f"  capped: first camera {first_cam['width']}×{first_cam['height']}")
        with open(out_path / "cameras.json", "w") as fh:
            json.dump({"cameras": sparse["cameras"], "images": sparse["images"]}, fh, indent=2)

    print(f"\n=== Phase 2: gsplat training ({impl}, {iterations} iterations) ===")
    # image_scale=1.0 whenever cameras have been pre-scaled to match the
    # desired training resolution (MASt3R 512px, native_resolution upscale,
    # explicit training_max_side cap, or any reuse-from-prior path that
    # already wrote the right dims). 0.5 only applies to COLMAP's
    # original-resolution cameras when no resize was done.
    train_image_scale = 1.0 if (sfm == "mast3r" or native_resolution
                                  or training_max_side is not None
                                  or reuse_sfm_from is not None) else 0.5
    gaussians = _train_gsplat(sparse, train_image_dir, impl=impl,
                                 iterations=iterations, seed=seed,
                                 image_scale=train_image_scale,
                                 ssim_weight=ssim_weight,
                                 output_ply=out_path / "gaussians.ply")

    elapsed = time.time() - t0
    metadata = {
        "version": SPLAT_BUILD_VERSION,
        "impl": impl,
        "sfm": sfm,
        "preprocess": preprocess,
        "iterations": iterations,
        "ssim_weight": ssim_weight,
        "seed": seed,
        "n_input_images": len(images),
        "n_registered_cameras": n_recovered,
        "n_sparse_points": n_points,
        "elapsed_s": round(elapsed, 1),
        "refs_dir": str(refs_path),
        "reuse_sfm_from": str(reuse_sfm_from) if reuse_sfm_from else None,
    }
    with open(out_path / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    if not os.environ.get("KEEP_SPLAT_WORK"):
        shutil.rmtree(work_path, ignore_errors=True)

    print(f"\nwrote .splat: {out_path}")
    return metadata
