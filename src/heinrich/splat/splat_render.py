"""Load a .splat and render at a requested camera pose.

The renderer is gsplat's differentiable rasterizer. For inject-mode use
(splat-render → diffusion conditioning), only RGB is needed; depth and
normal channels are also exposed for future ControlNet-style modes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPLAT_RENDER_VERSION = "0.1"


@dataclass
class CameraView:
    """Specification of a camera for splat rendering.

    Coordinates: COLMAP convention (right-down-forward, +Z forward).
    """
    position: tuple[float, float, float]   # camera world position
    look_at: tuple[float, float, float]    # world point camera is aimed at
    up: tuple[float, float, float]         # world up direction (e.g. (0, -1, 0))
    fov_y: float                            # vertical field of view, radians
    width: int                              # render width in pixels
    height: int                             # render height in pixels

    @classmethod
    def canonical_frontal(cls, splat_dir: str, *, width: int = 1024, height: int = 1024,
                            fov_y: float = 0.7, distance_factor: float = 2.5) -> "CameraView":
        """Build a sensible front-facing camera looking at the subject's center.

        Reads the .splat's bbox.json to position the camera in front of the
        subject at a distance proportional to bbox extent.
        """
        with open(Path(splat_dir) / "bbox.json") as fh:
            bbox = json.load(fh)
        center = np.asarray(bbox["center"])
        extent = np.linalg.norm(np.asarray(bbox["max"]) - np.asarray(bbox["min"]))
        distance = float(extent * distance_factor)
        # Camera positioned along world +Z from center
        position = (center[0], center[1], center[2] + distance)
        return cls(position=tuple(position),
                     look_at=tuple(center.tolist()),
                     up=(0.0, -1.0, 0.0),
                     fov_y=fov_y,
                     width=width, height=height)


def _check_deps() -> None:
    try:
        import gsplat  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "splat-render needs gsplat: pip install gsplat (and nvcc on PATH for first-use JIT)"
        )


def _load_gaussians_ply(path: Path, *, device: str = "cuda", sh_degree: int = 3) -> dict:
    """Load 3DGS-format PLY → dict of torch tensors on `device`.

    Reverses _save_gaussians_ply in splat_build.py — same column layout.
    """
    import torch
    K = (sh_degree + 1) ** 2
    with open(path, "rb") as fh:
        header_lines = []
        while True:
            line = fh.readline().decode("ascii", errors="replace").rstrip()
            header_lines.append(line)
            if line == "end_header":
                break
        # parse N
        n = None
        for line in header_lines:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
                break
        if n is None:
            raise ValueError(f"could not parse vertex count from {path}")
        # build same dtype as writer
        cols = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ]
        for i in range(3 * (K - 1)):
            cols.append((f"f_rest_{i}", "<f4"))
        cols += [
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ]
        rec = np.fromfile(fh, dtype=cols, count=n)
    means = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"]], axis=1).astype(np.float32)
    f_rest = np.stack([rec[f"f_rest_{i}"] for i in range(3 * (K - 1))],
                        axis=1).astype(np.float32)
    f_rest = f_rest.reshape(n, 3, K - 1).transpose(0, 2, 1)  # back to [N, K-1, 3]
    opacity = np.asarray(rec["opacity"]).astype(np.float32)
    scales = np.stack([rec["scale_0"], rec["scale_1"], rec["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"]], axis=1).astype(np.float32)

    sh0 = f_dc[:, None, :]  # [N, 1, 3]
    return {
        "means": torch.from_numpy(means).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "opacities": torch.from_numpy(opacity).to(device),
        "sh0": torch.from_numpy(sh0).to(device),
        "shN": torch.from_numpy(f_rest).to(device),
    }


def _viewmat_from_camera_view(view: CameraView) -> np.ndarray:
    """CameraView → 4x4 world-to-camera matrix.

    Build a look-at viewmat: forward = look_at - position; right = forward x up;
    new_up = right x forward. Then construct rotation from these basis vectors
    and combine with translation.
    """
    pos = np.asarray(view.position, dtype=np.float64)
    look = np.asarray(view.look_at, dtype=np.float64)
    up = np.asarray(view.up, dtype=np.float64)

    forward = look - pos
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)
    new_up = np.cross(right, forward)
    new_up = new_up / max(np.linalg.norm(new_up), 1e-8)

    # Camera-from-world: world points expressed in camera frame.
    # Camera frame: x=right, y=down, z=forward (gsplat / COLMAP convention).
    R = np.stack([right, -new_up, forward], axis=0)  # [3, 3] world-to-camera rotation
    t = -R @ pos
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _intrinsics_from_camera_view(view: CameraView) -> np.ndarray:
    """CameraView → 3x3 camera intrinsics K."""
    fy = view.height * 0.5 / np.tan(view.fov_y * 0.5)
    fx = fy   # square pixels
    cx = view.width * 0.5
    cy = view.height * 0.5
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)


def _viewmat_from_colmap_pose(qvec, tvec) -> np.ndarray:
    """COLMAP cam_from_world (qvec [w,x,y,z], tvec) → 4x4 viewmat (numpy)."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return M


def _intrinsics_from_colmap_camera(cam_meta: dict) -> tuple[np.ndarray, int, int]:
    model = cam_meta["model"]
    params = cam_meta["params"]
    W, H = int(cam_meta["width"]), int(cam_meta["height"])
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ("PINHOLE", "RADIAL", "OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"unsupported COLMAP camera model: {model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, W, H


def render_splat(
    splat_dir: str,
    view: CameraView,
    *,
    return_depth: bool = False,
    return_normal: bool = False,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sh_degree: int | None = None,
):
    """Render a splat at the given camera view.

    Returns:
        if return_depth or return_normal:
            (PIL.Image, dict) where dict has 'depth' and/or 'normal' as numpy arrays
        else:
            PIL.Image (RGB)
    """
    _check_deps()
    import torch
    from PIL import Image
    from gsplat import rasterization, rasterization_2dgs

    splat_path = Path(splat_dir)
    with open(splat_path / "metadata.json") as fh:
        meta = json.load(fh)
    impl = meta.get("impl", "2dgs")
    if sh_degree is None:
        sh_degree = meta.get("sh_degree", 3)

    splats = _load_gaussians_ply(splat_path / "gaussians.ply",
                                    device="cuda", sh_degree=sh_degree)
    viewmat = torch.from_numpy(_viewmat_from_camera_view(view)).float().unsqueeze(0).cuda()
    K = torch.from_numpy(_intrinsics_from_camera_view(view)).float().unsqueeze(0).cuda()
    bg = torch.tensor([list(bg_color)], device="cuda").float()  # [1, 3]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    raster_fn = rasterization_2dgs if impl == "2dgs" else rasterization
    render_mode = "RGB+D" if return_depth else "RGB"
    out = raster_fn(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmat, Ks=K,
        width=view.width, height=view.height,
        sh_degree=sh_degree,
        packed=False,
        backgrounds=bg,
        render_mode=render_mode,
    )
    rendered = out[0] if isinstance(out, tuple) else out  # [1, H, W, C]
    img = rendered[0, :, :, :3].clamp(0, 1).detach().cpu().numpy()
    pil = Image.fromarray((img * 255).astype(np.uint8))
    extras = {}
    if return_depth and rendered.shape[-1] >= 4:
        extras["depth"] = rendered[0, :, :, 3].detach().cpu().numpy()
    if return_normal:
        # 2DGS rasterization returns normals; 3DGS doesn't natively.
        if impl == "2dgs" and isinstance(out, tuple) and len(out) > 1:
            try:
                normals = out[1]  # gsplat 2dgs returns (rgb, normals, ...) — index varies
                if hasattr(normals, "shape") and normals.ndim == 4:
                    extras["normal"] = normals[0].detach().cpu().numpy()
            except Exception:
                pass
    if extras:
        return pil, extras
    return pil


def select_diverse_training_views(splat_dir: str, n: int = 6) -> list[int]:
    """Pick `n` training-camera indices spread out around the subject.

    Uses farthest-point sampling on world-space camera positions. Index 0
    is always included (deterministic seed); subsequent picks maximise the
    minimum distance to already-picked cameras. With n >= total cameras,
    returns all indices.
    """
    with open(Path(splat_dir) / "cameras.json") as fh:
        cam_data = json.load(fh)
    images = cam_data["images"]
    if not images:
        raise ValueError(f"{splat_dir} has no training cameras")

    positions = []
    for img in images:
        qvec = np.asarray(img["qvec"], dtype=np.float64)
        tvec = np.asarray(img["tvec"], dtype=np.float64)
        w, x, y, z = qvec
        R_w2c = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])
        cam_pos = -R_w2c.T @ tvec  # c2w translation
        positions.append(cam_pos)
    positions = np.stack(positions, axis=0)
    N = positions.shape[0]
    if n >= N:
        return list(range(N))

    picked = [0]
    dists = np.linalg.norm(positions - positions[0], axis=1)
    for _ in range(n - 1):
        next_idx = int(np.argmax(dists))
        picked.append(next_idx)
        new_dists = np.linalg.norm(positions - positions[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return picked


def render_at_training_camera(splat_dir: str, image_index: int = 0):
    """Render at one of the training-recovered camera poses (for sanity checks)."""
    _check_deps()
    import torch
    from PIL import Image
    from gsplat import rasterization, rasterization_2dgs

    splat_path = Path(splat_dir)
    with open(splat_path / "cameras.json") as fh:
        cam_data = json.load(fh)
    if not cam_data["images"]:
        raise ValueError(f"{splat_dir} has no recovered cameras")
    img_meta = cam_data["images"][image_index]
    cam_meta = cam_data["cameras"][str(img_meta["camera_id"])
                                       if str(img_meta["camera_id"]) in cam_data["cameras"]
                                       else img_meta["camera_id"]]
    K_np, W, H = _intrinsics_from_colmap_camera(cam_meta)
    viewmat_np = _viewmat_from_colmap_pose(img_meta["qvec"], img_meta["tvec"])

    with open(splat_path / "metadata.json") as fh:
        meta = json.load(fh)
    impl = meta.get("impl", "2dgs")
    sh_degree = meta.get("sh_degree", 3)

    splats = _load_gaussians_ply(splat_path / "gaussians.ply",
                                    device="cuda", sh_degree=sh_degree)
    viewmat = torch.from_numpy(viewmat_np).float().unsqueeze(0).cuda()
    K = torch.from_numpy(K_np).float().unsqueeze(0).cuda()
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    raster_fn = rasterization_2dgs if impl == "2dgs" else rasterization
    out = raster_fn(
        means=splats["means"], quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmat, Ks=K, width=W, height=H,
        sh_degree=sh_degree, packed=False, render_mode="RGB",
    )
    rendered = out[0] if isinstance(out, tuple) else out
    img = rendered[0, :, :, :3].clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))
