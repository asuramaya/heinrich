"""Safetensors file parsing and tensor loading."""
from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import Any
import numpy as np

NUMPY_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_), "F16": np.dtype("<f2"), "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"), "I8": np.dtype("i1"), "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"), "I64": np.dtype("<i8"), "U8": np.dtype("u1"),
}

# FP8/BF16 need torch for decoding
TORCH_DTYPES: dict[str, str] = {
    "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


def parse_safetensors_header(path: Path | str) -> dict[str, Any]:
    """Parse the JSON header of a safetensors file without loading tensors."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated header in: {path}")
    payload = json.loads(header_bytes)
    metadata = payload.pop("__metadata__", {})
    data_start = 8 + header_len
    tensors = []
    for name, meta in payload.items():
        shape = [int(d) for d in meta["shape"]]
        start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        tensors.append({
            "name": name, "dtype": meta["dtype"], "shape": shape,
            "nbytes": end - start, "file_offset": data_start + start,
            "complete": data_start + end <= file_size,
        })
    return {"path": str(path), "file_bytes": file_size, "header_bytes": header_len,
            "metadata": metadata, "tensor_count": len(tensors), "tensors": tensors}


def load_safetensors_tensors(
    path: Path | str,
    *,
    names: set[str] | None = None,
    only_2d: bool = False,
) -> dict[str, np.ndarray]:
    """Load tensors from a safetensors file, casting to float64."""
    path = Path(path)
    header = parse_safetensors_header(path)
    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        for entry in header["tensors"]:
            name = entry["name"]
            if names is not None and name not in names:
                continue
            if not entry["complete"]:
                continue
            f.seek(entry["file_offset"])
            blob = f.read(entry["nbytes"])
            arr = _decode_tensor(blob, entry["dtype"], tuple(entry["shape"]))
            if only_2d and arr.ndim != 2:
                continue
            out[name] = arr
    return out


def load_tensors(path: Path | str, **kwargs) -> dict[str, np.ndarray]:
    """Load tensors from .npz or .safetensors file."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {name: np.array(data[name], dtype=np.float64) for name in data.files}
    if path.suffix == ".safetensors":
        return load_safetensors_tensors(path, **kwargs)
    raise ValueError(f"Unsupported format: {path.suffix}")


def _decode_tensor(blob: bytes, dtype_name: str, shape: tuple[int, ...]) -> np.ndarray:
    count = 1
    for d in shape:
        count *= d
    if dtype_name in NUMPY_DTYPES:
        arr = np.frombuffer(blob, dtype=NUMPY_DTYPES[dtype_name], count=count)
        return arr.reshape(shape).astype(np.float64, copy=False)
    if dtype_name == "BF16":
        return _decode_bfloat16(blob, count).reshape(shape)
    if dtype_name in TORCH_DTYPES:
        return _decode_via_torch(blob, dtype_name, count).reshape(shape)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _decode_bfloat16(blob: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint16, count=count)
    fp32_bytes = np.zeros(count, dtype=np.uint32)
    fp32_bytes[:] = raw.astype(np.uint32) << 16
    return fp32_bytes.view(np.float32).astype(np.float64)


def _decode_via_torch(blob: bytes, dtype_name: str, count: int) -> np.ndarray:
    try:
        import torch
        torch_dtype = getattr(torch, TORCH_DTYPES[dtype_name])
        tensor = torch.frombuffer(bytearray(blob), dtype=torch_dtype)[:count]
        return tensor.float().numpy().astype(np.float64)
    except ImportError:
        raise ImportError(f"torch required for {dtype_name} decoding: pip install heinrich[probe]")
