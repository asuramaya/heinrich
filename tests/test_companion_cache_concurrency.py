from __future__ import annotations

import json
import threading
import time

import numpy as np

from heinrich import companion as c


def _run_concurrently(fn, n_threads: int = 6):
    barrier = threading.Barrier(n_threads)
    results = [None] * n_threads
    errors: list[BaseException] = []

    def worker(i: int):
        try:
            barrier.wait()
            results[i] = fn()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def test_get_scores_f32_coalesces_concurrent_cold_loads(tmp_path, monkeypatch):
    decomp = tmp_path / "decomp"
    decomp.mkdir()
    score_path = decomp / "L00_scores.npy"
    np.save(score_path, np.random.randn(32, 16).astype(np.float16))

    with c._score_cache_lock:
        c._score_cache.clear()
        c._score_cache_pending.clear()
        c._score_cache_bytes = 0

    real_load = c.np.load
    target = str(score_path)
    load_count = 0
    load_count_lock = threading.Lock()

    def fake_load(*args, **kwargs):
        nonlocal load_count
        if str(args[0]) == target and kwargs.get("mmap_mode") is None:
            with load_count_lock:
                load_count += 1
            time.sleep(0.05)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(c.np, "load", fake_load)

    results = _run_concurrently(lambda: c._get_scores_f32(str(tmp_path), 0))

    assert load_count == 1
    assert all(r is results[0] for r in results)


def test_get_score_mmap_coalesces_concurrent_cold_loads(tmp_path, monkeypatch):
    decomp = tmp_path / "decomp"
    decomp.mkdir()
    score_path = decomp / "L00_scores.npy"
    np.save(score_path, np.random.randn(24, 8).astype(np.float16))
    (decomp / "meta.json").write_text(json.dumps({"n_layers": 1, "n_real_layers": 1}))

    with c._decomp_score_cache_lock:
        c._decomp_score_cache.clear()
        c._decomp_score_cache_pending.clear()

    real_load = c.np.load
    target = str(score_path)
    load_count = 0
    load_count_lock = threading.Lock()

    def fake_load(*args, **kwargs):
        nonlocal load_count
        if str(args[0]) == target and kwargs.get("mmap_mode") == "r":
            with load_count_lock:
                load_count += 1
            time.sleep(0.05)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(c.np, "load", fake_load)

    results = _run_concurrently(lambda: c._get_score_mmap(str(tmp_path), 0))

    assert load_count == 1
    assert all(r is results[0] for r in results)


def test_get_mlp_mmaps_coalesces_concurrent_cold_loads(tmp_path, monkeypatch):
    mlp_dir = tmp_path / "mlp"
    mlp_dir.mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model": {"n_layers": 2},
        "capture": {"intermediate_size": 4},
    }))

    targets = []
    for layer in range(2):
        gate = mlp_dir / f"L{layer:02d}_gate.npy"
        up = mlp_dir / f"L{layer:02d}_up.npy"
        np.save(gate, np.random.randn(12, 4).astype(np.float16))
        np.save(up, np.random.randn(12, 4).astype(np.float16))
        targets.extend([str(gate), str(up)])

    with c._mlp_mmap_lock:
        c._mlp_mmap_cache.clear()
        c._mlp_mmap_pending.clear()

    real_load = c.np.load
    target_set = set(targets)
    load_count = 0
    load_count_lock = threading.Lock()

    def fake_load(*args, **kwargs):
        nonlocal load_count
        if str(args[0]) in target_set and kwargs.get("mmap_mode") == "r":
            with load_count_lock:
                load_count += 1
            time.sleep(0.02)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(c.np, "load", fake_load)

    results = _run_concurrently(lambda: c._get_mlp_mmaps(str(tmp_path)))

    assert load_count == 4
    assert all(r is results[0] for r in results)
