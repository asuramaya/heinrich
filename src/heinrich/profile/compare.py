"""Compare .frt, .shrt, .sht profiles across models or stages.

Three analysis modes:
  chain   — connect .frt → .shrt → .sht for one model (three-stage correlation)
  cross   — compare .shrt files across two models (rank correlation)
  inspect — summarize a single profile file
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def chain(frt_path: str, shrt_path: str, sht_path: str) -> dict:
    """Connect .frt → .shrt → .sht for one model.

    Returns correlation matrix, per-script breakdown, and chain statistics.
    """
    from .frt import load_frt
    from .shrt import load_shrt
    from .sht import load_sht

    frt = load_frt(frt_path)
    shrt = load_shrt(shrt_path)
    sht = load_sht(sht_path)

    # Build lookups
    fl = {int(frt['token_ids'][i]): (int(frt['byte_counts'][i]), str(frt['scripts'][i]))
          for i in range(len(frt['token_ids']))}
    sl = {int(shrt['token_ids'][i]): float(shrt['deltas'][i])
          for i in range(len(shrt['token_ids']))}
    tl = {int(sht['token_ids'][i]): (float(sht['kl_divs'][i]), float(sht['entropies'][i]))
          for i in range(len(sht['token_ids']))}

    shared = set(fl) & set(sl) & set(tl)
    if len(shared) < 10:
        return {"error": f"Only {len(shared)} shared tokens across all three profiles"}

    ba, da, ka, ea, sa = [], [], [], [], []
    for t in shared:
        ba.append(fl[t][0]); sa.append(fl[t][1])
        da.append(sl[t]); ka.append(tl[t][0]); ea.append(tl[t][1])

    ba = np.array(ba, dtype=np.float32)
    da = np.array(da, dtype=np.float32)
    ka = np.array(ka, dtype=np.float32)
    ea = np.array(ea, dtype=np.float32)

    c = lambda x, y: round(float(np.corrcoef(x, y)[0, 1]), 4)

    # Per-script breakdown
    by_script = defaultdict(lambda: ([], []))
    for i in range(len(sa)):
        by_script[sa[i]][0].append(da[i])
        by_script[sa[i]][1].append(ka[i])

    scripts = {}
    for s in sorted(by_script, key=lambda x: -np.mean(by_script[x][0])):
        dd, kk = by_script[s]
        if len(dd) >= 10:
            scripts[s] = {
                "n": len(dd),
                "mean_delta": round(float(np.mean(dd)), 1),
                "mean_kl": round(float(np.mean(kk)), 2),
                "delta_ci": round(1.96 * float(np.std(dd)) / np.sqrt(len(dd)), 1),
                "kl_ci": round(1.96 * float(np.std(kk)) / np.sqrt(len(kk)), 2),
            }

    return {
        "n_shared": len(shared),
        "correlations": {
            "bytes_to_delta": c(ba, da),
            "delta_to_kl": c(da, ka),
            "bytes_to_kl": c(ba, ka),
            "delta_to_entropy": c(da, ea),
            "kl_to_entropy": c(ka, ea),
        },
        "scripts": scripts,
        "shrt_baseline_entropy": shrt['metadata']['baseline']['entropy'],
        "sht_baseline_entropy": sht['metadata']['baseline']['entropy'],
    }


def cross(shrt_a_path: str, shrt_b_path: str, frt_path: str | None = None) -> dict:
    """Compare two .shrt files (potentially different models).

    Computes rank correlation on shared tokens. If frt_path is provided,
    breaks down by script. Flags baseline mismatches.
    """
    from .shrt import load_shrt

    a = load_shrt(shrt_a_path)
    b = load_shrt(shrt_b_path)

    al = {int(a['token_ids'][i]): float(a['deltas'][i])
          for i in range(len(a['token_ids']))}
    bl = {int(b['token_ids'][i]): float(b['deltas'][i])
          for i in range(len(b['token_ids']))}

    shared = sorted(set(al) & set(bl))
    if len(shared) < 10:
        return {"error": f"Only {len(shared)} shared tokens"}

    da = np.array([al[t] for t in shared])
    db = np.array([bl[t] for t in shared])

    r_delta = round(float(np.corrcoef(da, db)[0, 1]), 4)
    r_rank = round(float(np.corrcoef(
        np.argsort(np.argsort(-da)),
        np.argsort(np.argsort(-db))
    )[0, 1]), 4)

    h_a = a['metadata']['model']['hidden_size']
    h_b = b['metadata']['model']['hidden_size']

    result = {
        "model_a": a['metadata']['model']['name'],
        "model_b": b['metadata']['model']['name'],
        "n_shared": len(shared),
        "delta_correlation": r_delta,
        "rank_correlation": r_rank,
        "baseline_a": {
            "entropy": a['metadata']['baseline']['entropy'],
            "layer": a['metadata']['baseline']['layer'],
            "top_token": a['metadata']['baseline']['top_token'],
        },
        "baseline_b": {
            "entropy": b['metadata']['baseline']['entropy'],
            "layer": b['metadata']['baseline']['layer'],
            "top_token": b['metadata']['baseline']['top_token'],
        },
        "baseline_match": abs(
            a['metadata']['baseline']['entropy'] - b['metadata']['baseline']['entropy']
        ) < 0.5,
        "mean_a": round(float(da.mean()), 1),
        "mean_b": round(float(db.mean()), 1),
        # Normalized sensitivity: delta / hidden_size
        # Measures how disturbed the model is per dimension
        "sensitivity_a": round(float(da.mean()) / h_a, 4),
        "sensitivity_b": round(float(db.mean()) / h_b, 4),
        "sensitivity_ratio": round(
            (float(da.mean()) / h_a) / max(float(db.mean()) / h_b, 1e-8), 2),
    }

    # Per-script breakdown if frt available
    if frt_path:
        from .frt import load_frt
        frt = load_frt(frt_path)
        fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
              for i in range(len(frt['token_ids']))}

        by_script = defaultdict(lambda: ([], []))
        for t in shared:
            s = fl.get(t, 'unknown')
            by_script[s][0].append(al[t])
            by_script[s][1].append(bl[t])

        scripts = {}
        for s in sorted(by_script, key=lambda x: -np.mean(by_script[x][0])):
            d0, d1 = by_script[s]
            if len(d0) >= 5:
                scripts[s] = {
                    "n": len(d0),
                    "mean_a": round(float(np.mean(d0)), 1),
                    "mean_b": round(float(np.mean(d1)), 1),
                    "ratio": round(float(np.mean(d1)) / max(float(np.mean(d0)), 0.01), 2),
                }
        result["scripts"] = scripts

    return result


def survey(shrt_paths: list[str], frt_paths: list[str] | None = None) -> dict:
    """Compare multiple .shrt profiles. Baseline-independent: uses within-model
    rank and relative script displacement, not absolute deltas.

    Args:
        shrt_paths: list of .shrt.npz file paths
        frt_paths: optional list of matching .frt.npz paths (for script breakdown)

    Returns dict with per-model summaries, concordance, and script rankings.
    """
    from .shrt import load_shrt

    profiles = []
    for i, path in enumerate(shrt_paths):
        s = load_shrt(path)
        m = s['metadata']
        h = m['model']['hidden_size']
        deltas = s['deltas'].astype(np.float32)

        # Within-model script breakdown (if frt available)
        scripts_breakdown = {}
        if frt_paths and i < len(frt_paths):
            from .frt import load_frt
            frt = load_frt(frt_paths[i])
            fl = {int(frt['token_ids'][j]): (str(frt['scripts'][j]), int(frt['byte_counts'][j]))
                  for j in range(len(frt['token_ids']))}

            by_script = defaultdict(lambda: {'deltas': [], 'dpb': []})
            for j in range(len(s['token_ids'])):
                tid = int(s['token_ids'][j])
                if tid not in fl:
                    continue
                sc, byte_count = fl[tid]
                if sc in ('special', 'unknown'):
                    continue
                d = float(deltas[j])
                by_script[sc]['deltas'].append(d)
                if byte_count > 0:
                    by_script[sc]['dpb'].append(d / byte_count)

            grand_mean = float(deltas.mean())
            all_dpb = []
            for data in by_script.values():
                all_dpb.extend(data['dpb'])
            grand_dpb = float(np.mean(all_dpb)) if all_dpb else 1.0

            for sc, data in by_script.items():
                if len(data['deltas']) >= 5:
                    mean_dpb = float(np.mean(data['dpb'])) if data['dpb'] else 0
                    scripts_breakdown[sc] = {
                        "n": len(data['deltas']),
                        "mean_delta": round(float(np.mean(data['deltas'])), 1),
                        "relative": round(float(np.mean(data['deltas'])) / max(grand_mean, 1e-8), 3),
                        "relative_per_byte": round(mean_dpb / max(grand_dpb, 1e-8), 3),
                    }

        profiles.append({
            "path": path,
            "model": m['model']['name'],
            "hidden_size": h,
            "n_layers": m['model']['n_layers'],
            "n_sampled": m['index']['n_sampled'],
            "baseline_entropy": m['baseline']['entropy'],
            "baseline_top": m['baseline']['top_token'],
            "mean_delta": round(float(deltas.mean()), 1),
            "sensitivity": round(float(deltas.mean()) / h, 4),
            "scripts": scripts_breakdown,
        })

    # Script concordance: do models agree on which scripts are hard?
    # Use relative displacement (baseline-independent)
    concordance = {}
    if any(p['scripts'] for p in profiles):
        all_scripts = set()
        for p in profiles:
            all_scripts.update(p['scripts'].keys())

        # For each script: collect within-model relative values (raw delta only)
        for sc in sorted(all_scripts):
            relatives = []
            for p in profiles:
                if sc in p['scripts']:
                    relatives.append(p['scripts'][sc]['relative'])
            if len(relatives) >= 2:
                concordance[sc] = {
                    "n_models": len(relatives),
                    "mean_relative": round(float(np.mean(relatives)), 3),
                    "std_relative": round(float(np.std(relatives)), 3),
                    "min": round(float(np.min(relatives)), 3),
                    "max": round(float(np.max(relatives)), 3),
                    "consistent": float(np.std(relatives)) < 0.15,
                }

        # Kendall's W across models that have script data
        models_with_scripts = [p for p in profiles if p['scripts']]
        common = set.intersection(*[set(p['scripts'].keys()) for p in models_with_scripts])
        if len(common) >= 3 and len(models_with_scripts) >= 2:
            k = len(models_with_scripts)
            n = len(common)
            common_sorted = sorted(common)

            # Build rank matrix: [n_scripts, n_models]
            rank_matrix = np.zeros((n, k))
            for j, p in enumerate(models_with_scripts):
                rels = [p['scripts'][sc]['relative'] for sc in common_sorted]
                rank_matrix[:, j] = np.argsort(np.argsort([-r for r in rels]))

            R = rank_matrix.sum(axis=1)
            S = float(np.sum((R - R.mean())**2))
            W = 12 * S / (k**2 * (n**3 - n)) if n > 1 else 0
        else:
            W = None
    else:
        W = None

    return {
        "n_models": len(profiles),
        "profiles": profiles,
        "concordance": concordance,
        "kendalls_w": round(W, 4) if W is not None else None,
    }


def mismatch(shrt_path: str, frt_path: str) -> dict:
    """Compute the tokenizer-weight mismatch for a single model.

    Accounts for tokenizer contribution by reporting delta per byte
    alongside raw delta. Longer tokens naturally displace more — delta
    per byte factors out the tokenizer's encoding decisions.

    Returns per-script allocation, displacement, delta/byte, and mismatch.
    """
    from .shrt import load_shrt
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)

    h = shrt['metadata']['model']['hidden_size']

    # Tokenizer properties per token
    frt_data = {}
    for i in range(len(frt['token_ids'])):
        frt_data[int(frt['token_ids'][i])] = {
            'script': str(frt['scripts'][i]),
            'bytes': int(frt['byte_counts'][i]),
        }

    frt_scripts = [v['script'] for v in frt_data.values()]
    total_real = sum(1 for s in frt_scripts if s not in ('special', 'unknown'))
    from collections import Counter
    script_counts = Counter(s for s in frt_scripts if s not in ('special', 'unknown'))
    allocation = {s: n / total_real for s, n in script_counts.items()}

    deltas = shrt['deltas'].astype(np.float32)
    grand_mean = float(deltas.mean())

    by_script = defaultdict(lambda: {'deltas': [], 'bytes': [], 'dpb': []})
    for i in range(len(shrt['token_ids'])):
        tid = int(shrt['token_ids'][i])
        if tid not in frt_data:
            continue
        fd = frt_data[tid]
        s = fd['script']
        if s in ('special', 'unknown'):
            continue
        d = float(deltas[i])
        b = fd['bytes']
        by_script[s]['deltas'].append(d)
        by_script[s]['bytes'].append(b)
        if b > 0:
            by_script[s]['dpb'].append(d / b)

    # Grand mean delta per byte for normalization
    all_dpb = []
    for s, data in by_script.items():
        all_dpb.extend(data['dpb'])
    grand_dpb = float(np.mean(all_dpb)) if all_dpb else 1.0

    scripts = {}
    for s in sorted(by_script):
        data = by_script[s]
        if len(data['deltas']) >= 5:
            mean_d = float(np.mean(data['deltas']))
            mean_b = float(np.mean(data['bytes']))
            mean_dpb = float(np.mean(data['dpb'])) if data['dpb'] else 0
            rel_raw = mean_d / max(grand_mean, 1e-8)
            rel_dpb = mean_dpb / max(grand_dpb, 1e-8)
            alloc = allocation.get(s, 0)

            scripts[s] = {
                "n_tokens_vocab": script_counts.get(s, 0),
                "allocation_pct": round(alloc * 100, 2),
                "n_measured": len(data['deltas']),
                "mean_delta": round(mean_d, 1),
                "mean_bytes": round(mean_b, 1),
                "delta_per_byte": round(mean_dpb, 2),
                "relative_raw": round(rel_raw, 3),
                "relative_per_byte": round(rel_dpb, 3),
            }

    return {
        "model": shrt['metadata']['model']['name'],
        "hidden_size": h,
        "vocab_size": frt['metadata']['tokenizer']['vocab_size'],
        "sensitivity": round(grand_mean / h, 4),
        "grand_delta_per_byte": round(grand_dpb, 2),
        "scripts": scripts,
    }


def depth_compare(shrt_paths: list[str], frt_paths: list[str] | None = None,
                   checkpoints: list[float] | None = None) -> dict:
    """Compare models at relative depth, not absolute layer.

    Normalizes by layer/n_layers and hidden_size so models with
    different architectures are comparable. If frt_paths provided,
    includes per-script layer trajectories and explosion detection.
    """
    from .shrt import load_shrt

    if checkpoints is None:
        checkpoints = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    profiles = []
    for i, path in enumerate(shrt_paths):
        s = load_shrt(path)
        m = s['metadata']
        if 'layer_stats' not in m:
            return {"error": f"{path}: no layer_stats — run with --layers all"}
        n = len(m['layers'])
        h = m['model']['hidden_size']
        layers_list = m['layers']

        depth_data = {}
        for pct in checkpoints:
            target = min(int(pct * (n - 1)), n - 1)
            ls = m['layer_stats'].get(str(layers_list[target]))
            if ls:
                depth_data[pct] = {
                    "layer": layers_list[target],
                    "norm_delta": round(ls['mean'] / h, 4),
                    "cv": round(ls['cv'], 4),
                    "raw_mean": round(ls['mean'], 2),
                }

        # Per-layer jump detection
        jumps = []
        prev_mean = 0
        for layer in layers_list:
            ls = m['layer_stats'].get(str(layer), {})
            mean = ls.get('mean', 0)
            if prev_mean > 0.1:
                jump_pct = (mean - prev_mean) / prev_mean * 100
                if jump_pct > 40:
                    jumps.append({"layer": layer, "jump_pct": round(jump_pct, 1),
                                  "from": round(prev_mean, 2), "to": round(mean, 2)})
            prev_mean = mean

        profile = {
            "path": path,
            "model": m['model']['name'],
            "n_layers": n,
            "hidden_size": h,
            "depths": depth_data,
            "jumps": jumps,
        }

        # Per-script layer analysis (needs frt + per-layer delta arrays in npz)
        if frt_paths and i < len(frt_paths) and len(layers_list) > 1:
            from .frt import load_frt
            frt = load_frt(frt_paths[i])
            fl = {int(frt['token_ids'][j]): str(frt['scripts'][j])
                  for j in range(len(frt['token_ids']))}

            # Check for per-layer delta arrays
            has_layer_deltas = any(
                f"deltas_L{l}" in s for l in layers_list[:2])

            if has_layer_deltas:
                token_ids = s['token_ids']
                scripts = [fl.get(int(tid), 'unknown') for tid in token_ids]

                # Compare second-to-last vs last layer per script
                second_last = layers_list[-2]
                last = layers_list[-1]
                key_sl = f"deltas_L{second_last}"
                key_l = f"deltas_L{last}"

                if key_sl in s and key_l in s:
                    d_sl = s[key_sl].astype(np.float32)
                    d_l = s[key_l].astype(np.float32)
                    ratio = d_l / np.maximum(d_sl, 0.01)

                    by_script = defaultdict(list)
                    for j, sc in enumerate(scripts):
                        if sc not in ('special', 'unknown'):
                            by_script[sc].append(float(ratio[j]))

                    # Count vocab tokens per script for ceiling
                    from collections import Counter
                    frt_all_scripts = [str(frt['scripts'][j]) for j in range(len(frt['scripts']))]
                    vocab_ceiling = Counter(s for s in frt_all_scripts if s not in ('special', 'unknown'))

                    script_explosions = {}
                    for sc in sorted(by_script, key=lambda x: -np.mean(by_script[x])):
                        vals = by_script[sc]
                        if len(vals) >= 3:
                            mean_r = float(np.mean(vals))
                            std_r = float(np.std(vals))
                            se = std_r / np.sqrt(len(vals))
                            ceiling = vocab_ceiling.get(sc, 0)
                            script_explosions[sc] = {
                                "n": len(vals),
                                "vocab_ceiling": ceiling,
                                "pct_sampled": round(len(vals) / max(ceiling, 1) * 100, 0),
                                "mean_ratio": round(mean_r, 2),
                                "std_ratio": round(std_r, 2),
                                "ci_95": round(1.96 * se, 2),
                                "provisional": len(vals) < 30,
                                "exhaustible": ceiling < 50,
                            }

                    profile["final_layer_explosion"] = {
                        "from_layer": second_last,
                        "to_layer": last,
                        "overall_ratio": round(float(ratio.mean()), 2),
                        "by_script": script_explosions,
                    }

        profiles.append(profile)

    return {"checkpoints": checkpoints, "profiles": profiles}


def inspect_profile(path: str) -> dict:
    """Summarize any profile file (.frt, .shrt, .sht)."""
    p = Path(path)
    if '.frt' in p.name:
        from .frt import load_frt
        d = load_frt(path)
        return d['metadata']
    elif '.shrt' in p.name:
        from .shrt import load_shrt
        d = load_shrt(path)
        return d['metadata']
    elif '.sht' in p.name:
        from .sht import load_sht
        d = load_sht(path)
        return d['metadata']
    else:
        return {"error": f"Unknown profile type: {p.name}"}
