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

        # Displacement dimensionality from vectors
        vectors = s.get('vectors')
        dim_fingerprint = {}
        if vectors is not None and len(vectors) >= 20:
            vecs = vectors.astype(np.float32)
            # Filter to non-special tokens
            if frt_paths and i < len(frt_paths):
                keep = [j for j in range(len(s['token_ids']))
                        if fl.get(int(s['token_ids'][j]), ('unknown',))[0]
                        not in ('special', 'unknown')]
            else:
                keep = list(range(len(vecs)))
            if len(keep) >= 20:
                vecs_filtered = vecs[keep]
                centered = vecs_filtered - vecs_filtered.mean(axis=0)
                _, S_vals, _ = np.linalg.svd(centered, full_matrices=False)
                variance = S_vals ** 2
                total_var = float(variance.sum())
                cumulative = np.cumsum(variance) / total_var

                dim_fingerprint = {
                    "pc1_pct": round(float(cumulative[0]) * 100, 1),
                    "pc2_pct": round(float(variance[1] / total_var) * 100, 1) if len(variance) > 1 else 0,
                    "pcs_50": int(np.searchsorted(cumulative, 0.5)) + 1,
                    "pcs_80": int(np.searchsorted(cumulative, 0.8)) + 1,
                    "pcs_90": int(np.searchsorted(cumulative, 0.9)) + 1,
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
            "dimensionality": dim_fingerprint,
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
                    "std_below_0.15": float(np.std(relatives)) < 0.15,
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


def layer_scripts(shrt_path: str, frt_path: str,
                  safety_shrt_path: str | None = None) -> dict:
    """Per-script displacement at every layer. Shows how script rankings
    change through the model's depth. Requires --layers all .shrt file.

    If safety_shrt_path is provided (a .shrt with discovered safety direction),
    the crossing significance analysis tests overlap at that model's discovered
    safety layer specifically, not at hardcoded ranges.
    """
    from .shrt import load_shrt
    from .frt import load_frt

    s = load_shrt(shrt_path)
    m = s['metadata']
    if 'layer_stats' not in m:
        return {"error": "No layer data — run with --layers all"}

    # Try to get discovered safety layer from metadata or separate .shrt
    safety_layer = None
    safety_accuracy = None
    if safety_shrt_path:
        safety_shrt = load_shrt(safety_shrt_path)
        sm = safety_shrt['metadata']
        if sm.get('direction', {}).get('available'):
            safety_layer = sm['direction']['layer']
            safety_accuracy = sm['direction']['accuracy']
    elif m.get('direction', {}).get('available'):
        safety_layer = m['direction']['layer']
        safety_accuracy = m['direction']['accuracy']

    frt = load_frt(frt_path)
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}

    layers = m['layers']
    token_ids = s['token_ids']
    scripts = [fl.get(int(tid), 'unknown') for tid in token_ids]

    # For each layer, compute mean delta per script
    layer_script_data = {}
    for layer in layers:
        key = f"deltas_L{layer}"
        if key not in s:
            continue
        deltas = s[key].astype(np.float32)
        grand_mean = float(deltas.mean())

        by_script = defaultdict(list)
        for i, sc in enumerate(scripts):
            if sc not in ('special', 'unknown'):
                by_script[sc].append(float(deltas[i]))

        layer_data = {}
        for sc, vals in by_script.items():
            if len(vals) >= 5:
                layer_data[sc] = {
                    "n": len(vals),
                    "mean": round(float(np.mean(vals)), 2),
                    "relative": round(float(np.mean(vals)) / max(grand_mean, 1e-8), 3),
                }
        layer_script_data[layer] = layer_data

    # Find rank changes: which scripts move most across layers?
    all_scripts = set()
    for ld in layer_script_data.values():
        all_scripts.update(ld.keys())

    script_trajectories = {}
    for sc in sorted(all_scripts):
        relatives = []
        for layer in layers:
            ld = layer_script_data.get(layer, {})
            if sc in ld:
                relatives.append((layer, ld[sc]['relative']))
        if len(relatives) >= 3:
            rels = [r for _, r in relatives]
            script_trajectories[sc] = {
                "n_layers": len(relatives),
                "min_rel": round(min(rels), 3),
                "max_rel": round(max(rels), 3),
                "range": round(max(rels) - min(rels), 3),
                "early": round(rels[0], 3) if rels else 0,
                "mid": round(rels[len(rels)//2], 3) if rels else 0,
                "late": round(rels[-1], 3) if rels else 0,
                "trajectory": relatives,
            }

    # Detect crossings: where does one script's relative overtake another's?
    crossings = []
    scripts_with_traj = [s for s in script_trajectories
                         if len(script_trajectories[s].get('trajectory', [])) >= 3]
    for i, s1 in enumerate(scripts_with_traj):
        for s2 in scripts_with_traj[i+1:]:
            t1 = dict(script_trajectories[s1]['trajectory'])
            t2 = dict(script_trajectories[s2]['trajectory'])
            common_layers = sorted(set(t1) & set(t2))
            if len(common_layers) < 3:
                continue
            # Check if s1 > s2 at start and s1 < s2 at end (or vice versa)
            first = common_layers[0]
            last = common_layers[-1]
            if (t1[first] > t2[first] and t1[last] < t2[last]) or \
               (t1[first] < t2[first] and t1[last] > t2[last]):
                # Find the crossing layer
                for j in range(len(common_layers) - 1):
                    l_a = common_layers[j]
                    l_b = common_layers[j + 1]
                    if (t1[l_a] >= t2[l_a] and t1[l_b] < t2[l_b]) or \
                       (t1[l_a] <= t2[l_a] and t1[l_b] > t2[l_b]):
                        crossings.append({
                            "scripts": (s1, s2),
                            "cross_layer": l_b,
                            "s1_before": round(t1[l_a], 3),
                            "s2_before": round(t2[l_a], 3),
                            "s1_after": round(t1[l_b], 3),
                            "s2_after": round(t2[l_b], 3),
                        })
                        break

    # Crossing density: how many crossings, where do they cluster
    n_total_layers = len(layers)
    n_scripts = len(scripts_with_traj)
    n_pairs = n_scripts * (n_scripts - 1) // 2
    n_crossings = len(crossings)

    crossing_significance = {
        "n_crossings": n_crossings,
        "n_script_pairs": n_pairs,
        "n_layers": n_total_layers,
        "crossing_rate": round(n_crossings / max(n_pairs, 1), 3),
    }

    if n_crossings >= 2:
        cross_layers = [c['cross_layer'] for c in crossings]
        from collections import Counter
        layer_counts = Counter(cross_layers)

        # For each window of K layers, count how many crossings fall in it
        # Test windows of size 2, 3, 4 (matching the L14-17 claim = 4-layer window)
        for window_size in [2, 3, 4]:
            best_window = None
            best_count = 0
            for start_idx in range(n_total_layers - window_size + 1):
                window_layers = set(layers[start_idx:start_idx + window_size])
                count = sum(1 for cl in cross_layers if cl in window_layers)
                if count > best_count:
                    best_count = count
                    best_window = (layers[start_idx], layers[start_idx + window_size - 1])

            # Expected crossings in a random K-layer window if crossings were uniform
            expected = n_crossings * window_size / n_total_layers
            crossing_significance[f"window_{window_size}"] = {
                "best_window": best_window,
                "crossings_in_window": best_count,
                "expected_if_uniform": round(expected, 2),
                "ratio": round(best_count / max(expected, 0.01), 2),
            }

        # Per-layer crossing density
        crossing_significance["by_layer"] = {
            str(layer): count for layer, count in sorted(layer_counts.items())
        }

        # Safety layer overlap test (model-specific, not hardcoded)
        if safety_layer is not None:
            # Count crossings at the safety layer and its immediate neighbors
            safety_window = {safety_layer - 1, safety_layer, safety_layer + 1}
            safety_crossings = sum(1 for cl in cross_layers if cl in safety_window)
            expected_3 = n_crossings * 3 / n_total_layers
            at_safety = layer_counts.get(safety_layer, 0)

            # What scripts cross at or near the safety layer?
            safety_crossing_scripts = [
                c for c in crossings if c['cross_layer'] in safety_window
            ]

            crossing_significance["safety_layer"] = {
                "layer": safety_layer,
                "accuracy": safety_accuracy,
                "crossings_at_layer": at_safety,
                "crossings_in_window": safety_crossings,
                "window": sorted(l for l in safety_window if l in set(layers)),
                "expected_if_uniform": round(expected_3, 2),
                "ratio": round(safety_crossings / max(expected_3, 0.01), 2),
                "scripts_crossing": [
                    {"scripts": c["scripts"], "layer": c["cross_layer"]}
                    for c in safety_crossing_scripts
                ],
            }
        else:
            crossing_significance["safety_layer"] = {
                "status": "not discovered",
                "note": "Run discovery pipeline or provide --safety-shrt with discovered direction",
            }

    return {
        "model": m['model']['name'],
        "n_layers": len(layers),
        "layers": layers,
        "safety_layer": safety_layer,
        "safety_accuracy": safety_accuracy,
        "layer_script_data": layer_script_data,
        "script_trajectories": script_trajectories,
        "crossings": crossings,
        "crossing_significance": crossing_significance,
    }


def tokenizer_health(frt_path: str, shrt_path: str | None = None) -> dict:
    """Tokenizer round-trip analysis.

    For every token: does decode→re-encode preserve the ID?
    Groups tokens into clean, collapsed (decode collision), and silent (empty decode).
    If a .shrt is provided, includes displacement per group.
    """
    from .frt import load_frt, _detect_script

    frt = load_frt(frt_path)
    meta = frt['metadata']

    # Load shrt if available
    delta_lookup = {}
    if shrt_path:
        from .shrt import load_shrt
        shrt = load_shrt(shrt_path)
        for i in range(len(shrt['token_ids'])):
            delta_lookup[int(shrt['token_ids'][i])] = float(shrt['deltas'][i])

    token_ids = frt['token_ids']
    token_texts = frt['token_texts']
    byte_counts = frt['byte_counts']
    scripts = frt['scripts']

    # Classify each token
    clean = []      # round-trip preserves ID
    collapsed = []  # multiple IDs → same text (decode collision)
    expanded = []   # 1 ID → multiple IDs on re-encode
    silent = []     # decode produces empty/whitespace

    text_to_ids = defaultdict(list)
    for i in range(len(token_ids)):
        tid = int(token_ids[i])
        text = str(token_texts[i])
        text_to_ids[text].append(tid)

    # Find decode collisions (multiple IDs → same text)
    collisions = {text: ids for text, ids in text_to_ids.items() if len(ids) > 1}

    for i in range(len(token_ids)):
        tid = int(token_ids[i])
        text = str(token_texts[i])
        b = int(byte_counts[i])
        sc = str(scripts[i])
        delta = delta_lookup.get(tid)

        entry = {
            "id": tid,
            "text": text[:30],
            "bytes": b,
            "script": sc,
        }
        if delta is not None:
            entry["delta"] = round(delta, 2)

        if not text.strip():
            entry["reason"] = "empty_decode"
            silent.append(entry)
        elif text in collisions and len(collisions[text]) > 1:
            entry["reason"] = "collision"
            entry["n_colliders"] = len(collisions[text])
            collapsed.append(entry)
        else:
            clean.append(entry)

    # Summary statistics per group
    result = {
        "vocab_size": int(meta['tokenizer']['vocab_size']),
        "n_clean": len(clean),
        "n_collapsed": len(collapsed),
        "n_silent": len(silent),
        "collision_groups": len(collisions),
        "total_collision_tokens": sum(len(ids) for ids in collisions.values()),
    }

    # Properties of each group — let the reader find the pattern
    for group_name, group in [("clean", clean), ("collapsed", collapsed), ("silent", silent)]:
        if not group:
            continue
        ids = [e["id"] for e in group]
        bytes_list = [e["bytes"] for e in group]
        scripts_list = [e["script"] for e in group]

        from collections import Counter
        script_dist = Counter(scripts_list)

        group_data = {
            "n": len(group),
            "mean_id": round(float(np.mean(ids)), 0),
            "mean_bytes": round(float(np.mean(bytes_list)), 1),
            "scripts": dict(script_dist.most_common()),
        }

        if any("delta" in e for e in group):
            deltas = [e["delta"] for e in group if "delta" in e]
            if deltas:
                group_data["mean_delta"] = round(float(np.mean(deltas)), 2)
                group_data["n_with_delta"] = len(deltas)

        result[group_name] = group_data

    # Largest collision groups
    biggest = sorted(collisions.items(), key=lambda x: -len(x[1]))[:10]
    result["largest_collisions"] = [
        {"text": repr(text)[:30], "n_ids": len(ids), "ids": ids[:5]}
        for text, ids in biggest
    ]

    return result


def embedding_profile(model_id: str, frt_path: str, shrt_path: str | None = None) -> dict:
    """Embedding table norms per script, with optional displacement correlation.

    Reports: embedding norms per script, norm distribution across the vocabulary.
    If .shrt provided, computes r(embedding_norm, delta) overall and per script.
    """
    from ..backend.protocol import load_backend
    from .frt import load_frt

    backend = load_backend(model_id)
    inner = getattr(backend.model, 'model', backend.model)

    import mlx.core as mx
    embed = np.array(inner.embed_tokens.weight)  # [vocab_size, hidden_size]
    norms = np.linalg.norm(embed, axis=1).astype(np.float32)

    frt = load_frt(frt_path)
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}

    # Per-script embedding norms
    by_script = defaultdict(list)
    for tid in range(len(norms)):
        s = fl.get(tid)
        if s and s not in ('special', 'unknown'):
            by_script[s].append(float(norms[tid]))

    script_norms = {}
    for s in sorted(by_script, key=lambda x: -np.mean(by_script[x])):
        vals = by_script[s]
        if len(vals) >= 10:
            script_norms[s] = {
                "n": len(vals),
                "mean_norm": round(float(np.mean(vals)), 4),
                "std_norm": round(float(np.std(vals)), 4),
            }

    result = {
        "model": model_id,
        "embedding_shape": list(embed.shape),
        "norm_mean": round(float(norms.mean()), 4),
        "norm_std": round(float(norms.std()), 4),
        "norm_min": round(float(norms.min()), 4),
        "norm_max": round(float(norms.max()), 4),
        "script_norms": script_norms,
    }

    # Correlation with displacement if .shrt available
    if shrt_path:
        from .shrt import load_shrt
        shrt = load_shrt(shrt_path)
        shrt_ids = shrt['token_ids']
        shrt_deltas = shrt['deltas'].astype(np.float32)
        embed_norms_matched = np.array([float(norms[tid]) for tid in shrt_ids])
        r = float(np.corrcoef(embed_norms_matched, shrt_deltas)[0, 1])
        result["r_norm_delta"] = round(r, 4)

        # Per-script r(embedding_norm, delta)
        script_correlations = {}
        for s in by_script:
            s_mask = [fl.get(int(tid)) == s for tid in shrt_ids]
            s_norms = embed_norms_matched[s_mask]
            s_deltas = shrt_deltas[s_mask]
            if len(s_norms) >= 20:
                r_s = float(np.corrcoef(s_norms, s_deltas)[0, 1])
                script_correlations[s] = round(r_s, 4)
        result["script_norm_delta_r"] = script_correlations

    return result


def displacement_output_scatter(shrt_path: str, sht_path: str, frt_path: str | None = None) -> dict:
    """Joint distribution of residual displacement (.shrt) and output KL
    divergence (.sht) for shared tokens.

    Reports r(delta, KL), median-split quadrants, and per-quadrant
    script breakdown if .frt provided.
    """
    from .shrt import load_shrt
    from .sht import load_sht

    shrt = load_shrt(shrt_path)
    sht = load_sht(sht_path)

    sl = {int(shrt['token_ids'][i]): float(shrt['deltas'][i])
          for i in range(len(shrt['token_ids']))}
    tl = {int(sht['token_ids'][i]): float(sht['kl_divs'][i])
          for i in range(len(sht['token_ids']))}

    shared = sorted(set(sl) & set(tl))
    if len(shared) < 20:
        return {"error": f"Only {len(shared)} shared tokens"}

    deltas = np.array([sl[t] for t in shared])
    kls = np.array([tl[t] for t in shared])

    # Median split — data decides the threshold
    med_d = float(np.median(deltas))
    med_k = float(np.median(kls))

    # Classify
    fl = None
    if frt_path:
        from .frt import load_frt
        frt = load_frt(frt_path)
        fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
              for i in range(len(frt['token_ids']))}

    categories = {"high_both": [], "high_delta": [], "high_kl": [], "low_both": []}
    for i, tid in enumerate(shared):
        d, k = deltas[i], kls[i]
        if d >= med_d and k >= med_k:
            cat = "high_both"
        elif d >= med_d and k < med_k:
            cat = "high_delta"
        elif d < med_d and k >= med_k:
            cat = "high_kl"
        else:
            cat = "low_both"

        entry = {"id": tid, "delta": round(float(d), 2), "kl": round(float(k), 4)}
        if fl and tid in fl:
            entry["script"] = fl[tid]
        categories[cat].append(entry)

    # Per-category statistics
    result = {
        "n_shared": len(shared),
        "median_delta": round(med_d, 2),
        "median_kl": round(med_k, 4),
        "r_delta_kl": round(float(np.corrcoef(deltas, kls)[0, 1]), 4),
    }

    for cat_name, tokens in categories.items():
        if not tokens:
            result[cat_name] = {"n": 0}
            continue

        cat_d = [t["delta"] for t in tokens]
        cat_k = [t["kl"] for t in tokens]

        cat_result = {
            "n": len(tokens),
            "pct": round(len(tokens) / len(shared) * 100, 1),
            "mean_delta": round(float(np.mean(cat_d)), 2),
            "mean_kl": round(float(np.mean(cat_k)), 4),
        }

        # Script breakdown if available
        if fl:
            from collections import Counter
            scripts = Counter(t.get("script", "?") for t in tokens)
            cat_result["scripts"] = dict(scripts.most_common())

        # Top 5 tokens in this category
        if cat_name in ("high_both", "high_kl"):
            top = sorted(tokens, key=lambda t: -t["kl"])[:5]
        else:
            top = sorted(tokens, key=lambda t: -t["delta"])[:5]
        cat_result["top"] = top

        result[cat_name] = cat_result

    return result


def within_script_analysis(shrt_path: str, frt_path: str) -> dict:
    """Within-script dispersion and correlations.

    For each script: cv(delta), r(byte_count, delta), r(token_id, delta),
    and displacement broken down by byte count.

    Also: global displacement by byte-count bin, and r(script_mean, script_cv).
    """
    from .shrt import load_shrt
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)

    # Build lookups
    frt_data = {}
    for i in range(len(frt['token_ids'])):
        tid = int(frt['token_ids'][i])
        frt_data[tid] = {
            'script': str(frt['scripts'][i]),
            'bytes': int(frt['byte_counts'][i]),
            'chars': int(frt['char_counts'][i]),
        }

    shrt_data = {}
    for i in range(len(shrt['token_ids'])):
        tid = int(shrt['token_ids'][i])
        shrt_data[tid] = float(shrt['deltas'][i])

    shared = set(frt_data) & set(shrt_data)
    if len(shared) < 50:
        return {"error": f"Only {len(shared)} shared tokens"}

    # Group by script
    by_script = defaultdict(lambda: {'deltas': [], 'bytes': [], 'chars': [], 'ids': []})
    for tid in shared:
        f = frt_data[tid]
        s = f['script']
        if s in ('special', 'unknown'):
            continue
        by_script[s]['deltas'].append(shrt_data[tid])
        by_script[s]['bytes'].append(f['bytes'])
        by_script[s]['chars'].append(f['chars'])
        by_script[s]['ids'].append(tid)

    c = lambda x, y: round(float(np.corrcoef(x, y)[0, 1]), 4) if np.std(x) > 0 and np.std(y) > 0 else 0.0

    scripts = {}
    for s in sorted(by_script, key=lambda x: -np.mean(by_script[x]['deltas'])):
        data = by_script[s]
        deltas = np.array(data['deltas'], dtype=np.float32)
        bytes_ = np.array(data['bytes'], dtype=np.float32)
        ids = np.array(data['ids'], dtype=np.float32)
        n = len(deltas)
        if n < 10:
            continue

        mean_d = float(deltas.mean())
        std_d = float(deltas.std())
        cv = std_d / mean_d if mean_d > 1e-8 else 0.0

        # Within-script correlations
        r_bytes = c(bytes_, deltas)
        r_id = c(ids, deltas)

        # Quartile analysis: split by byte count within this script
        byte_vals = sorted(set(bytes_))
        quartiles = {}
        if len(byte_vals) >= 2:
            for bv in byte_vals[:8]:  # cap at 8 distinct byte values
                mask = bytes_ == bv
                if mask.sum() >= 5:
                    q_deltas = deltas[mask]
                    quartiles[int(bv)] = {
                        "n": int(mask.sum()),
                        "mean_delta": round(float(q_deltas.mean()), 2),
                        "std_delta": round(float(q_deltas.std()), 2),
                    }

        scripts[s] = {
            "n": n,
            "mean_delta": round(mean_d, 2),
            "std_delta": round(std_d, 2),
            "cv": round(cv, 4),
            "r_bytes_delta": r_bytes,
            "r_id_delta": r_id,
            "by_byte_count": quartiles,
        }

    # Global byte-count bins
    all_deltas = []
    all_bytes = []
    for tid in shared:
        f = frt_data[tid]
        if f['script'] in ('special', 'unknown'):
            continue
        all_deltas.append(shrt_data[tid])
        all_bytes.append(f['bytes'])

    all_deltas = np.array(all_deltas, dtype=np.float32)
    all_bytes = np.array(all_bytes, dtype=np.float32)

    byte_bins = {}
    for b in range(1, 9):
        if b < 8:
            mask = all_bytes == b
        else:
            mask = all_bytes >= 8
            b = "8+"
        count = int(mask.sum())
        if count >= 3:
            bd = all_deltas[mask]
            byte_bins[str(b)] = {
                "n": count,
                "mean_delta": round(float(bd.mean()), 2),
                "std_delta": round(float(bd.std()), 2),
                "cv": round(float(bd.std() / bd.mean()) if bd.mean() > 1e-8 else 0, 4),
                "provisional": count < 20,
            }

    # r(mean, cv) across scripts
    script_means = []
    script_cvs = []
    for s, data in scripts.items():
        if data['n'] >= 20:
            script_means.append(data['mean_delta'])
            script_cvs.append(data['cv'])

    r_mean_cv = c(np.array(script_means), np.array(script_cvs)) if len(script_means) >= 3 else None

    return {
        "model": shrt['metadata']['model']['name'],
        "n_shared": len(shared),
        "scripts": scripts,
        "byte_bins": byte_bins,
        "r_script_mean_vs_cv": r_mean_cv,
    }


def displacement_directions(shrt_path: str, frt_path: str,
                            safety_shrt_path: str | None = None) -> dict:
    """Directional analysis of displacement vectors.

    (1) Within-script cosine coherence: mean pairwise cosine similarity
        of displacement vectors within each script.
    (2) Between-script separation: cosine similarity of mean displacement
        vectors across scripts.
    (3) Safety direction projection: per-token dot product with the
        discovered safety direction, grouped by script.
    """
    from .shrt import load_shrt
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)

    vectors = shrt['vectors'].astype(np.float32)
    token_ids = shrt['token_ids']
    hidden_size = vectors.shape[1] if len(vectors.shape) > 1 else 0

    if hidden_size == 0:
        return {"error": "No vectors in .shrt file"}

    # Build script lookup
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}

    # Group vector indices by script
    script_indices = defaultdict(list)
    for i in range(len(token_ids)):
        tid = int(token_ids[i])
        s = fl.get(tid, 'unknown')
        if s in ('special', 'unknown'):
            continue
        script_indices[s].append(i)

    # (1) Within-script cosine coherence
    # For each script: mean pairwise cosine of displacement vectors
    # For large groups, sample pairs to keep computation bounded
    script_coherence = {}
    script_mean_vecs = {}
    scripts_excluded = {}
    for s, indices in sorted(script_indices.items()):
        if len(indices) < 5:
            scripts_excluded[s] = len(indices)
            continue
        vecs = vectors[indices]
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs_normed = vecs / norms

        # Mean direction
        mean_vec = vecs.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_vec))
        script_mean_vecs[s] = mean_vec

        # Mean cosine with mean direction (efficient proxy for pairwise)
        if mean_norm > 1e-8:
            mean_dir = mean_vec / mean_norm
            cosines_to_mean = vecs_normed @ mean_dir
            coherence = float(cosines_to_mean.mean())
        else:
            coherence = 0.0

        # Also compute pairwise for small scripts (exact)
        n = len(indices)
        if n <= 200:
            gram = vecs_normed @ vecs_normed.T
            # Upper triangle, excluding diagonal
            triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            pairwise_mean = float(gram[triu_mask].mean())
        else:
            # Sample 5000 random pairs
            rng = np.random.RandomState(42)
            i1 = rng.randint(0, n, size=5000)
            i2 = rng.randint(0, n, size=5000)
            mask = i1 != i2
            i1, i2 = i1[mask], i2[mask]
            cosines = (vecs_normed[i1] * vecs_normed[i2]).sum(axis=1)
            pairwise_mean = float(cosines.mean())

        script_coherence[s] = {
            "n": n,
            "coherence_to_mean": round(coherence, 4),
            "pairwise_cosine": round(pairwise_mean, 4),
            "mean_vec_norm": round(mean_norm, 2),
        }

    # (2) Between-script separation
    # Cosine similarity of mean displacement vectors
    script_names = sorted(s for s in script_mean_vecs if s in script_coherence)
    separation = {}
    for i, s1 in enumerate(script_names):
        for s2 in script_names[i+1:]:
            v1 = script_mean_vecs[s1]
            v2 = script_mean_vecs[s2]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos = float(np.dot(v1, v2) / (n1 * n2))
            else:
                cos = 0.0
            separation[f"{s1}:{s2}"] = round(cos, 4)

    # Summary: mean within-script coherence vs mean between-script cosine
    mean_within = float(np.mean([v['pairwise_cosine']
                                 for v in script_coherence.values()])) if script_coherence else 0
    mean_between = float(np.mean(list(separation.values()))) if separation else 0

    # PCA on displacement vectors: how many dimensions does the shart occupy?
    # If PC1 explains >80%, displacement is effectively one-dimensional.
    all_vecs = vectors[np.array([i for indices in script_indices.values()
                                 for i in indices])]
    if len(all_vecs) >= 20:
        centered = all_vecs - all_vecs.mean(axis=0)
        # Use SVD on centered data (more stable than covariance eigendecomp)
        # Compute top-k singular values only — full SVD on [15000, 896] is fine
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        variance = S ** 2
        total_var = float(variance.sum())
        cumulative = np.cumsum(variance) / total_var

        # How many PCs for 50%, 80%, 90%, 95%?
        pca_thresholds = {}
        for pct in [0.5, 0.8, 0.9, 0.95]:
            n_pcs = int(np.searchsorted(cumulative, pct)) + 1
            pca_thresholds[str(pct)] = n_pcs

        pca = {
            "n_vectors": len(all_vecs),
            "pc1_variance_pct": round(float(cumulative[0]) * 100, 2),
            "pc2_variance_pct": round(float(variance[1] / total_var) * 100, 2) if len(variance) > 1 else 0,
            "pc3_variance_pct": round(float(variance[2] / total_var) * 100, 2) if len(variance) > 2 else 0,
            "pcs_for_threshold": pca_thresholds,
            "top_10_pct": [round(float(v / total_var) * 100, 2) for v in variance[:10]],
        }

        # Per-script: project onto PC1, report mean projection
        pc1 = Vt[0]
        script_pc1 = {}
        for s, indices in sorted(script_indices.items()):
            if len(indices) < 5:
                continue
            vecs_s = vectors[indices]
            projections = vecs_s @ pc1
            script_pc1[s] = {
                "n": len(indices),
                "mean_pc1": round(float(projections.mean()), 2),
                "std_pc1": round(float(projections.std()), 2),
            }
        pca["script_pc1_projections"] = script_pc1
    else:
        pca = {"error": "too few vectors for PCA"}

    result = {
        "model": shrt['metadata']['model']['name'],
        "hidden_size": hidden_size,
        "n_tokens": len(token_ids),
        "script_coherence": script_coherence,
        "scripts_excluded": scripts_excluded,
        "between_script_cosine": separation,
        "mean_within_coherence": round(mean_within, 4),
        "mean_between_cosine": round(mean_between, 4),
        "pca": pca,
    }

    # (3) Safety direction projection
    safety_dir = None
    safety_layer = None
    if safety_shrt_path:
        safety_shrt = load_shrt(safety_shrt_path)
        sm = safety_shrt['metadata']
        if sm.get('direction', {}).get('available'):
            safety_layer = sm['direction']['layer']
            # The safety direction should be in the vectors or stored separately
            # Check if the .shrt has safety_shift pre-computed in metadata
    elif (shrt['metadata'].get('direction') or {}).get('available'):
        safety_layer = shrt['metadata']['direction']['layer']

    # Try to load safety direction from the DB blob naming convention
    # or compute from the vectors if safety shifts are in the token data
    # For now: if individual tokens have safety_shift in the metadata,
    # extract from there; otherwise check for direction in vectors
    top_sharts = shrt['metadata'].get('top_sharts', [])
    has_safety_shift = any('safety_shift' in t for t in top_sharts)

    if has_safety_shift and safety_layer is not None:
        # Safety shifts are pre-computed per token in metadata
        # But they're only in top/bottom sharts, not all tokens
        # We need to recompute from vectors + direction
        pass

    # If we have vectors and the shrt was run with a safety direction,
    # the direction vector itself isn't stored in the .shrt —
    # it's in the DB. We can compute mean direction of known-harmful tokens
    # as a proxy, but that's interpretation. Instead, report what we have.
    if safety_layer is not None:
        result["safety_layer"] = safety_layer
        result["safety_note"] = ("Safety direction vector is in the DB, "
                                 "not in the .shrt file. Run with --db to project.")

    return result


def code_anatomy(shrt_path: str, frt_path: str,
                 all_layers_shrt_path: str | None = None) -> dict:
    """Subcategorize 'code' tokens and report displacement per subcategory.

    Categories: structural ({} () [] ;), operators (= + - * /),
    keywords (def class return if), whitespace, identifiers, mixed.

    Reports vocab counts, measured counts, displacement, and per-layer
    trajectories if all-layers shrt is provided.
    """
    from .shrt import load_shrt
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)

    # Build lookups
    frt_texts = {}
    frt_scripts = {}
    for i in range(len(frt['token_ids'])):
        tid = int(frt['token_ids'][i])
        frt_texts[tid] = str(frt['token_texts'][i])
        frt_scripts[tid] = str(frt['scripts'][i])

    shrt_deltas = {}
    for i in range(len(shrt['token_ids'])):
        tid = int(shrt['token_ids'][i])
        shrt_deltas[tid] = float(shrt['deltas'][i])

    shared = set(frt_texts) & set(shrt_deltas)

    # Classify code tokens
    _STRUCTURAL = set('{}()[];')
    _OPERATORS = {'=', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/',
                  '%', '&', '|', '^', '~', '<<', '>>', '**', '//', '+=',
                  '-=', '*=', '/=', '&&', '||', '!', '?', ':', '::', '=>',
                  '->', '...', '.', ','}
    _KEYWORDS = {'def', 'function', 'class', 'return', 'if', 'else', 'elif',
                 'for', 'while', 'do', 'break', 'continue', 'import', 'from',
                 'export', 'const', 'let', 'var', 'int', 'float', 'str',
                 'bool', 'void', 'null', 'None', 'true', 'false', 'True',
                 'False', 'try', 'except', 'catch', 'finally', 'throw',
                 'raise', 'async', 'await', 'yield', 'lambda', 'with', 'as',
                 'in', 'not', 'and', 'or', 'is', 'new', 'delete', 'typeof',
                 'instanceof', 'switch', 'case', 'default', 'enum', 'struct',
                 'trait', 'impl', 'pub', 'fn', 'mut', 'self', 'super',
                 'static', 'final', 'abstract', 'interface', 'extends',
                 'implements', 'override', 'virtual', 'template', 'namespace',
                 'using', 'package', 'private', 'protected', 'public',
                 'require', 'include', 'pragma', 'ifdef', 'endif', 'define',
                 'print', 'println', 'printf', 'fmt', 'log', 'console'}

    def _classify_code(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            if any(c in text for c in '\n\t\r'):
                return 'whitespace'
            if text and all(c == ' ' for c in text):
                return 'whitespace'
            return 'whitespace'
        if stripped in _STRUCTURAL:
            return 'structural'
        if stripped in _OPERATORS:
            return 'operators'
        if stripped in _KEYWORDS:
            return 'keywords'
        # Check if it looks like a keyword prefix/suffix (partial BPE)
        if stripped.isalpha() and len(stripped) <= 3:
            return 'mixed'
        if stripped.startswith('_') or stripped.startswith('__'):
            return 'identifiers'
        if stripped.isalpha():
            return 'identifiers'
        return 'mixed'

    # First pass: classify ALL code tokens in the full .frt vocabulary
    # This shows whether small n is sampling bias or structural absence in BPE
    vocab_cats = defaultdict(lambda: {'count': 0, 'examples': []})
    for tid, text in frt_texts.items():
        if frt_scripts.get(tid) == 'code':
            cat = _classify_code(text)
            vocab_cats[cat]['count'] += 1
            if len(vocab_cats[cat]['examples']) < 5:
                vocab_cats[cat]['examples'].append(repr(text)[:30])

    # Second pass: collect measured code tokens by subcategory
    by_cat = defaultdict(lambda: {'deltas': [], 'ids': [], 'texts': []})
    # Also collect non-code scripts for context
    by_script = defaultdict(lambda: {'deltas': []})

    for tid in shared:
        script = frt_scripts.get(tid, 'unknown')
        delta = shrt_deltas[tid]

        if script == 'code':
            text = frt_texts[tid]
            cat = _classify_code(text)
            by_cat[cat]['deltas'].append(delta)
            by_cat[cat]['ids'].append(tid)
            by_cat[cat]['texts'].append(text)
        elif script not in ('special', 'unknown'):
            by_script[script]['deltas'].append(delta)

    # Summary per code subcategory
    grand_mean = float(np.mean([shrt_deltas[t] for t in shared
                                if frt_scripts.get(t) not in ('special', 'unknown')]))

    categories = {}
    for cat in sorted(by_cat, key=lambda x: -np.mean(by_cat[x]['deltas']) if by_cat[x]['deltas'] else 0):
        data = by_cat[cat]
        if len(data['deltas']) < 3:
            continue
        deltas = np.array(data['deltas'], dtype=np.float32)
        vc = vocab_cats.get(cat, {'count': 0})
        categories[cat] = {
            "n_measured": len(deltas),
            "n_vocab": vc['count'],
            "coverage_pct": round(len(deltas) / max(vc['count'], 1) * 100, 1),
            "mean_delta": round(float(deltas.mean()), 2),
            "std_delta": round(float(deltas.std()), 2),
            "relative": round(float(deltas.mean()) / max(grand_mean, 1e-8), 3),
            "examples": [repr(t)[:20] for t in data['texts'][:8]],
        }

    # Report categories that exist in vocab but have no measured tokens
    for cat, vc in vocab_cats.items():
        if cat not in categories and vc['count'] > 0:
            categories[cat] = {
                "n_measured": 0,
                "n_vocab": vc['count'],
                "coverage_pct": 0.0,
                "mean_delta": None,
                "examples": vc['examples'],
                "note": "exists in vocab but not in .shrt sample",
            }

    # Context: non-code scripts for comparison
    context_scripts = {}
    for s in sorted(by_script, key=lambda x: -np.mean(by_script[x]['deltas'])):
        deltas = np.array(by_script[s]['deltas'], dtype=np.float32)
        if len(deltas) >= 10:
            context_scripts[s] = {
                "n": len(deltas),
                "mean_delta": round(float(deltas.mean()), 2),
                "relative": round(float(deltas.mean()) / max(grand_mean, 1e-8), 3),
            }

    total_code_vocab = sum(vc['count'] for vc in vocab_cats.values())
    result = {
        "model": shrt['metadata']['model']['name'],
        "n_code_measured": sum(len(by_cat[c]['deltas']) for c in by_cat),
        "n_code_vocab": total_code_vocab,
        "grand_mean_delta": round(grand_mean, 2),
        "code_subcategories": categories,
        "context_scripts": context_scripts,
    }

    # Per-layer trajectory if all-layers shrt available
    if all_layers_shrt_path:
        als = load_shrt(all_layers_shrt_path)
        meta = als['metadata']
        if 'layers' not in meta:
            result["layer_error"] = "No layer data in all-layers file"
            return result

        layers = meta['layers']
        al_ids = als['token_ids']

        # Build token_id -> index map for the all-layers file
        al_idx = {int(al_ids[i]): i for i in range(len(al_ids))}

        # For each code subcategory, compute trajectory
        cat_trajectories = {}
        for cat, data in by_cat.items():
            if len(data['deltas']) < 5:
                continue
            # Find token indices present in the all-layers file
            cat_indices = [al_idx[tid] for tid in data['ids'] if tid in al_idx]
            if len(cat_indices) < 5:
                continue

            trajectory = []
            for layer in layers:
                key = f"deltas_L{layer}"
                if key not in als:
                    continue
                layer_deltas = als[key].astype(np.float32)
                # Grand mean at this layer for normalization
                layer_grand = float(layer_deltas.mean())
                # This subcategory's mean at this layer
                cat_deltas_at_layer = layer_deltas[cat_indices]
                cat_mean = float(cat_deltas_at_layer.mean())
                relative = cat_mean / max(layer_grand, 1e-8)
                trajectory.append((layer, round(relative, 3)))

            if trajectory:
                rels = [r for _, r in trajectory]
                cat_trajectories[cat] = {
                    "n_tokens": len(cat_indices),
                    "early": round(rels[0], 3) if rels else 0,
                    "mid": round(rels[len(rels) // 2], 3) if rels else 0,
                    "late": round(rels[-1], 3) if rels else 0,
                    "range": round(max(rels) - min(rels), 3) if rels else 0,
                    "falls": rels[-1] < rels[0] - 0.05,
                    "trajectory": trajectory,
                }

        # Also compute non-code script trajectories for context
        script_trajectories = {}
        for s, sdata in by_script.items():
            if len(sdata['deltas']) < 10:
                continue
            s_ids = [tid for tid in shared if frt_scripts.get(tid) == s]
            s_indices = [al_idx[tid] for tid in s_ids if tid in al_idx]
            if len(s_indices) < 10:
                continue

            trajectory = []
            for layer in layers:
                key = f"deltas_L{layer}"
                if key not in als:
                    continue
                layer_deltas = als[key].astype(np.float32)
                layer_grand = float(layer_deltas.mean())
                s_mean = float(layer_deltas[s_indices].mean())
                relative = s_mean / max(layer_grand, 1e-8)
                trajectory.append((layer, round(relative, 3)))

            if trajectory:
                rels = [r for _, r in trajectory]
                script_trajectories[s] = {
                    "n_tokens": len(s_indices),
                    "early": round(rels[0], 3),
                    "late": round(rels[-1], 3),
                    "range": round(max(rels) - min(rels), 3),
                    "falls": rels[-1] < rels[0] - 0.05,
                }

        result["code_trajectories"] = cat_trajectories
        result["script_trajectories"] = script_trajectories

    return result


def data_matrix(data_dir: str) -> dict:
    """Scan data directory for .frt, .shrt, .sht files.
    Report per-model coverage: what measurements exist, what's missing.
    """
    from .shrt import load_shrt
    from .frt import load_frt

    p = Path(data_dir)
    if not p.is_dir():
        return {"error": f"Not a directory: {data_dir}"}

    shrt_files = sorted(p.glob("*.shrt.npz"))
    sht_files = sorted(p.glob("*.sht.npz"))
    frt_files = sorted(p.glob("*.frt.npz"))
    decompose_files = sorted(p.glob("*decompose*.npz"))

    models = {}

    for f in shrt_files:
        try:
            s = load_shrt(str(f))
            m = s['metadata']
            arch = m['model']['name']
            h = m['model']['hidden_size']
            nl = m['model']['n_layers']
            name = f"{arch}_{h}_{nl}L"
            if name not in models:
                models[name] = {
                    "hidden_size": h,
                    "n_layers": nl,
                    "shrt_files": [], "frt_file": None,
                    "sht_files": [], "decompose_file": None,
                }
            models[name]["shrt_files"].append({
                "file": f.name,
                "n_sampled": m['index']['n_sampled'],
                "has_vectors": 'vectors' in s,
                "has_direction": bool((m.get('direction') or {}).get('available')),
                "direction_layer": (m.get('direction') or {}).get('layer'),
                "has_layer_data": 'layer_stats' in m,
                "n_layers_measured": len(m.get('layers', [])) if m.get('layers') else 1,
                "baseline_entropy": m['baseline']['entropy'],
            })
        except Exception:
            pass

    for f in frt_files:
        try:
            frt = load_frt(str(f))
            m = frt['metadata']
            vocab_hash = m['tokenizer']['vocab_hash']
            # Match frt to models by vocab_hash if stored in shrt,
            # or by filename heuristic: strip punctuation and check overlap
            fname_clean = f.name.lower().replace('-', '').replace('_', '').replace('.', '')
            for model_name in models:
                # Extract arch name (before first _digit)
                arch_part = model_name.split('_')[0].lower().replace('-', '')
                if arch_part in fname_clean:
                    # For same-arch models (qwen2_896 vs qwen2_2048),
                    # only match if no frt assigned yet or vocab matches
                    if models[model_name]["frt_file"] is None:
                        models[model_name]["frt_file"] = f.name
            # Note: same frt may match multiple models (same tokenizer family)
        except Exception:
            pass

    for f in sht_files:
        try:
            from .sht import load_sht
            s = load_sht(str(f))
            m = s['metadata']
            arch = m['model']['name']
            h_s = m['model']['hidden_size']
            nl_s = m['model']['n_layers']
            name = f"{arch}_{h_s}_{nl_s}L"
            if name in models:
                models[name]["sht_files"].append({
                    "file": f.name, "n_sampled": m['index']['n_sampled'],
                })
        except Exception:
            pass

    for f in decompose_files:
        for model_name in models:
            if model_name.lower().replace('-', '')[:5] in f.name.lower():
                models[model_name]["decompose_file"] = f.name
                break

    # Scan for .trd files
    trd_files = sorted(p.glob("*.trd.npz"))
    for f in trd_files:
        for model_name in models:
            arch = model_name.split('_')[0].lower().replace('-', '')
            if arch in f.name.lower().replace('-', ''):
                models[model_name].setdefault("trd_file", f.name)
                break

    # Scan for direction .npy files
    npy_files = sorted(p.glob("*safety*.npy")) + sorted(p.glob("*comply*.npy"))
    for f in npy_files:
        for model_name in models:
            arch = model_name.split('_')[0].lower().replace('-', '')
            if arch in f.name.lower().replace('-', ''):
                models[model_name].setdefault("direction_files", [])
                models[model_name]["direction_files"].append(f.name)
                break

    coverage = {}
    for name, data in models.items():
        shrt_list = data.get("shrt_files", [])
        single = [s for s in shrt_list if not s.get("has_layer_data")]
        alllayer = [s for s in shrt_list if s.get("has_layer_data")]

        coverage[name] = {
            "hidden": data.get("hidden_size"),
            "layers": data.get("n_layers"),
            "frt": bool(data.get("frt_file")),
            "shrt": bool(single),
            "shrt_n": max((s["n_sampled"] for s in single), default=0),
            "vectors": any(s.get("has_vectors") for s in single),
            "direction": any(s.get("has_direction") for s in shrt_list),
            "dir_layer": next((s["direction_layer"] for s in shrt_list if s.get("has_direction")), None),
            "all_layers": bool(alllayer),
            "alllayers_n": max((s["n_sampled"] for s in alllayer), default=0),
            "sht": bool(data.get("sht_files")),
            "sht_n": max((s["n_sampled"] for s in data.get("sht_files", [])), default=0),
            "decompose": bool(data.get("decompose_file")),
            "trd": bool(data.get("trd_file")),
            "n_directions": len(data.get("direction_files", [])),
        }

    return {"data_dir": data_dir, "n_models": len(coverage), "coverage": coverage}


def silence_profile(backend, shrt_path: str, frt_path: str,
                    db=None, direction_override=None) -> dict:
    """Measure the silence — the baseline state the .shrt measures from.

    Captures the baseline residual vector, projects it into the PCA space
    of the displacement vectors, and onto the safety direction if available.
    Reports where silence sits relative to the token displacement distribution.
    """
    from .shrt import load_shrt, _extract_clean_baseline
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)
    meta = shrt['metadata']
    primary_layer = meta['baseline']['layer']

    # Capture silence residual at primary layer
    clean_baseline = _extract_clean_baseline(backend.tokenizer)
    baseline_fwd = backend.forward(clean_baseline, return_residual=True,
                                    residual_layer=primary_layer)
    silence_vec = baseline_fwd.residual.astype(np.float32)
    silence_norm = float(np.linalg.norm(silence_vec))

    # Load displacement vectors
    vectors = shrt['vectors'].astype(np.float32)
    n_tokens = len(vectors)

    # Build script lookup
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}
    if 'scripts' in shrt:
        scripts = shrt['scripts']
    else:
        scripts = np.array([fl.get(int(tid), 'unknown')
                           for tid in shrt['token_ids']])

    # Absolute positions: silence + displacement = token state
    # The displacement vectors are (token_state - silence_vec)
    # So token_state = silence_vec + displacement
    absolute_vecs = vectors + silence_vec

    # PCA of displacement vectors (same as profile-directions)
    mask = np.array([s not in ('special', 'unknown') for s in scripts])
    vecs_filtered = vectors[mask]
    centered = vecs_filtered - vecs_filtered.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    variance = S**2
    total_var = float(variance.sum())

    # Project silence into PCA space
    silence_centered = silence_vec - vecs_filtered.mean(axis=0)
    silence_pca = Vt @ silence_centered  # projection onto each PC

    # How far is silence from the displacement centroid?
    centroid = vecs_filtered.mean(axis=0) + silence_vec  # absolute centroid
    silence_to_centroid = float(np.linalg.norm(centroid - silence_vec))

    # Silence projection on top PCs
    silence_pc_projections = {}
    for i in range(min(20, len(silence_pca))):
        silence_pc_projections[f"PC{i+1}"] = {
            "projection": round(float(silence_pca[i]), 2),
            "pc_std": round(float(S[i] / np.sqrt(n_tokens)), 2),
            "n_stds_from_center": round(float(abs(silence_pca[i]) / max(S[i] / np.sqrt(n_tokens), 1e-8)), 2),
        }

    result = {
        "model": meta['model']['name'],
        "layer": primary_layer,
        "silence_norm": round(silence_norm, 2),
        "silence_entropy": meta['baseline']['entropy'],
        "silence_top_token": meta['baseline']['top_token'],
        "n_tokens": n_tokens,
        "displacement_mean_norm": round(float(np.linalg.norm(vecs_filtered, axis=1).mean()), 2),
        "silence_to_centroid": round(silence_to_centroid, 2),
        "silence_pc_projections": silence_pc_projections,
    }

    # Safety direction projection
    safety_dir = None
    if direction_override is not None:
        safety_dir = direction_override.astype(np.float32)
        safety_dir = safety_dir / np.linalg.norm(safety_dir)
    elif db is not None:
        try:
            import sqlite3
            if isinstance(db, str):
                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
            else:
                conn = db._conn

            # Find the safety direction for this model at this layer
            # Match by vector dimension (hidden_size * 4 bytes for float32)
            expected_bytes = meta['model']['hidden_size'] * 4
            rows = conn.execute(
                "SELECT d.vector_blob, d.effect_size FROM directions d "
                "WHERE d.layer = ? AND d.name = 'safety' "
                "AND length(d.vector_blob) = ? "
                "ORDER BY d.effect_size DESC LIMIT 1",
                (primary_layer, expected_bytes)
            ).fetchall()

            if rows:
                blob = rows[0]['vector_blob']
                safety_dir = np.frombuffer(blob, dtype=np.float32)
                safety_dir = safety_dir / np.linalg.norm(safety_dir)
        except Exception:
            pass

    if safety_dir is not None:
        silence_safety_proj = float(silence_vec @ safety_dir)
        # Compare to token distribution on safety axis
        token_safety_projs = (vectors + silence_vec) @ safety_dir  # absolute
        disp_safety_projs = vectors @ safety_dir  # displacement only

        result["safety"] = {
            "silence_projection": round(silence_safety_proj, 4),
            "silence_as_pctile": round(float(
                np.mean(token_safety_projs[mask] < silence_safety_proj) * 100), 1),
            "displacement_mean_proj": round(float(disp_safety_projs[mask].mean()), 4),
            "displacement_std_proj": round(float(disp_safety_projs[mask].std()), 4),
            "absolute_mean_proj": round(float(token_safety_projs[mask].mean()), 4),
            "absolute_std_proj": round(float(token_safety_projs[mask].std()), 4),
        }

        # Per-script absolute safety position
        script_safety = {}
        for s in sorted(set(scripts[mask])):
            s_mask = scripts[mask] == s
            if s_mask.sum() < 10:
                continue
            abs_projs = token_safety_projs[mask][s_mask]
            script_safety[s] = {
                "n": int(s_mask.sum()),
                "absolute_mean": round(float(abs_projs.mean()), 2),
                "absolute_std": round(float(abs_projs.std()), 2),
            }
        result["safety"]["by_script"] = script_safety
    else:
        result["safety"] = {"status": "no safety direction available for this model/layer"}

    return result


def safety_rank(shrt_path: str, direction_path: str,
                frt_path: str | None = None,
                trd_path: str | None = None) -> dict:
    """Project all displacement vectors onto a safety direction and rank tokens.

    Reports: per-token safety projection ranked by magnitude, per-script
    statistics, and per-head safety contribution if .trd provided.
    """
    from .shrt import load_shrt

    shrt = load_shrt(shrt_path)
    direction = np.load(direction_path).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    vectors = shrt['vectors'].astype(np.float32)
    token_ids = shrt['token_ids']
    token_texts = shrt['token_texts']
    deltas = shrt['deltas'].astype(np.float32)

    # Scripts from .shrt v0.3 or .frt
    if 'scripts' in shrt:
        scripts = shrt['scripts']
    elif frt_path:
        from .frt import load_frt
        frt = load_frt(frt_path)
        fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
              for i in range(len(frt['token_ids']))}
        scripts = np.array([fl.get(int(tid), 'unknown') for tid in token_ids])
    else:
        scripts = np.array(['unknown'] * len(token_ids))

    # Project all tokens onto safety direction
    projections = vectors @ direction
    n = len(projections)

    # Rank by projection (most negative = most toward compliance)
    rank_comply = np.argsort(projections)  # ascending: most comply first
    rank_refuse = np.argsort(-projections)  # descending: most refuse first

    # Top comply (most negative projection = pushed furthest toward compliance)
    top_comply = []
    for idx in rank_comply[:50]:
        top_comply.append({
            "rank": int(np.where(rank_comply == idx)[0][0]) + 1,
            "id": int(token_ids[idx]),
            "token": str(token_texts[idx])[:30],
            "projection": round(float(projections[idx]), 4),
            "delta": round(float(deltas[idx]), 2),
            "script": str(scripts[idx]),
        })

    # Top refuse (most positive projection)
    top_refuse = []
    for idx in rank_refuse[:50]:
        top_refuse.append({
            "rank": int(np.where(rank_refuse == idx)[0][0]) + 1,
            "id": int(token_ids[idx]),
            "token": str(token_texts[idx])[:30],
            "projection": round(float(projections[idx]), 4),
            "delta": round(float(deltas[idx]), 2),
            "script": str(scripts[idx]),
        })

    # Per-script safety statistics
    script_stats = {}
    for s in sorted(set(scripts)):
        if s in ('special', 'unknown'):
            continue
        s_mask = scripts == s
        if s_mask.sum() < 5:
            continue
        s_proj = projections[s_mask]
        s_delta = deltas[s_mask]
        script_stats[s] = {
            "n": int(s_mask.sum()),
            "mean_proj": round(float(s_proj.mean()), 4),
            "std_proj": round(float(s_proj.std()), 4),
            "mean_delta": round(float(s_delta.mean()), 2),
            "pct_comply": round(float((s_proj < 0).sum() / s_mask.sum() * 100), 1),
            "pct_refuse": round(float((s_proj > 0).sum() / s_mask.sum() * 100), 1),
        }

    result = {
        "model": shrt['metadata']['model']['name'],
        "n_tokens": n,
        "direction_dim": len(direction),
        "overall_mean_proj": round(float(projections.mean()), 4),
        "overall_std_proj": round(float(projections.std()), 4),
        "pct_comply": round(float((projections < 0).sum() / n * 100), 1),
        "pct_refuse": round(float((projections > 0).sum() / n * 100), 1),
        "top_comply": top_comply,
        "top_refuse": top_refuse,
        "by_script": script_stats,
    }

    # Per-head safety contribution from .trd
    if trd_path:
        from .trd import load_trd
        trd = load_trd(trd_path)
        trd_ids = set(int(x) for x in trd['token_ids'])
        # Match tokens between .shrt and .trd
        shrt_id_to_idx = {int(token_ids[i]): i for i in range(len(token_ids))}

        head_safety = {}
        for key in trd:
            if not key.startswith("heads_L"):
                continue
            layer = int(key.split("L")[1])
            hc = trd[key].astype(np.float32)  # [n_trd_tokens, n_heads]
            n_heads = hc.shape[1]

            # For each token in .trd, get its safety projection from .shrt
            trd_projs = []
            for i, tid in enumerate(trd['token_ids']):
                tid = int(tid)
                if tid in shrt_id_to_idx:
                    trd_projs.append(projections[shrt_id_to_idx[tid]])
                else:
                    trd_projs.append(0.0)
            trd_projs = np.array(trd_projs)

            # Correlate each head's contribution with safety projection
            head_corrs = []
            for h in range(n_heads):
                r = float(np.corrcoef(hc[:, h], trd_projs)[0, 1]) if len(trd_projs) > 10 else 0
                head_corrs.append(round(r, 4))

            # Top heads by absolute correlation with safety
            ranked = sorted(range(n_heads), key=lambda h: -abs(head_corrs[h]))
            head_safety[str(layer)] = {
                "n_heads": n_heads,
                "top_safety_heads": [
                    {"head": h, "r_safety": head_corrs[h]}
                    for h in ranked[:5]
                ],
                "all_correlations": head_corrs,
            }

        result["head_safety"] = head_safety

    return result


def discover_safety_direction(backend, db_path: str,
                               n_harmful: int = 100, n_benign: int = 100,
                               seed: int = 42) -> dict:
    """Discover the safety direction natively on a model using DB prompts.

    Uses mean-difference on contrastive residual states at the model's
    primary layer. Reports accuracy, effect size, stability (5 random splits).
    Returns the direction vector for projection.
    """
    import sqlite3
    import random
    from ..discover.directions import find_direction, capture_residual_states
    from ..cartography.templates import build_prompt

    cfg = backend.config
    primary_layer = cfg.safety_layers[-1] if cfg.safety_layers else cfg.n_layers - 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    harmful = [dict(r) for r in conn.execute(
        "SELECT text FROM prompts WHERE is_benign = 0"
    ).fetchall()]
    benign = [dict(r) for r in conn.execute(
        "SELECT text FROM prompts WHERE is_benign = 1"
    ).fetchall()]

    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(benign)

    h_prompts = [build_prompt(r["text"], model_config=cfg) for r in harmful[:n_harmful]]
    b_prompts = [build_prompt(r["text"], model_config=cfg) for r in benign[:n_benign]]

    print(f"  Capturing residuals: {len(h_prompts)} harmful + {len(b_prompts)} benign at L{primary_layer}...")
    states = capture_residual_states(
        backend.model, backend.tokenizer,
        h_prompts + b_prompts,
        layers=[primary_layer],
        backend=backend,
    )

    if primary_layer not in states:
        return {"error": f"No residuals captured at layer {primary_layer}"}

    sl = states[primary_layer]
    h_states = sl[:len(h_prompts)]
    b_states = sl[len(h_prompts):]

    # Primary direction (all data)
    dr = find_direction(h_states, b_states, name="safety", layer=primary_layer)

    # Stability: 5 random sub-splits
    stability_cosines = []
    for i in range(5):
        rng_sub = random.Random(seed + i + 1)
        h_idx = list(range(len(h_states)))
        b_idx = list(range(len(b_states)))
        rng_sub.shuffle(h_idx)
        rng_sub.shuffle(b_idx)
        n_sub = min(20, len(h_idx) // 2, len(b_idx) // 2)
        if n_sub < 5:
            continue
        dr_sub = find_direction(
            h_states[h_idx[:n_sub]], b_states[b_idx[:n_sub]],
            name="safety_sub", layer=primary_layer)
        cos = float(np.dot(dr.direction, dr_sub.direction))
        stability_cosines.append(cos)

    stability = float(np.mean(stability_cosines)) if stability_cosines else 0

    return {
        "model": cfg.model_type,
        "layer": primary_layer,
        "hidden_size": cfg.hidden_size,
        "n_harmful": len(h_prompts),
        "n_benign": len(b_prompts),
        "accuracy": round(dr.separation_accuracy, 4),
        "effect_size": round(dr.effect_size, 2),
        "mean_gap": round(dr.mean_gap, 2),
        "stability": round(stability, 4),
        "direction": dr.direction,
    }


def silence_scatter_html(result: dict, output: str) -> str:
    """Generate a standalone HTML scatter plot from silence_profile result.

    X = displacement magnitude (delta), Y = safety projection.
    Each dot is a script centroid. Silence marked with a cross.
    """
    from pathlib import Path

    safety = result.get('safety', {})
    if 'by_script' not in safety:
        return ""

    # Build data points: script centroids
    points = []
    # We need delta per script — not in the silence result directly
    # Use displacement_mean_norm as the overall reference
    # Actually the silence_profile doesn't store per-script delta means
    # But we have absolute safety. For the scatter we need both.
    # The within_script_analysis has delta per script.
    # For now: use absolute safety position and mark silence.

    silence_safety = safety.get('silence_projection', 0)
    scripts_data = safety.get('by_script', {})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Silence Map: {result['model']}</title>
<style>
body {{ font-family: monospace; background: #111; color: #ccc; margin: 20px; }}
svg {{ background: #1a1a1a; border: 1px solid #333; }}
.label {{ font-size: 11px; fill: #aaa; }}
.axis {{ stroke: #444; }}
.silence {{ fill: #ff4444; }}
.dot {{ opacity: 0.8; }}
.title {{ font-size: 14px; fill: #ddd; }}
</style></head><body>
<h2>Silence Map: {result['model']} (L{result['layer']})</h2>
<p>Y = absolute safety projection. Silence at {silence_safety:.2f}. Red cross = silence.</p>
<svg width="800" height="500" viewBox="0 0 800 500">
"""

    # Compute layout
    all_y = [d['absolute_mean'] for d in scripts_data.values()]
    all_y.append(silence_safety)
    y_min, y_max = min(all_y) - 5, max(all_y) + 5
    y_range = y_max - y_min

    # X position: spread scripts evenly
    script_list = sorted(scripts_data.keys())
    n = len(script_list)

    margin_l, margin_r, margin_t, margin_b = 80, 40, 40, 60
    w = 800 - margin_l - margin_r
    h = 500 - margin_t - margin_b

    def to_x(i):
        return margin_l + (i + 0.5) / n * w

    def to_y(val):
        return margin_t + h - (val - y_min) / y_range * h

    # Safety axis line at 0
    if y_min < 0 < y_max:
        zero_y = to_y(0)
        html += f'<line x1="{margin_l}" y1="{zero_y}" x2="{800-margin_r}" y2="{zero_y}" stroke="#666" stroke-dasharray="4"/>\n'
        html += f'<text x="{margin_l-5}" y="{zero_y-5}" class="label" text-anchor="end">safety=0</text>\n'

    # Silence line
    sy = to_y(silence_safety)
    html += f'<line x1="{margin_l}" y1="{sy}" x2="{800-margin_r}" y2="{sy}" stroke="#ff4444" stroke-dasharray="2" opacity="0.5"/>\n'
    html += f'<text x="{margin_l-5}" y="{sy+4}" class="label" text-anchor="end" fill="#ff6666">silence</text>\n'

    # Script dots
    colors = {
        'latin': '#6699cc', 'code': '#66cc99', 'CJK': '#cc6666',
        'Arabic': '#cc9966', 'Hebrew': '#9966cc', 'Cyrillic': '#cc6699',
        'Japanese': '#cc3333', 'Korean': '#3399cc', 'Thai': '#33cc99',
        'Greek': '#9999cc', 'Devanagari': '#cccc66', 'other': '#999999',
    }

    for i, s in enumerate(script_list):
        d = scripts_data[s]
        x = to_x(i)
        y = to_y(d['absolute_mean'])
        color = colors.get(s, '#aaaaaa')
        r = max(4, min(12, d['n'] ** 0.3))
        html += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" class="dot"><title>{s} n={d["n"]} safety={d["absolute_mean"]:.2f}</title></circle>\n'
        # Error bar (±1 std)
        y_top = to_y(d['absolute_mean'] + d['absolute_std'])
        y_bot = to_y(d['absolute_mean'] - d['absolute_std'])
        html += f'<line x1="{x}" y1="{y_top}" x2="{x}" y2="{y_bot}" stroke="{color}" opacity="0.3"/>\n'
        # Label
        html += f'<text x="{x}" y="{500-margin_b+15}" class="label" text-anchor="middle" transform="rotate(-45,{x},{500-margin_b+15})">{s}</text>\n'

    # Y axis labels
    for val in range(int(y_min), int(y_max) + 1, max(1, int(y_range / 8))):
        y = to_y(val)
        html += f'<text x="{margin_l-10}" y="{y+4}" class="label" text-anchor="end">{val}</text>\n'
        html += f'<line x1="{margin_l}" y1="{y}" x2="{margin_l+5}" y2="{y}" class="axis"/>\n'

    html += f'<text x="400" y="20" class="title" text-anchor="middle">{result["model"]} — absolute safety projection per script (silence = red line)</text>\n'

    html += '</svg></body></html>'

    Path(output).write_text(html)
    return output


def pca_anatomy(shrt_path: str, frt_path: str,
                direction_paths: dict[str, str] | None = None,
                n_components: int = 20) -> dict:
    """Name the unnamed axes of displacement.

    For each of the top N principal components:
      - Per-script mean projection (what scripts does this PC separate?)
      - Top 20 tokens at each extreme (what tokens define this axis?)
      - Cosine with known directions (safety, comply, language) if provided

    direction_paths: {"safety": "path.npy", "comply": "path.npy", ...}
    """
    from .shrt import load_shrt
    from .frt import load_frt

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)

    vectors = shrt['vectors'].astype(np.float32)
    token_ids = shrt['token_ids']
    token_texts = shrt['token_texts']
    deltas = shrt['deltas'].astype(np.float32)

    hidden_size = vectors.shape[1] if len(vectors.shape) > 1 else 0
    if hidden_size == 0:
        return {"error": "No vectors in .shrt file"}

    # Build script lookup
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}
    scripts = np.array([fl.get(int(tid), 'unknown') for tid in token_ids])

    # Filter out special/unknown
    valid_mask = np.array([s not in ('special', 'unknown') for s in scripts])
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 100:
        return {"error": f"Only {len(valid_indices)} valid tokens"}

    all_vecs = vectors[valid_indices]
    all_scripts = scripts[valid_indices]
    all_texts = token_texts[valid_indices]
    all_ids = token_ids[valid_indices]
    all_deltas = deltas[valid_indices]

    # PCA via eigendecomposition of covariance (avoids [N, d] U matrix)
    centered = all_vecs - all_vecs.mean(axis=0)
    cov = centered.T @ centered  # [d, d]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order — reverse for descending variance
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    variance = np.maximum(eigenvalues, 0)  # numerical stability
    total_var = float(variance.sum())
    Vt = eigenvectors.T  # [d, d] — rows are principal components

    n_components = min(n_components, len(Vt))

    # Load known directions if provided
    known_dirs = {}
    if direction_paths:
        for name, path in direction_paths.items():
            try:
                d = np.load(path).astype(np.float32)
                d = d / np.linalg.norm(d)
                known_dirs[name] = d
            except Exception:
                pass

    # Analyze each PC
    components = []
    for k in range(n_components):
        pc = Vt[k]
        var_pct = float(variance[k] / total_var) * 100
        projections = all_vecs @ pc

        # Per-script statistics
        script_proj = {}
        unique_scripts = sorted(set(all_scripts))
        for s in unique_scripts:
            s_mask = all_scripts == s
            if s_mask.sum() < 5:
                continue
            s_p = projections[s_mask]
            script_proj[s] = {
                "n": int(s_mask.sum()),
                "mean": round(float(s_p.mean()), 2),
                "std": round(float(s_p.std()), 2),
            }

        # Sort scripts by mean projection to see what this PC separates
        sorted_scripts = sorted(script_proj.items(), key=lambda x: x[1]["mean"])

        # Top tokens at each extreme
        top_pos_idx = np.argsort(-projections)[:20]
        top_neg_idx = np.argsort(projections)[:20]

        top_positive = []
        for idx in top_pos_idx:
            top_positive.append({
                "token": str(all_texts[idx])[:40],
                "id": int(all_ids[idx]),
                "projection": round(float(projections[idx]), 2),
                "delta": round(float(all_deltas[idx]), 2),
                "script": str(all_scripts[idx]),
            })

        top_negative = []
        for idx in top_neg_idx:
            top_negative.append({
                "token": str(all_texts[idx])[:40],
                "id": int(all_ids[idx]),
                "projection": round(float(projections[idx]), 2),
                "delta": round(float(all_deltas[idx]), 2),
                "script": str(all_scripts[idx]),
            })

        # Cosine with known directions
        dir_cosines = {}
        for name, d in known_dirs.items():
            if len(d) == len(pc):
                cos = float(np.dot(pc, d))
                dir_cosines[name] = round(cos, 4)

        # Script summary: what's at each pole
        if sorted_scripts:
            neg_pole = [f"{s}({v['mean']:.1f})" for s, v in sorted_scripts[:3]]
            pos_pole = [f"{s}({v['mean']:.1f})" for s, v in sorted_scripts[-3:]]
        else:
            neg_pole, pos_pole = [], []

        components.append({
            "pc": k + 1,
            "variance_pct": round(var_pct, 2),
            "cumulative_pct": round(float(np.sum(variance[:k+1]) / total_var) * 100, 2),
            "neg_pole_scripts": neg_pole,
            "pos_pole_scripts": pos_pole,
            "by_script": {s: v for s, v in sorted_scripts},
            "top_positive": top_positive,
            "top_negative": top_negative,
            "direction_cosines": dir_cosines,
        })

    return {
        "model": shrt['metadata']['model']['name'],
        "n_tokens": len(valid_indices),
        "hidden_size": hidden_size,
        "n_components": n_components,
        "total_variance_explained": round(
            float(np.sum(variance[:n_components]) / total_var) * 100, 2),
        "components": components,
    }


def pca_survey(shrt_frt_pairs: list[tuple[str, str]],
               n_components: int = 10) -> dict:
    """Compare PCA structure across models by script ordering.

    For each model, compute top PCs and their script mean projections.
    Then correlate script orderings across models to find shared vs unique axes.

    shrt_frt_pairs: [(shrt_path, frt_path), ...]
    """
    from .shrt import load_shrt
    from .frt import load_frt
    def _spearman(a, b):
        """Spearman rank correlation without scipy."""
        a, b = np.array(a), np.array(b)
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        if denom < 1e-12:
            return 0.0
        return float((ra * rb).sum() / denom)

    # Shared script list for comparison
    common_scripts = ['Arabic', 'CJK', 'Cyrillic', 'Greek', 'Hebrew',
                      'Japanese', 'Korean', 'Thai', 'code', 'latin', 'other']

    models = []
    for shrt_path, frt_path in shrt_frt_pairs:
        shrt = load_shrt(shrt_path)
        frt = load_frt(frt_path)
        vectors = shrt['vectors'].astype(np.float32)
        token_ids = shrt['token_ids']
        hidden_size = vectors.shape[1] if len(vectors.shape) > 1 else 0
        if hidden_size == 0:
            continue

        fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
              for i in range(len(frt['token_ids']))}
        scripts = np.array([fl.get(int(tid), 'unknown') for tid in token_ids])
        valid_mask = np.array([s not in ('special', 'unknown') for s in scripts])
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 100:
            continue

        all_vecs = vectors[valid_indices]
        all_scripts = scripts[valid_indices]

        centered = all_vecs - all_vecs.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        variance = np.maximum(eigenvalues, 0)
        total_var = float(variance.sum())
        Vt = eigenvectors.T

        k = min(n_components, len(Vt))
        model_name = shrt['metadata']['model']['name']

        # Script indices
        script_indices = {}
        for s in common_scripts:
            mask = all_scripts == s
            if mask.sum() >= 5:
                script_indices[s] = np.where(mask)[0]

        pcs = []
        for c in range(k):
            pc = Vt[c]
            projections = all_vecs @ pc
            var_pct = float(variance[c] / total_var) * 100

            script_means = {}
            for s, idx in script_indices.items():
                script_means[s] = float(projections[idx].mean())

            # Sort to find poles
            sorted_s = sorted(script_means.items(), key=lambda x: x[1])
            neg_pole = sorted_s[0][0] if sorted_s else ""
            pos_pole = sorted_s[-1][0] if sorted_s else ""

            pcs.append({
                "var_pct": round(var_pct, 2),
                "script_means": script_means,
                "neg_pole": neg_pole,
                "pos_pole": pos_pole,
            })

        # PCs for 50%
        cumulative = np.cumsum(variance) / total_var
        pcs_50 = int(np.searchsorted(cumulative, 0.5)) + 1

        models.append({
            "name": model_name,
            "hidden_size": hidden_size,
            "n_tokens": len(valid_indices),
            "pcs_for_50pct": pcs_50,
            "pcs": pcs,
        })

    if len(models) < 2:
        return {"error": "Need at least 2 models for comparison"}

    # Cross-model PC matching: for each pair of models, find which PCs
    # have the most similar script ordering (by Spearman correlation)
    matches = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            for p1_idx, p1 in enumerate(m1['pcs']):
                for p2_idx, p2 in enumerate(m2['pcs']):
                    # Build vectors of script means for shared scripts
                    shared = sorted(set(p1['script_means']) & set(p2['script_means']))
                    if len(shared) < 11:
                        continue
                    v1 = [p1['script_means'][s] for s in shared]
                    v2 = [p2['script_means'][s] for s in shared]
                    rho = _spearman(v1, v2)
                    if abs(rho) >= 0.7:
                        matches.append({
                            "model_a": m1['name'],
                            "pc_a": p1_idx + 1,
                            "var_a": p1['var_pct'],
                            "model_b": m2['name'],
                            "pc_b": p2_idx + 1,
                            "var_b": p2['var_pct'],
                            "spearman_rho": round(float(rho), 3),
                            "shared_scripts": len(shared),
                            "pole_a": f"{p1['neg_pole']}→{p1['pos_pole']}",
                            "pole_b": f"{p2['neg_pole']}→{p2['pos_pole']}",
                        })

    matches.sort(key=lambda x: -abs(x['spearman_rho']))

    # Summary table per model
    summary = []
    for m in models:
        summary.append({
            "name": m['name'],
            "hidden_size": m['hidden_size'],
            "n_tokens": m['n_tokens'],
            "pcs_for_50pct": m['pcs_for_50pct'],
            "pc1_pct": m['pcs'][0]['var_pct'],
            "pc1_poles": f"{m['pcs'][0]['neg_pole']}→{m['pcs'][0]['pos_pole']}",
        })

    return {
        "n_models": len(models),
        "summary": summary,
        "matches": matches,
        "models": models,
    }


def pca_depth(mri_path: str, n_sample: int = 5000,
              layers: list[int] | None = None) -> dict:
    """PCA structure at every layer of an MRI.

    Shows how dimensionality and dominant axes evolve through the network.
    Reports PC1%, PCs for 50%, and dominant script poles at each layer.
    """
    from .mri import load_mri
    import json
    from pathlib import Path

    mri_dir = Path(mri_path)
    meta = json.loads((mri_dir / 'metadata.json').read_text())
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('mode', 'unknown')

    tokens = dict(np.load(mri_dir / 'tokens.npz', allow_pickle=True))
    scripts = tokens['scripts']

    common = ['Arabic', 'CJK', 'Cyrillic', 'Japanese', 'Korean',
              'Thai', 'Hebrew', 'code', 'latin', 'other']
    valid = np.array([str(s) in common for s in scripts])
    valid_idx = np.where(valid)[0]

    rng = np.random.RandomState(42)
    if len(valid_idx) > n_sample:
        idx = rng.choice(valid_idx, n_sample, replace=False)
    else:
        idx = valid_idx
    ss = scripts[idx]

    if layers is None:
        layers = list(range(n_layers))

    results = []
    for layer in layers:
        f = mri_dir / f'L{layer:02d}_exit.npy'
        if not f.exists():
            continue
        vecs = np.load(str(f))[idx].astype(np.float32)
        centered = vecs - vecs.mean(axis=0)
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        var = S ** 2
        total = float(var.sum())
        cum = np.cumsum(var) / total
        pcs50 = int(np.searchsorted(cum, 0.5)) + 1

        pc1 = Vt[0]
        proj = vecs @ pc1
        script_means = {}
        for s in common:
            mask = ss == s
            if mask.sum() >= 3:
                script_means[s] = round(float(proj[mask].mean()), 2)

        sorted_s = sorted(script_means.items(), key=lambda x: x[1])

        results.append({
            "layer": layer,
            "pc1_pct": round(float(var[0] / total) * 100, 1),
            "pc2_pct": round(float(var[1] / total) * 100, 1),
            "pc3_pct": round(float(var[2] / total) * 100, 1) if len(var) > 2 else 0,
            "pcs_for_50pct": pcs50,
            "total_variance": round(total, 0),
            "neg_pole": sorted_s[0][0] if sorted_s else "",
            "pos_pole": sorted_s[-1][0] if sorted_s else "",
            "script_means": script_means,
        })

    return {
        "model": model_name,
        "mode": mode,
        "n_layers": n_layers,
        "n_sampled": len(idx),
        "layers": results,
    }


def gate_analysis(mri_path: str, *, n_sample: int | None = None, _mri=None) -> dict:
    """Which MLP neurons fire and how they specialize.

    Per-layer: diversity, concentration, top neurons, per-script differences.
    """
    from .mri import load_mri
    from collections import Counter

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    n_tok = meta['capture']['n_tokens']
    model_name = meta['model']['name']
    mode = meta['capture']['mode']
    gate_format = meta['capture'].get('gate_format', 'topk')

    # Check for gate data in either format
    has_full_gates = "gate_L0" in mri
    has_topk_gates = "gate_indices_L0" in mri
    if not has_full_gates and not has_topk_gates:
        return {"error": "MRI has no gate data"}

    scripts = mri.get('scripts', np.array(['?'] * n_tok))

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    import sys
    sampled_scripts = scripts[idx]
    gate_k = 32  # default for analysis reporting

    layers = []
    for i in range(n_layers):
        print(f"  gate_analysis L{i}/{n_layers}", end="\r", file=sys.stderr)
        if has_full_gates and f"gate_L{i}" in mri:
            # Full gate values: [n_tok, intermediate]
            g_full = mri[f"gate_L{i}"][idx].astype(np.float32)
            # Derive top-K for compatibility
            gate_k = 32
            g_topk_idx = np.argpartition(-np.abs(g_full), gate_k, axis=1)[:, :gate_k]
            g_idx = g_topk_idx
            g_val = np.take_along_axis(g_full, g_topk_idx, axis=1)
        elif has_topk_gates and f"gate_indices_L{i}" in mri:
            g_idx = mri[f"gate_indices_L{i}"][idx]
            g_val = mri[f"gate_values_L{i}"][idx].astype(np.float32)
            gate_k = g_idx.shape[1]
        else:
            continue

        all_neurons = g_idx.flatten()
        unique = len(set(all_neurons.tolist()))

        top1 = g_idx[:, 0]
        top1_counts = Counter(top1.tolist())
        mode_neuron, mode_count = top1_counts.most_common(1)[0]
        concentration = mode_count / n_sample

        neuron_freq = Counter(all_neurons.tolist())
        top_neurons = neuron_freq.most_common(10)

        mean_act = float(np.abs(g_val).mean())
        max_act = float(np.abs(g_val).max())

        script_top1 = {}
        for s in ['latin', 'CJK', 'Cyrillic', 'code', 'Arabic', 'Japanese']:
            mask = sampled_scripts == s
            if mask.sum() < 10:
                continue
            s_top1 = Counter(g_idx[mask, 0].tolist())
            s_mode, s_count = s_top1.most_common(1)[0]
            script_top1[s] = {"neuron": int(s_mode),
                              "concentration": round(s_count / mask.sum(), 3)}

        layers.append({
            "layer": i,
            "unique_neurons": unique,
            "total_entries": n_sample * gate_k,
            "top1_concentration": round(concentration, 3),
            "top1_neuron": int(mode_neuron),
            "mean_activation": round(mean_act, 3),
            "max_activation": round(max_act, 3),
            "top_neurons": [(int(n), c) for n, c in top_neurons],
            "script_top1": script_top1,
        })

    print(" " * 40, end="\r", file=sys.stderr)

    return {
        "model": model_name, "mode": mode, "n_tokens": n_sample,
        "gate_k": gate_k, "n_layers": len(layers), "layers": layers,
    }


def attention_analysis(mri_path: str, *, n_sample: int | None = None, _mri=None) -> dict:
    """Where does the token attend in template mode?

    Per-layer: self vs prefix vs suffix weight, entropy, per-head focus.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    n_tok = meta['capture']['n_tokens']
    n_heads = meta['model']['n_heads']
    model_name = meta['model']['name']
    mode = meta['capture']['mode']
    token_pos = meta['capture'].get('token_pos', 0)
    seq_len = meta['capture'].get('seq_len', 1)

    if not meta['capture'].get('has_attention', False):
        return {"error": "MRI has no attention data"}
    # New format: attn_weights_L0, legacy: attn_L0
    attn_prefix = "attn_weights_L" if "attn_weights_L0" in mri else "attn_L"
    if f"{attn_prefix}0" not in mri:
        return {"error": "MRI has no attention data loaded"}

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    import sys
    prefix_range = list(range(token_pos))
    suffix_range = list(range(token_pos + 1, seq_len))

    layers = []
    for i in range(n_layers):
        print(f"  attention_analysis L{i}/{n_layers}", end="\r", file=sys.stderr)
        akey = f"{attn_prefix}{i}"
        if akey not in mri:
            continue
        aw = mri[akey][idx].astype(np.float32)

        self_w = float(aw[:, :, token_pos].mean())
        prefix_w = float(aw[:, :, prefix_range].sum(axis=2).mean()) if prefix_range else 0.0
        suffix_w = float(aw[:, :, suffix_range].sum(axis=2).mean()) if suffix_range else 0.0

        aw_clipped = np.clip(aw, 1e-8, 1.0)
        entropy = float(-np.sum(aw_clipped * np.log2(aw_clipped), axis=2).mean())

        head_focus = []
        for h in range(n_heads):
            mean_per_pos = aw[:, h, :].mean(axis=0)
            peak_pos = int(mean_per_pos.argmax())
            peak_val = float(mean_per_pos[peak_pos])
            if peak_pos < token_pos:
                focus = f"prefix[{peak_pos}]"
            elif peak_pos == token_pos:
                focus = "self"
            else:
                focus = f"suffix[{peak_pos - token_pos - 1}]"
            head_focus.append({"head": h, "focus": focus, "weight": round(peak_val, 3)})

        layers.append({
            "layer": i, "self_weight": round(self_w, 4),
            "prefix_weight": round(prefix_w, 4), "suffix_weight": round(suffix_w, 4),
            "entropy": round(entropy, 3), "head_focus": head_focus,
        })

    print(" " * 40, end="\r", file=sys.stderr)

    return {
        "model": model_name, "mode": mode, "n_tokens": n_sample,
        "n_heads": n_heads, "seq_len": seq_len, "token_pos": token_pos,
        "n_layers": len(layers), "layers": layers,
    }


def bandwidth_efficiency(mri_path: str, *, n_sample: int | None = None,
                         delta_threshold: float = 0.01) -> dict:
    """Per-token bandwidth efficiency: what fraction of model bytes do useful work?

    For each token, estimates:
      - MLP active bytes: (active_neurons / total_neurons) × MLP weight size
      - Skippable layers: layers where delta_norm < threshold × exit_norm
      - Total active bytes vs total model bytes

    Returns bandwidth efficiency = active_bytes / total_bytes.
    The gap is wasted bandwidth — bytes loaded from memory, never used.
    """
    from .mri import load_mri
    from pathlib import Path
    import json

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    hidden = meta['model']['hidden_size']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    gate_k = meta['capture'].get('gate_k', 0)
    n_tok = len(mri['token_ids'])

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    # Compute weight sizes per layer from stored weights
    p = Path(mri_path)
    wdir = p / "weights"
    layer_weight_bytes = {}
    attn_bytes_per_layer = 0
    mlp_bytes_per_layer = 0
    for i in range(n_layers):
        ldir = wdir / f"L{i:02d}" if wdir.exists() else None
        if ldir and ldir.exists():
            attn_b = sum(f.stat().st_size for f in ldir.glob("*_proj.npy")
                         if f.stem in ("q_proj", "k_proj", "v_proj", "o_proj"))
            mlp_b = sum(f.stat().st_size for f in ldir.glob("*_proj.npy")
                        if f.stem in ("gate_proj", "up_proj", "down_proj"))
            layer_weight_bytes[i] = {"attn": attn_b, "mlp": mlp_b, "total": attn_b + mlp_b}
            if i == 0:
                attn_bytes_per_layer = attn_b
                mlp_bytes_per_layer = mlp_b

    total_model_bytes = sum(v["total"] for v in layer_weight_bytes.values())
    embed_bytes = (p / "embedding.npy").stat().st_size if (p / "embedding.npy").exists() else 0
    lmhead_bytes = (p / "lmhead_raw.npy").stat().st_size if (p / "lmhead_raw.npy").exists() else 0
    total_model_bytes += embed_bytes + lmhead_bytes

    # Infer intermediate_size from gate weights
    if wdir.exists() and (wdir / "L00" / "gate_proj.npy").exists():
        gate_w = np.load(wdir / "L00" / "gate_proj.npy", mmap_mode='r')
        intermediate_size = gate_w.shape[0]
    else:
        intermediate_size = hidden * 4  # fallback estimate

    # Per-layer analysis
    layer_results = []
    prev_exit = None
    for i in range(n_layers):
        exit_key = f"exit_L{i}"
        if exit_key not in mri:
            continue
        exits = mri[exit_key][idx].astype(np.float32)
        exit_norms = np.linalg.norm(exits, axis=1)

        # Layer delta
        if prev_exit is not None:
            deltas = exits - prev_exit
            delta_norms = np.linalg.norm(deltas, axis=1)
        else:
            delta_norms = exit_norms

        # Fraction of tokens where this layer is skippable
        if prev_exit is not None:
            prev_norms = np.linalg.norm(prev_exit, axis=1)
            skippable = (delta_norms < delta_threshold * np.maximum(prev_norms, 1e-8))
            skip_frac = float(skippable.mean())
        else:
            skip_frac = 0.0

        # MLP active fraction from gates
        gate_key_full = f"gate_L{i}"
        gate_key_topk = f"gate_indices_L{i}"
        if gate_key_full in mri:
            # Full gate activations: count neurons above threshold
            g = mri[gate_key_full][idx].astype(np.float32)
            active_per_token = np.sum(np.abs(g) > 1.0, axis=1)
            mean_active = float(active_per_token.mean())
            mlp_active_frac = mean_active / intermediate_size
        elif gate_k and gate_key_topk in mri:
            g_idx = mri[gate_key_topk][idx]
            unique_per_token = np.array([len(set(row)) for row in g_idx])
            mean_active = float(unique_per_token.mean())
            mlp_active_frac = mean_active / intermediate_size
        else:
            mlp_active_frac = 1.0  # no gate data, assume all active
            mean_active = intermediate_size

        # Active bytes for this layer
        wb = layer_weight_bytes.get(i, {"attn": attn_bytes_per_layer,
                                         "mlp": mlp_bytes_per_layer,
                                         "total": attn_bytes_per_layer + mlp_bytes_per_layer})
        # Attention is dense (all heads needed), MLP is sparse (only active neurons)
        active_bytes = wb["attn"] + wb["mlp"] * mlp_active_frac
        if skip_frac > 0.5:
            # Majority skippable — this layer is mostly wasted
            active_bytes *= (1.0 - skip_frac)

        layer_results.append({
            "layer": i,
            "skip_fraction": round(skip_frac, 3),
            "mlp_active_fraction": round(mlp_active_frac, 4),
            "mlp_active_neurons": round(mean_active, 1),
            "mlp_total_neurons": intermediate_size,
            "active_bytes": int(active_bytes),
            "total_bytes": wb["total"],
            "efficiency": round(active_bytes / max(wb["total"], 1), 4),
        })

        prev_exit = exits

    total_active = sum(r["active_bytes"] for r in layer_results) + embed_bytes + lmhead_bytes
    efficiency = total_active / max(total_model_bytes, 1)

    return {
        "model": model_name,
        "mode": mode,
        "n_tokens": n_sample,
        "total_model_bytes": total_model_bytes,
        "total_active_bytes": total_active,
        "bandwidth_efficiency": round(efficiency, 4),
        "wasted_fraction": round(1.0 - efficiency, 4),
        "intermediate_size": intermediate_size,
        "gate_k": gate_k,
        "layers": layer_results,
    }


def lookup_fraction(mri_path: str, *, n_sample: int | None = None) -> dict:
    """How much of language modeling is a table lookup?

    For each token, compares:
      - embedding prediction: lmhead @ embedding[token_id] (no computation)
      - full prediction: lmhead @ norm(exit_L[-1]) (24 layers of computation)

    If top-1 matches, the layers added nothing — the token was lookup-solvable.
    """
    from .mri import load_mri
    import sys

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')

    if 'lmhead_raw' not in mri or 'embedding' not in mri:
        return {"error": "MRI missing lmhead_raw or embedding"}

    lmhead = mri['lmhead_raw'].astype(np.float32)  # [vocab, hidden]
    embedding = mri['embedding'].astype(np.float32)  # [vocab, hidden]
    token_ids = mri['token_ids']
    n_tok = len(token_ids)
    vocab_size = lmhead.shape[0]

    final_norm_w = mri.get('norm_final')
    if final_norm_w is not None:
        final_norm_w = np.array(final_norm_w).astype(np.float32)

    last_layer = f"exit_L{n_layers - 1}"
    if last_layer not in mri:
        return {"error": f"Missing {last_layer}"}

    # OOM guard: [n_sample, vocab] logits at float32 per layer
    if n_sample is None:
        n_sample = min(n_tok, 5000)
    peak_bytes = n_sample * vocab_size * 4
    if peak_bytes > 4 * 1024**3:
        print(f"  WARNING: lookup_fraction peak alloc ~{peak_bytes / 1024**3:.1f} GB "
              f"({n_sample} tokens x {vocab_size} vocab)", file=sys.stderr)

    if n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))

    sampled_ids = token_ids[idx]
    exit_states = mri[last_layer][idx].astype(np.float32)

    # Apply RMSNorm to exit states
    if final_norm_w is not None:
        rms = np.sqrt(np.mean(exit_states ** 2, axis=1, keepdims=True) + 1e-6)
        exit_states = exit_states / rms * final_norm_w

    # Embedding prediction: lmhead @ embedding[tid]
    emb_vecs = embedding[sampled_ids]  # [n_sample, hidden]
    emb_logits = emb_vecs @ lmhead.T  # [n_sample, vocab]
    emb_top1 = np.argmax(emb_logits, axis=1)

    # Full prediction: lmhead @ normed_exit
    full_logits = exit_states @ lmhead.T  # [n_sample, vocab]
    full_top1 = np.argmax(full_logits, axis=1)

    # Compare
    match = emb_top1 == full_top1
    lookup_count = int(match.sum())
    compute_count = n_sample - lookup_count

    # Per-script breakdown
    scripts = mri.get('scripts', np.array(['?'] * n_tok))
    sampled_scripts = scripts[idx]
    script_stats = {}
    for s in sorted(set(sampled_scripts)):
        mask = sampled_scripts == s
        if mask.sum() < 5:
            continue
        s_match = match[mask].sum()
        script_stats[str(s)] = {
            "n": int(mask.sum()),
            "lookup": int(s_match),
            "fraction": round(float(s_match / mask.sum()), 3),
        }

    # Per-layer: at which layer does the prediction diverge from embedding?
    layer_divergence = []
    for i in range(n_layers):
        exit_key = f"exit_L{i}"
        if exit_key not in mri:
            continue
        states = mri[exit_key][idx].astype(np.float32)
        if final_norm_w is not None:
            rms = np.sqrt(np.mean(states ** 2, axis=1, keepdims=True) + 1e-6)
            states = states / rms * final_norm_w
        layer_logits = states @ lmhead.T
        layer_top1 = np.argmax(layer_logits, axis=1)
        layer_match = (layer_top1 == emb_top1).sum()
        layer_divergence.append({
            "layer": i,
            "matches_embedding": int(layer_match),
            "fraction": round(float(layer_match / n_sample), 3),
        })

    return {
        "model": model_name,
        "mode": mode,
        "n_tokens": n_sample,
        "lookup_solvable": lookup_count,
        "compute_needed": compute_count,
        "lookup_fraction": round(lookup_count / n_sample, 4),
        "by_script": script_stats,
        "by_layer": layer_divergence,
    }


def distribution_drift(mri_path: str, *, n_sample: int | None = None,
                       top_k: int = 20) -> dict:
    """What do the frozen zone whispers change?

    Computes the full output distribution (via lmhead) at each layer.
    Reports: does top-1 change? Does the distribution shift? How much
    probability mass moves between layers?
    """
    from .mri import load_mri
    import sys

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    n_tok = len(mri['token_ids'])
    vocab_size = meta['model'].get('vocab_size', 0)

    if 'lmhead_raw' not in mri:
        return {"error": "MRI missing lmhead_raw"}
    lmhead = mri['lmhead_raw'].astype(np.float32)
    vocab_size = vocab_size or lmhead.shape[0]
    final_norm_w = mri.get('norm_final')
    if final_norm_w is not None:
        final_norm_w = np.array(final_norm_w).astype(np.float32)

    # OOM guard: [n_sample, vocab] logits at float32 is the peak per-layer alloc
    # Cap at 5000 by default; warn if requested size would exceed 4 GB
    if n_sample is None:
        n_sample = min(n_tok, 5000)
    peak_bytes = n_sample * vocab_size * 4 * 3  # logits + probs + prev_probs
    if peak_bytes > 4 * 1024**3:
        print(f"  WARNING: distribution_drift peak alloc ~{peak_bytes / 1024**3:.1f} GB "
              f"({n_sample} tokens x {vocab_size} vocab)", file=sys.stderr)

    if n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    def _get_probs(layer_idx):
        states = mri[f"exit_L{layer_idx}"][idx].astype(np.float32)
        if final_norm_w is not None:
            rms = np.sqrt(np.mean(states ** 2, axis=1, keepdims=True) + 1e-6)
            states = states / rms * final_norm_w
        logits = states @ lmhead.T
        # Stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        return probs

    layers = []
    prev_probs = None
    prev_top1 = None
    for i in range(n_layers):
        if f"exit_L{i}" not in mri:
            continue
        probs = _get_probs(i)
        top1 = np.argmax(probs, axis=1)

        entry = {"layer": i}

        if prev_probs is not None:
            # Top-1 changed?
            top1_changed = float((top1 != prev_top1).mean())
            # KL divergence from previous layer
            kl = np.sum(probs * np.log((probs + 1e-12) / (prev_probs + 1e-12)), axis=1)
            # Total variation distance
            tvd = 0.5 * np.abs(probs - prev_probs).sum(axis=1)
            # Entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-12), axis=1)

            entry["top1_changed"] = round(top1_changed, 4)
            entry["mean_kl"] = round(float(kl.mean()), 6)
            entry["mean_tvd"] = round(float(tvd.mean()), 4)
            entry["mean_entropy"] = round(float(entropy.mean()), 2)
        else:
            entry["top1_changed"] = 1.0
            entry["mean_kl"] = 0.0
            entry["mean_tvd"] = 0.0
            entropy = -np.sum(probs * np.log2(probs + 1e-12), axis=1)
            entry["mean_entropy"] = round(float(entropy.mean()), 2)

        layers.append(entry)
        prev_probs = probs
        prev_top1 = top1

    return {
        "model": model_name, "mode": mode, "n_tokens": n_sample,
        "n_layers": len(layers), "layers": layers,
    }


def retrieval_horizon(mri_path: str, *, n_sample: int | None = None) -> dict:
    """How far back does the token look?

    For each token at each layer, measures: which template position gets
    the most attention weight? The distance from token_pos to that position
    is the retrieval horizon — how far back the model reaches.

    Template mode only (raw/naked have seq_len=1).
    """
    from .mri import load_mri

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    n_heads = meta['model']['n_heads']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    token_pos = meta['capture'].get('token_pos', 0)
    seq_len = meta['capture'].get('seq_len', 1)
    n_tok = len(mri['token_ids'])

    if seq_len <= 1:
        return {"error": "Retrieval horizon requires template mode (seq_len > 1)"}

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    layers = []
    for i in range(n_layers):
        akey = f"attn_weights_L{i}"
        if akey not in mri:
            akey = f"attn_L{i}"  # legacy
            if akey not in mri:
                continue

        aw = mri[akey][idx].astype(np.float32)  # [n_sample, heads, seq_len]

        # Mean attention weight per position (averaged over heads and tokens)
        mean_per_pos = aw.mean(axis=(0, 1))  # [seq_len]

        # Per-head: which position gets most attention (averaged over tokens)
        head_peaks = []
        for h in range(n_heads):
            hw = aw[:, h, :].mean(axis=0)  # [seq_len]
            peak = int(np.argmax(hw))
            head_peaks.append(peak)

        # Retrieval horizon per token: distance from token_pos to argmax position
        # Average over heads: weighted position
        per_token_peaks = np.argmax(aw.mean(axis=1), axis=1)  # [n_sample]
        distances = token_pos - per_token_peaks  # positive = looking back
        mean_distance = float(distances.mean())

        # Self-attention fraction
        self_frac = float(aw[:, :, token_pos].mean())

        layers.append({
            "layer": i,
            "mean_retrieval_distance": round(mean_distance, 2),
            "self_attention": round(self_frac, 4),
            "peak_position_distribution": [round(float(mean_per_pos[p]), 4)
                                           for p in range(seq_len)],
            "head_peak_positions": head_peaks,
        })

    return {
        "model": model_name, "mode": mode, "n_tokens": n_sample,
        "n_heads": n_heads, "seq_len": seq_len, "token_pos": token_pos,
        "n_layers": len(layers), "layers": layers,
    }


def layer_opposition(mri_path: str, *, n_sample: int | None = None) -> dict:
    """Do MLP and attention oppose each other within each layer?

    Computes MLP output DIRECTLY from stored gate * up * down_proj (no derivation).
    Computes attention output from stored attn_out.
    Reports cosine between them, norms, and cancellation ratio.
    """
    from .mri import load_mri

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    hidden = meta['model']['hidden_size']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    n_tok = len(mri['token_ids'])

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    layers = []
    for i in range(n_layers):
        # Need: gate, up, down_proj, attn_out, exit delta
        gate_key = f"gate_L{i}"
        up_key = f"up_L{i}"
        down_key = f"down_proj_L{i}"
        attn_key = f"attn_out_L{i}"
        exit_key = f"exit_L{i}"
        exit_prev_key = f"exit_L{i-1}" if i > 0 else None

        if gate_key not in mri or up_key not in mri or down_key not in mri:
            continue
        if attn_key not in mri or exit_key not in mri:
            continue

        import sys
        print(f"  L{i}/{n_layers}...", end="\r", file=sys.stderr)

        g = mri[gate_key][idx].astype(np.float32)
        u = mri[up_key][idx].astype(np.float32)
        down = mri[down_key].astype(np.float32)  # [hidden, intermediate]
        attn = mri[attn_key][idx].astype(np.float32)

        # Direct MLP output: (gate * up) @ down_proj.T
        # down_proj stored as [hidden, intermediate] from identity probing
        neuron_act = g * u  # [n_sample, intermediate]
        mlp_out = neuron_act @ down.T  # [n_sample, inter] @ [inter, hidden] = [n_sample, hidden]

        exit_curr = mri[exit_key][idx].astype(np.float32)
        if exit_prev_key and exit_prev_key in mri:
            exit_prev = mri[exit_prev_key][idx].astype(np.float32)
        else:
            exit_prev = np.zeros_like(exit_curr)

        delta = exit_curr - exit_prev

        # For L0: delta = embedding + attn + mlp (embedding is the residual input)
        if i == 0 and "embedding" in mri:
            emb = mri["embedding"].astype(np.float32)
            sampled_tids = mri["token_ids"][idx]
            emb_vecs = emb[sampled_tids]
            recon = emb_vecs + attn + mlp_out
            verification_err = float(np.abs(exit_curr - recon).mean())
        else:
            recon = attn + mlp_out
            verification_err = float(np.abs(delta - recon).mean())

        mn = np.linalg.norm(mlp_out, axis=1)
        an = np.linalg.norm(attn, axis=1)
        delta_norm = np.linalg.norm(delta, axis=1)

        # Cosine between MLP and attention output
        dot_ma = (mlp_out * attn).sum(axis=1)
        cos_mlp_attn = float((dot_ma / (mn * an + 1e-8)).mean())

        # Relative verification error
        rel_err = verification_err / max(float(delta_norm.mean()), 1e-8)

        # Cancellation
        larger_norm = np.maximum(mn, an)
        cancellation = float((1.0 - delta_norm / (larger_norm + 1e-8)).mean())

        layers.append({
            "layer": i,
            "cos_mlp_attn": round(cos_mlp_attn, 4),
            "mlp_norm": round(float(mn.mean()), 2),
            "attn_norm": round(float(an.mean()), 2),
            "delta_norm": round(float(delta_norm.mean()), 2),
            "cancellation": round(cancellation, 3),
            "verification_error": round(verification_err, 4),
            "relative_error": round(rel_err, 4),
        })

    return {
        "model": model_name,
        "mode": mode,
        "n_tokens": n_sample,
        "n_layers": len(layers),
        "layers": layers,
    }


def cross_model(mri_paths: list[str], *, n_sample: int | None = None) -> dict:
    """Cross-model comparison on shared vocabulary.

    Matches tokens by decoded TEXT across models with different tokenizers.
    Compares displacement, gradient, crystal activation on the shared set.
    """
    from .mri import load_mri
    from scipy.stats import spearmanr

    if len(mri_paths) < 2:
        return {"error": "Need at least 2 MRI paths"}

    # Load all MRIs and build text→index maps
    mris = []
    for p in mri_paths:
        m = load_mri(p)
        meta = m['metadata']
        texts = [str(t) for t in m['token_texts']]
        text_to_idx = {}
        for i, t in enumerate(texts):
            if t.strip() and t not in text_to_idx:
                text_to_idx[t] = i
        n_layers = meta['model']['n_layers']
        # Unique label from directory structure (model_dir/mode.mri)
        from pathlib import Path
        pp = Path(p)
        label = f"{pp.parent.name}/{pp.name.replace('.mri','')}"
        mris.append({
            "path": p,
            "name": label,
            "mri": m,
            "text_to_idx": text_to_idx,
            "n_layers": n_layers,
        })

    # Find shared vocabulary (tokens present in ALL models by text)
    shared_texts = set(mris[0]["text_to_idx"].keys())
    for mi in mris[1:]:
        shared_texts &= set(mi["text_to_idx"].keys())
    shared_texts = sorted(shared_texts)

    if n_sample and n_sample < len(shared_texts):
        rng = np.random.RandomState(42)
        shared_texts = [shared_texts[i] for i in sorted(
            rng.choice(len(shared_texts), n_sample, replace=False))]

    n_shared = len(shared_texts)
    if n_shared < 10:
        return {"error": f"Only {n_shared} shared tokens"}

    # Compute per-model displacement and gradient on shared set
    model_data = []
    for mi in mris:
        m = mi["mri"]
        idx = [mi["text_to_idx"][t] for t in shared_texts]
        n_layers = mi["n_layers"]

        exit_last = m[f"exit_L{n_layers - 1}"][idx].astype(np.float32)
        displacements = np.linalg.norm(exit_last, axis=1)

        from pathlib import Path
        eg_path = Path(mi["path"]) / "embedding_grad.npy"
        if eg_path.exists():
            eg = np.load(eg_path)[idx].astype(np.float32)
            grad_norms = np.linalg.norm(eg, axis=1)
        else:
            grad_norms = None

        model_data.append({
            "name": mi["name"],
            "path": mi["path"],
            "displacements": displacements,
            "grad_norms": grad_norms,
        })

    # Pairwise comparisons
    comparisons = []
    for i in range(len(model_data)):
        for j in range(i + 1, len(model_data)):
            a, b = model_data[i], model_data[j]
            disp_rho, _ = spearmanr(a["displacements"], b["displacements"])

            entry = {
                "model_a": a["name"],
                "model_b": b["name"],
                "displacement_rho": round(float(disp_rho), 4),
            }

            if a["grad_norms"] is not None and b["grad_norms"] is not None:
                grad_rho, _ = spearmanr(a["grad_norms"], b["grad_norms"])
                entry["gradient_rho"] = round(float(grad_rho), 4)

            # Top shart overlap
            top_n = min(100, n_shared // 5)
            top_a = set(np.argsort(-a["displacements"])[:top_n])
            top_b = set(np.argsort(-b["displacements"])[:top_n])
            entry["top_overlap"] = len(top_a & top_b)
            entry["top_n"] = top_n

            comparisons.append(entry)

    # Per-model stats on shared set
    per_model = []
    for md in model_data:
        d = md["displacements"]
        entry = {
            "name": md["name"],
            "mean_disp": round(float(d.mean()), 1),
            "std_disp": round(float(d.std()), 1),
        }
        if md["grad_norms"] is not None:
            entry["mean_grad"] = round(float(md["grad_norms"].mean()), 1)
        per_model.append(entry)

    # Shared shart examples: tokens with highest mean displacement across models
    mean_disps = np.mean([md["displacements"] for md in model_data], axis=0)
    top_shared = np.argsort(-mean_disps)[:20]
    shared_sharts = []
    for rank, si in enumerate(top_shared):
        entry = {
            "rank": rank + 1,
            "token": shared_texts[si],
        }
        for k, md in enumerate(model_data):
            entry[f"disp_{md['name']}"] = round(float(md["displacements"][si]), 1)
        shared_sharts.append(entry)

    return {
        "n_shared": n_shared,
        "n_models": len(mris),
        "models": per_model,
        "comparisons": comparisons,
        "shared_sharts": shared_sharts,
    }


def shart_anatomy(mri_path: str, *, n_sample: int | None = None,
                   top_n: int = 100) -> dict:
    """What makes a shart a shart?

    For each token, measures:
      - Final displacement (exit state norm)
      - Embedding gradient norm (sensitivity to input perturbation)
      - Crystal neuron identification (dominant neuron at peak amplification layer)
      - Crystal neuron activation magnitude vs displacement
      - Gate activity in the frozen zone (opposing or idle?)
      - Active neuron count (bandwidth)

    Returns ranked tokens with full decomposition.
    """
    from .mri import load_mri

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    hidden = meta['model']['hidden_size']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    token_ids = mri['token_ids']
    token_texts = mri.get('token_texts', np.array([''] * len(token_ids)))
    scripts = mri.get('scripts', np.array(['?'] * len(token_ids)))
    n_tok = len(token_ids)

    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    # Final displacement
    exit_key = f"exit_L{n_layers - 1}"
    if exit_key not in mri:
        return {"error": f"Missing {exit_key}"}
    exit_last = mri[exit_key][idx].astype(np.float32)
    displacements = np.linalg.norm(exit_last, axis=1)

    # Embedding gradient
    from pathlib import Path
    eg_path = Path(mri_path) / "embedding_grad.npy"
    if eg_path.exists():
        eg = np.load(eg_path)[idx].astype(np.float32)
        grad_norms = np.linalg.norm(eg, axis=1)
        disp_grad_corr = float(np.corrcoef(displacements, grad_norms)[0, 1])
    else:
        grad_norms = None
        disp_grad_corr = None

    # Find peak amplification layer (crystal layer)
    # Use a small subsample to avoid reading all layers at full size
    amp_sample = idx[:min(500, len(idx))]
    max_amp_layer = 0
    max_amp = 0
    prev_exit_s = None
    prev_norms_mean_s = 0
    import sys
    for i in range(n_layers):
        ek = f"exit_L{i}"
        if ek not in mri:
            continue
        print(f"  scanning L{i}/{n_layers}...", end="\r", file=sys.stderr)
        curr_s = mri[ek][amp_sample].astype(np.float32)
        curr_norms_s = np.linalg.norm(curr_s, axis=1)
        if prev_exit_s is not None:
            delta_norms_s = np.linalg.norm(curr_s - prev_exit_s, axis=1)
            amp = delta_norms_s.mean() / max(prev_norms_mean_s, 1e-8)
            if amp > max_amp:
                max_amp = amp
                max_amp_layer = i
        prev_exit_s = curr_s
        prev_norms_mean_s = curr_norms_s.mean()

    # Crystal neuron at peak layer
    crystal_layer = max_amp_layer
    gate_key = f"gate_L{crystal_layer}"
    up_key = f"up_L{crystal_layer}"
    crystal_neuron = -1
    crystal_corr = 0.0
    crystal_energy_frac = 0.0
    if gate_key in mri and up_key in mri:
        g = mri[gate_key][idx].astype(np.float32)
        u = mri[up_key][idx].astype(np.float32)
        neuron_act = g * u
        mean_abs = np.abs(neuron_act).mean(axis=0)
        crystal_neuron = int(np.argmax(mean_abs))
        crystal_act = np.abs(neuron_act[:, crystal_neuron])
        crystal_corr = float(np.corrcoef(crystal_act, displacements)[0, 1])
        total_energy = np.abs(neuron_act).sum(axis=1)
        crystal_energy_frac = float((crystal_act / (total_energy + 1e-8)).mean())
    elif gate_key in mri:
        g = mri[gate_key][idx].astype(np.float32)
        mean_abs = np.abs(g).mean(axis=0)
        crystal_neuron = int(np.argmax(mean_abs))

    # Frozen zone: gate cosine between early and late frozen layers
    frozen_start = crystal_layer + 2
    frozen_end = n_layers - 3
    frozen_gate_cosine = None
    if frozen_end > frozen_start and f"gate_L{frozen_start}" in mri and f"gate_L{frozen_end}" in mri:
        ga = mri[f"gate_L{frozen_start}"][idx].astype(np.float32)
        gb = mri[f"gate_L{frozen_end}"][idx].astype(np.float32)
        dot = (ga * gb).sum(axis=1)
        na = np.linalg.norm(ga, axis=1)
        nb = np.linalg.norm(gb, axis=1)
        frozen_gate_cosine = float((dot / (na * nb + 1e-8)).mean())

    # Active neuron count at crystal layer
    if gate_key in mri and up_key in mri:
        active_per_token = (np.abs(neuron_act) > 1.0).sum(axis=1).astype(float)
        active_disp_corr = float(np.corrcoef(active_per_token, displacements)[0, 1])
    else:
        active_per_token = None
        active_disp_corr = None

    # Rank tokens
    order = np.argsort(-displacements)
    top_tokens = []
    for rank, oi in enumerate(order[:top_n]):
        real_idx = idx[oi]
        entry = {
            "rank": rank + 1,
            "token_id": int(token_ids[real_idx]),
            "token": str(token_texts[real_idx]),
            "script": str(scripts[real_idx]),
            "displacement": round(float(displacements[oi]), 1),
        }
        if grad_norms is not None:
            entry["grad_norm"] = round(float(grad_norms[oi]), 2)
        if gate_key in mri and up_key in mri:
            entry["crystal_activation"] = round(float(np.abs(neuron_act[oi, crystal_neuron])), 1)
            entry["active_neurons"] = int(active_per_token[oi])
        top_tokens.append(entry)

    bottom_tokens = []
    for rank, oi in enumerate(order[-top_n:]):
        real_idx = idx[oi]
        entry = {
            "rank": n_sample - top_n + rank + 1,
            "token_id": int(token_ids[real_idx]),
            "token": str(token_texts[real_idx]),
            "script": str(scripts[real_idx]),
            "displacement": round(float(displacements[oi]), 1),
        }
        if grad_norms is not None:
            entry["grad_norm"] = round(float(grad_norms[oi]), 2)
        bottom_tokens.append(entry)

    return {
        "model": model_name,
        "mode": mode,
        "n_tokens": n_sample,
        "displacement": {
            "min": round(float(displacements.min()), 1),
            "max": round(float(displacements.max()), 1),
            "median": round(float(np.median(displacements)), 1),
            "ratio_max_median": round(float(displacements.max() / np.median(displacements)), 2),
        },
        "gradient": {
            "corr_with_displacement": disp_grad_corr,
            "top_sharts_mean": round(float(grad_norms[order[:top_n]].mean()), 1) if grad_norms is not None else None,
            "bottom_mean": round(float(grad_norms[order[-top_n:]].mean()), 1) if grad_norms is not None else None,
        },
        "crystal": {
            "layer": crystal_layer,
            "amplification": round(max_amp, 1),
            "neuron": crystal_neuron,
            "corr_with_displacement": round(crystal_corr, 4),
            "energy_fraction": round(crystal_energy_frac, 3),
        },
        "frozen_zone": {
            "gate_cosine": round(frozen_gate_cosine, 4) if frozen_gate_cosine is not None else None,
            "interpretation": "opposing" if frozen_gate_cosine is not None and frozen_gate_cosine < 0 else "aligned" if frozen_gate_cosine is not None else "unknown",
        },
        "bandwidth": {
            "active_neuron_disp_corr": round(active_disp_corr, 4) if active_disp_corr is not None else None,
            "interpretation": "magnitude not count" if active_disp_corr is not None and active_disp_corr < 0.3 else "more neurons" if active_disp_corr is not None else "unknown",
        },
        "top_sharts": top_tokens,
        "bottom_tokens": bottom_tokens,
    }


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


def logit_lens(mri_path: str, *, top_k: int = 5,
               layers: list[int] | None = None,
               n_sample: int | None = None, _mri=None) -> dict:
    """Logit lens: what would the model predict at each layer?

    Applies final_norm + lmhead to each layer's exit state.
    Returns top-K token predictions per layer per token.
    Requires lmhead_raw.npy and norms.npz in the MRI directory.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    hidden = meta['model']['hidden_size']
    model_name = meta['model']['name']

    if 'lmhead_raw' not in mri:
        return {"error": "MRI missing lmhead_raw.npy — run mri-backfill"}
    lmhead = mri['lmhead_raw'].astype(np.float32)  # [vocab, hidden]

    # Final norm weights (RMSNorm: weight * x / rms(x))
    final_norm_w = mri.get('norm_final')
    if final_norm_w is not None:
        final_norm_w = np.array(final_norm_w).astype(np.float32)

    token_ids = mri['token_ids']
    token_texts = mri.get('token_texts', np.array([''] * len(token_ids)))
    n_tokens = len(token_ids)

    if layers is None:
        layers = list(range(n_layers))

    # Optional subsampling
    if n_sample and n_sample < n_tokens:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tokens, n_sample, replace=False))
    else:
        idx = list(range(n_tokens))
        n_sample = n_tokens

    import sys
    results = []
    for li, layer in enumerate(layers):
        print(f"  logit_lens {li+1}/{len(layers)}", end="\r", file=sys.stderr)
        exit_key = f"exit_L{layer}"
        if exit_key not in mri:
            continue

        states = mri[exit_key][idx].astype(np.float32)  # [n_sample, hidden]

        # Apply RMSNorm
        if final_norm_w is not None:
            rms = np.sqrt(np.mean(states ** 2, axis=1, keepdims=True) + 1e-6)
            states = states / rms * final_norm_w

        # Logits: [n_sample, vocab]
        logits = states @ lmhead.T

        # Top-K per token
        top_indices = np.argpartition(-logits, top_k, axis=1)[:, :top_k]
        layer_preds = []
        for t in range(len(idx)):
            top_idx = top_indices[t]
            top_logit = logits[t, top_idx]
            order = np.argsort(-top_logit)
            top_idx = top_idx[order]
            top_logit = top_logit[order]
            layer_preds.append({
                "token_id": int(token_ids[idx[t]]),
                "top_ids": top_idx.tolist(),
                "top_logits": [round(float(v), 2) for v in top_logit],
            })

        results.append({
            "layer": layer,
            "predictions": layer_preds,
        })

    print(" " * 40, end="\r", file=sys.stderr)

    return {
        "model": model_name,
        "n_tokens": n_sample,
        "n_layers": len(results),
        "top_k": top_k,
        "layers": results,
    }


def layer_deltas(mri_path: str, *, n_sample: int | None = None, _mri=None) -> dict:
    """Layer deltas: what each layer actually computes.

    Delta at layer i = exit_L[i] - exit_L[i-1].
    Delta at layer 0 = exit_L[0] - embedding (if embedding exists).

    Returns per-layer statistics: mean/max delta norm, dominant direction.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    n_tokens = len(mri['token_ids'])

    if n_sample and n_sample < n_tokens:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tokens, n_sample, replace=False))
    else:
        idx = list(range(n_tokens))
        n_sample = n_tokens

    scripts = mri.get('scripts', np.array(['?'] * n_tokens))
    sampled_scripts = scripts[idx]

    import sys
    results = []
    prev = None
    for i in range(n_layers):
        print(f"  layer_deltas L{i}/{n_layers}", end="\r", file=sys.stderr)
        exit_key = f"exit_L{i}"
        if exit_key not in mri:
            continue
        curr = mri[exit_key][idx].astype(np.float32)

        if prev is not None:
            delta = curr - prev
        else:
            delta = curr  # layer 0: delta from zero (or embedding)

        norms = np.linalg.norm(delta, axis=1)
        results.append({
            "layer": i,
            "mean_delta_norm": round(float(norms.mean()), 2),
            "max_delta_norm": round(float(norms.max()), 2),
            "std_delta_norm": round(float(norms.std()), 2),
            "amplification": round(float(norms.mean() / max(results[-1]["mean_delta_norm"], 1e-8)), 2)
                if results else 1.0,
        })
        prev = curr
    print(" " * 40, end="\r", file=sys.stderr)

    return {
        "model": model_name,
        "mode": mode,
        "n_tokens": n_sample,
        "n_layers": len(results),
        "layers": results,
    }


# ---------------------------------------------------------------------------
# Causal bank analysis tools
# ---------------------------------------------------------------------------

def _cb_sample(mri, n_sample: int | None):
    """Sample indices and scripts from an MRI (shared helper)."""
    n_tok = len(mri['token_ids'])
    scripts = mri.get('scripts', np.array(['?'] * n_tok))
    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok
    return idx, n_sample, scripts


def causal_bank_manifold(mri_path: str, *, n_sample: int | None = None, _mri=None) -> dict:
    """Manifold structure of causal bank substrate states.

    PCA decomposition, effective dimensionality, band loadings,
    readout alignment, and weight analysis from a single MRI.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}

    model_info = meta['model']
    n_modes = model_info['n_modes']
    n_bands = model_info.get('n_bands', 1)
    n_experts = model_info.get('n_experts', 1)
    embed_dim = model_info.get('embed_dim', 0)

    idx, n_sample, scripts = _cb_sample(mri, n_sample)
    sampled_scripts = scripts[idx]

    sub = mri['substrate_states'][idx].astype(np.float32)
    half_lives = mri.get('half_lives')
    if half_lives is not None:
        half_lives = np.asarray(half_lives, dtype=np.float32)

    # --- PCA ---
    sub_c = sub - sub.mean(axis=0)
    _, S, Vt = np.linalg.svd(sub_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var_exp)
    eff_dim = float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum()))

    pca = {
        "effective_dim": round(eff_dim, 1),
        "pcs_for_50": int(np.searchsorted(cum, 0.5)) + 1,
        "pcs_for_80": int(np.searchsorted(cum, 0.8)) + 1,
        "pcs_for_95": int(np.searchsorted(cum, 0.95)) + 1,
        "top_10_variance": [round(float(v) * 100, 2) for v in var_exp[:10]],
    }

    # --- Band loadings of top PCs ---
    band_loadings = []
    if half_lives is not None and n_bands > 1:
        band_size = n_modes // n_bands
        for pc_i in range(min(10, Vt.shape[0])):
            pc_vec = Vt[pc_i]
            bnorms = [float(np.linalg.norm(pc_vec[b * band_size:(b + 1) * band_size]))
                      for b in range(n_bands)]
            total = sum(bnorms) + 1e-10
            band_loadings.append({
                "pc": pc_i + 1,
                "variance_pct": round(float(var_exp[pc_i]) * 100, 2),
                "band_pcts": [round(bn / total * 100, 1) for bn in bnorms],
            })

    # --- Half-life bands ---
    bands_info = []
    if half_lives is not None:
        band_size = n_modes // max(n_bands, 1)
        for b in range(max(n_bands, 4)):
            sl = slice(b * band_size, min((b + 1) * band_size, n_modes))
            if sl.start >= n_modes:
                break
            hl_b = half_lives[sl]
            sub_b = sub[:, sl]
            bands_info.append({
                "band": b,
                "hl_min": round(float(hl_b.min()), 1),
                "hl_max": round(float(hl_b.max()), 1),
                "n_modes": int(sl.stop - sl.start),
                "substrate_l2": round(float(np.linalg.norm(sub_b, axis=1).mean()), 4),
            })

    # --- Readout alignment ---
    readout = {}
    # Find expert_in weights to measure what the readout sees
    band_size_r = n_modes // max(n_bands, 1)
    mode_attention = np.zeros(n_modes)
    expert_count = 0
    for b in range(n_bands):
        for e in range(n_experts):
            wkey = f"weight__band_readouts_{b}_experts_in_{e}_weight"
            if wkey not in mri:
                # Try single-readout path
                wkey = f"weight_linear_readout_experts_in_{e}_weight"
            if wkey in mri:
                w = mri[wkey].astype(np.float32)
                # Per-band substrate columns
                col_start = 0
                col_end = min(band_size_r, w.shape[1])
                if n_bands > 1:
                    col_end = min(band_size_r, w.shape[1] - embed_dim)
                elif w.shape[1] > n_modes:
                    col_end = n_modes
                mode_cols = w[:, :col_end]
                contrib = np.linalg.norm(mode_cols, axis=0)
                if n_bands > 1:
                    mode_attention[b * band_size_r:b * band_size_r + len(contrib)] += contrib
                else:
                    mode_attention[:len(contrib)] += contrib
                expert_count += 1

    if expert_count > 0:
        mode_attention /= expert_count
        # Per-band readout weight
        readout_bands = []
        for b in range(max(n_bands, 4)):
            sl = slice(b * band_size_r, min((b + 1) * band_size_r, n_modes))
            if sl.start >= n_modes:
                break
            ma = mode_attention[sl]
            readout_bands.append({
                "band": b,
                "mean_weight": round(float(ma.mean()), 4),
                "pct_of_total": round(float(ma.sum()) / (mode_attention.sum() + 1e-10) * 100, 1),
            })

        # Slow vs fast ratio
        if half_lives is not None:
            slow_mask = half_lives > (half_lives.max() * 0.4)
            fast_mask = half_lives < (half_lives.min() * 5)
            slow_w = float(mode_attention[slow_mask].mean()) if slow_mask.any() else 0
            fast_w = float(mode_attention[fast_mask].mean()) if fast_mask.any() else 1
            readout["slow_fast_ratio"] = round(slow_w / (fast_w + 1e-10), 4)

        readout["bands"] = readout_bands
        readout["dead_modes"] = int((mode_attention < mode_attention.mean() * 0.01).sum())

    # --- Router differentiation ---
    routing_info = {}
    for b in range(n_bands):
        rw_key = f"weight__band_readouts_{b}_router_weight"
        if rw_key not in mri:
            rw_key = "weight_linear_readout_router_weight"
        if rw_key in mri:
            rw = mri[rw_key].astype(np.float32)
            ne = rw.shape[0]
            norms = np.linalg.norm(rw, axis=1, keepdims=True)
            rw_n = rw / (norms + 1e-10)
            cos = rw_n @ rw_n.T
            off = cos[np.triu_indices(ne, k=1)]
            routing_info[f"band_{b}" if n_bands > 1 else "global"] = {
                "n_experts": int(ne),
                "router_cos_mean": round(float(off.mean()), 4),
                "router_cos_min": round(float(off.min()), 4),
                "router_cos_max": round(float(off.max()), 4),
            }

    # --- Gate analysis (for gated_delta models) ---
    gate_info = {}
    for gate_name in ["write", "retain", "erase"]:
        bias_key = f"weight__gated_delta_{gate_name}_gate_bias"
        if bias_key in mri:
            bias = mri[bias_key].astype(np.float32)
            sigmoid_bias = 1.0 / (1.0 + np.exp(-bias))
            gate_info[gate_name] = {
                "default_opening": round(float(sigmoid_bias.mean()), 4),
                "opening_range": [round(float(sigmoid_bias.min()), 4),
                                  round(float(sigmoid_bias.max()), 4)],
            }

    # --- SSM analysis (for scan models) ---
    ssm_info = {}
    if "weight__ssm_A" in mri:
        A = mri["weight__ssm_A"].astype(np.float32)
        retention = np.exp(np.clip(A, -20, 0))
        scan_hl = np.log(0.5) / np.log(np.clip(retention, 1e-6, 1 - 1e-6))
        ssm_info["A_shape"] = list(A.shape)
        ssm_info["scan_hl_range"] = [round(float(scan_hl.min()), 1),
                                     round(float(scan_hl.max()), 1)]
        if A.ndim == 2:
            ssm_info["per_head"] = [
                {"head": int(h),
                 "hl_range": [round(float(scan_hl[h].min()), 1),
                              round(float(scan_hl[h].max()), 1)]}
                for h in range(A.shape[0])
            ]

    return {
        "model": model_info.get('name', '?'),
        "n_modes": n_modes,
        "n_bands": n_bands,
        "n_experts": n_experts,
        "embed_dim": embed_dim,
        "n_tokens": n_sample,
        "pca": pca,
        "band_loadings": band_loadings,
        "bands": bands_info,
        "readout": readout,
        "routing": routing_info,
        "gate": gate_info,
        "ssm": ssm_info,
    }


def causal_bank_compare(mri_path_a: str, mri_path_b: str, *,
                        n_sample: int | None = None) -> dict:
    """Compare two causal bank MRIs: CKA, displacement correlation, routing."""
    from .mri import load_mri

    mri_a = load_mri(mri_path_a)
    mri_b = load_mri(mri_path_b)

    for label, mri in [("a", mri_a), ("b", mri_b)]:
        if mri['metadata'].get('architecture') != 'causal_bank':
            return {"error": f"MRI {label} is not a causal bank MRI"}

    n_tok = min(len(mri_a['token_ids']), len(mri_b['token_ids']))
    if n_sample and n_sample < n_tok:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))
    else:
        idx = list(range(n_tok))
        n_sample = n_tok

    sub_a = mri_a['substrate_states'][idx].astype(np.float32)
    sub_b = mri_b['substrate_states'][idx].astype(np.float32)

    # Displacement correlation
    disp_a = np.linalg.norm(sub_a, axis=1)
    disp_b = np.linalg.norm(sub_b, axis=1)
    disp_corr = float(np.corrcoef(disp_a, disp_b)[0, 1])

    # CKA
    def _linear_cka(X, Y):
        n = X.shape[0]
        X, Y = X - X.mean(0), Y - Y.mean(0)
        XX, YY = X @ X.T, Y @ Y.T
        H = np.eye(n) - np.ones((n, n)) / n
        hxy = np.trace(XX @ H @ YY @ H) / (n - 1) ** 2
        hxx = np.trace(XX @ H @ XX @ H) / (n - 1) ** 2
        hyy = np.trace(YY @ H @ YY @ H) / (n - 1) ** 2
        return float(hxy / (np.sqrt(hxx * hyy) + 1e-10))

    # Subsample for CKA (expensive)
    cka_n = min(500, n_sample)
    cka_idx = list(range(cka_n))
    cka = _linear_cka(sub_a[cka_idx], sub_b[cka_idx])

    # Router weight comparison (if both have routers)
    router_cos = {}
    meta_a = mri_a['metadata']['model']
    meta_b = mri_b['metadata']['model']
    n_bands_a = meta_a.get('n_bands', 1)
    n_bands_b = meta_b.get('n_bands', 1)
    if n_bands_a == n_bands_b:
        for b in range(n_bands_a):
            rw_key = f"weight__band_readouts_{b}_router_weight"
            if rw_key not in mri_a or rw_key not in mri_b:
                rw_key = "weight_linear_readout_router_weight"
            if rw_key in mri_a and rw_key in mri_b:
                rw_a = mri_a[rw_key].astype(np.float32)
                rw_b = mri_b[rw_key].astype(np.float32)
                if rw_a.shape == rw_b.shape:
                    cos = float(np.sum(rw_a * rw_b) / (
                        np.linalg.norm(rw_a) * np.linalg.norm(rw_b) + 1e-10))
                    router_cos[f"band_{b}" if n_bands_a > 1 else "global"] = round(cos, 4)

    return {
        "a": mri_path_a, "b": mri_path_b,
        "n_tokens": n_sample,
        "displacement_correlation": round(disp_corr, 4),
        "cka": round(cka, 4),
        "a_info": {"n_modes": meta_a['n_modes'], "n_bands": n_bands_a},
        "b_info": {"n_modes": meta_b['n_modes'], "n_bands": n_bands_b},
        "router_cosine": router_cos,
    }


def causal_bank_health(mri_path: str, *, _mri=None) -> dict:
    """Validate causal bank MRI: shapes, NaN, architecture consistency."""
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}

    model = meta['model']
    n_modes = model['n_modes']
    n_bands = model.get('n_bands', 1)
    n_experts = model.get('n_experts', 1)
    n_tokens = meta['capture']['n_tokens']

    checks = []
    ok = True

    # Substrate shape
    if 'substrate_states' in mri:
        sub = mri['substrate_states']
        expected = (n_tokens, n_modes)
        if sub.shape != expected:
            checks.append(f"FAIL substrate shape {sub.shape} != {expected}")
            ok = False
        elif np.isnan(sub).any():
            checks.append(f"FAIL substrate has NaN ({np.isnan(sub).sum()} values)")
            ok = False
        else:
            checks.append(f"OK substrate {sub.shape} {sub.dtype}")
    else:
        checks.append("FAIL substrate missing")
        ok = False

    # Half-lives
    if 'half_lives' in mri:
        hl = np.asarray(mri['half_lives'])
        if hl.shape != (n_modes,):
            checks.append(f"FAIL half_lives shape {hl.shape} != ({n_modes},)")
            ok = False
        elif (hl <= 0).any():
            checks.append(f"FAIL half_lives has non-positive values")
            ok = False
        else:
            checks.append(f"OK half_lives [{hl.min():.1f}, {hl.max():.1f}]")
    else:
        checks.append("WARN half_lives missing")

    # Routing
    if 'routing' in mri and n_experts > 1:
        rt = mri['routing']
        if rt.shape[0] != n_tokens:
            checks.append(f"FAIL routing shape {rt.shape}")
            ok = False
        else:
            checks.append(f"OK routing {rt.shape}")

    # Band logits
    if 'band_logits' in mri and n_bands > 1:
        bl = mri['band_logits']
        expected_bl = (n_tokens, n_bands, model.get('vocab_size', 1024))
        if bl.shape != expected_bl:
            checks.append(f"FAIL band_logits shape {bl.shape} != {expected_bl}")
            ok = False
        else:
            checks.append(f"OK band_logits {bl.shape}")

    # Embedding
    if 'embedding' in mri:
        emb = mri['embedding']
        checks.append(f"OK embedding {emb.shape}")
    else:
        checks.append("WARN embedding missing")

    # Weight count
    weight_keys = [k for k in mri.keys() if k.startswith("weight_")]
    checks.append(f"OK {len(weight_keys)} weight arrays")

    return {
        "path": str(mri_path),
        "architecture": "causal_bank",
        "n_modes": n_modes,
        "n_bands": n_bands,
        "n_experts": n_experts,
        "n_tokens": n_tokens,
        "ok": ok,
        "checks": checks,
    }


def causal_bank_loss(mri_path: str, *, _mri=None) -> dict:
    """Per-position loss decomposition from sequence-mode causal bank MRI.

    Breaks down loss by position range, band, and autocorrelation.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI (not impulse)"}

    loss = mri['loss']  # [n_seqs, seq_len]
    n_seqs, seq_len = loss.shape

    valid = ~np.isnan(loss)
    flat_loss = loss[valid]
    overall_bpb = float(np.mean(flat_loss))

    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk = loss[:, lo:hi]
        chunk_valid = chunk[~np.isnan(chunk)]
        if len(chunk_valid) == 0:
            continue
        by_pos.append({
            "range": f"{lo}-{hi}",
            "mean_bpb": round(float(np.mean(chunk_valid)), 4),
            "std_bpb": round(float(np.std(chunk_valid)), 4),
            "n_positions": hi - lo,
        })

    by_band = []
    if 'band_loss' in mri:
        band_loss = mri['band_loss']  # [n_seqs, seq_len, n_bands]
        n_bands = band_loss.shape[2]
        for b in range(n_bands):
            bl = band_loss[:, 1:, b]
            bl_valid = bl[~np.isnan(bl)]
            if len(bl_valid) > 0:
                by_band.append({
                    "band": b,
                    "mean_bpb": round(float(np.mean(bl_valid)), 4),
                })

    autocorr = []
    mean_loss = loss[:, 1:]
    for lag in [1, 2, 4, 8, 16, 32, 64, 128]:
        if lag >= seq_len - 1:
            break
        a = mean_loss[:, :-lag].flatten()
        b_arr = mean_loss[:, lag:].flatten()
        mask = ~(np.isnan(a) | np.isnan(b_arr))
        if mask.sum() < 10:
            continue
        r = float(np.corrcoef(a[mask], b_arr[mask])[0, 1])
        autocorr.append({"lag": lag, "r": round(r, 4)})

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "overall_bpb": round(overall_bpb, 4),
        "by_position": by_pos,
        "by_band": by_band,
        "autocorrelation": autocorr,
    }


def causal_bank_routing(mri_path: str, *, _mri=None) -> dict:
    """Sequence-level expert routing from sequence-mode causal bank MRI."""
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}
    if 'routing' not in mri or mri.get('routing') is None:
        return {"error": "MRI has no routing data (single-expert model?)"}

    routing = mri['routing'].astype(np.float32)  # [n_seqs, seq_len, n_experts]
    n_seqs, seq_len, n_experts = routing.shape

    winners = np.argmax(routing, axis=-1)
    overall_dist = []
    for e in range(n_experts):
        pct = float((winners == e).mean()) * 100
        overall_dist.append({"expert": e, "pct": round(pct, 2)})

    switches = (winners[:, 1:] != winners[:, :-1])
    switch_rate = float(switches.mean()) * 100

    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk_winners = winners[:, lo:hi]
        dist = [round(float((chunk_winners == e).mean()) * 100, 1) for e in range(n_experts)]
        sr = 0.0
        if hi - lo > 1:
            chunk_switches = (chunk_winners[:, 1:] != chunk_winners[:, :-1])
            sr = float(chunk_switches.mean()) * 100
        by_pos.append({
            "range": f"{lo}-{hi}",
            "distribution": dist,
            "switch_rate": round(sr, 2),
        })

    sorted_r = np.sort(routing, axis=-1)
    margin = float((sorted_r[:, :, -1] - sorted_r[:, :, -2]).mean())

    flat_routing = routing.reshape(-1, n_experts)
    flat_safe = flat_routing / (flat_routing.sum(axis=-1, keepdims=True) + 1e-10)
    entropy = float(-(flat_safe * np.log(flat_safe + 1e-10)).sum(axis=-1).mean())

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_experts": n_experts,
        "overall_distribution": overall_dist,
        "switch_rate": round(switch_rate, 2),
        "routing_margin": round(margin, 4),
        "routing_entropy": round(entropy, 4),
        "by_position": by_pos,
    }


def causal_bank_temporal(mri_path: str, *, _mri=None) -> dict:
    """Temporal attention forensics from sequence-mode causal bank MRI."""
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}
    if 'temporal_output' not in mri or mri.get('temporal_output') is None:
        return {"error": "MRI has no temporal attention data"}

    temporal_output = mri['temporal_output'].astype(np.float32)
    temporal_weights = mri.get('temporal_weights')
    substrate = mri['substrate_states'].astype(np.float32)
    embedding = mri['embedding'].astype(np.float32) if 'embedding' in mri else None

    n_seqs, seq_len, n_modes = temporal_output.shape
    snapshot_interval = meta['capture'].get('snapshot_interval', 64)

    ta_l2 = np.linalg.norm(temporal_output, axis=-1)
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    l2_by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk = ta_l2[:, lo:hi]
        l2_by_pos.append({
            "range": f"{lo}-{hi}",
            "mean_l2": round(float(chunk.mean()), 4),
            "max_l2": round(float(chunk.max()), 4),
        })

    sub_disp = np.linalg.norm(substrate, axis=-1).flatten()
    ta_disp = ta_l2.flatten()
    corr_chain = {}
    if embedding is not None:
        embed_norm = np.linalg.norm(embedding, axis=-1).flatten()
        corr_chain["embed_substrate"] = round(float(np.corrcoef(embed_norm, sub_disp)[0, 1]), 4)
        corr_chain["embed_temporal"] = round(float(np.corrcoef(embed_norm, ta_disp)[0, 1]), 4)
    corr_chain["substrate_temporal"] = round(float(np.corrcoef(sub_disp, ta_disp)[0, 1]), 4)

    snapshot_profile = {}
    if temporal_weights is not None:
        tw = temporal_weights.astype(np.float32)
        n_snapshots = tw.shape[2]
        mean_per_snap = tw.mean(axis=(0, 1))
        snapshot_profile = {
            "n_snapshots": int(n_snapshots),
            "snapshot_interval": snapshot_interval,
            "mean_weight_per_snapshot": [round(float(w), 4) for w in mean_per_snap],
            "peak_snapshot": int(np.argmax(mean_per_snap)),
        }

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "output_l2_by_position": l2_by_pos,
        "correlation_chain": corr_chain,
        "snapshot_profile": snapshot_profile,
    }


def causal_bank_modes(mri_path: str, *, _mri=None) -> dict:
    """Mode utilization from sequence-mode causal bank MRI."""
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)
    half_lives = mri.get('half_lives')
    n_seqs, seq_len, n_modes = substrate.shape

    if half_lives is not None:
        half_lives = np.asarray(half_lives, dtype=np.float32)
        quartile_edges = np.percentile(half_lives, [0, 25, 50, 75, 100])
    else:
        quartile_edges = None

    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_quartile = []
    if quartile_edges is not None:
        for q in range(4):
            lo_hl, hi_hl = quartile_edges[q], quartile_edges[q + 1]
            mask = (half_lives >= lo_hl) & (half_lives <= hi_hl) if q == 3 else \
                   (half_lives >= lo_hl) & (half_lives < hi_hl)
            if not mask.any():
                continue
            sub_q = substrate[:, :, mask]
            row = {
                "quartile": q,
                "hl_range": f"{lo_hl:.1f}-{hi_hl:.1f}",
                "n_modes": int(mask.sum()),
                "by_position": [],
            }
            for lo, hi in ranges:
                hi = min(hi, seq_len)
                if lo >= seq_len:
                    break
                chunk = np.abs(sub_q[:, lo:hi, :])
                row["by_position"].append({
                    "range": f"{lo}-{hi}",
                    "mean_abs": round(float(chunk.mean()), 4),
                })
            early = np.abs(sub_q[:, :min(4, seq_len), :]).mean()
            late = np.abs(sub_q[:, max(0, seq_len - 64):, :]).mean()
            row["ramp_ratio"] = round(float(late / (early + 1e-10)), 2)
            by_quartile.append(row)

    max_activation = np.abs(substrate).max(axis=(0, 1))
    mean_act = max_activation.mean()
    dead_modes = int((max_activation < mean_act * 0.01).sum())

    pos_std = np.abs(substrate).mean(axis=0).std(axis=0)
    top_varying = np.argsort(pos_std)[-5:][::-1]
    most_varying = [{"mode": int(m), "std": round(float(pos_std[m]), 4),
                     "hl": round(float(half_lives[m]), 1) if half_lives is not None else None}
                    for m in top_varying]

    l2_by_pos = np.linalg.norm(substrate, axis=-1).mean(axis=0)
    growth_curve = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        growth_curve.append({
            "range": f"{lo}-{hi}",
            "mean_l2": round(float(l2_by_pos[lo:hi].mean()), 4),
        })

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "by_quartile": by_quartile,
        "dead_modes": dead_modes,
        "most_varying": most_varying,
        "growth_curve": growth_curve,
    }


def causal_bank_decompose(mri_path: str, *, n_sample: int | None = None,
                          _mri=None) -> dict:
    """Manifold decomposition from sequence-mode causal bank MRI.

    PCA on sequence substrate. Identifies which PCs encode position (clock),
    which predict loss (content), and which are ghosts (neither).
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)
    loss = mri['loss']
    n_seqs, seq_len, n_modes = substrate.shape

    flat_sub = substrate.reshape(-1, n_modes)
    n_total = flat_sub.shape[0]
    flat_loss = loss.reshape(-1)
    flat_pos = np.tile(np.arange(seq_len), n_seqs)

    if n_sample and n_sample < n_total:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sample, replace=False)
        flat_sub = flat_sub[idx]
        flat_loss = flat_loss[idx]
        flat_pos = flat_pos[idx]

    sub_c = flat_sub - flat_sub.mean(axis=0)
    _, S, Vt = np.linalg.svd(sub_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var_exp)
    scores = sub_c @ Vt.T

    pca = {
        "effective_dim": round(float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum())), 1),
        "pcs_for_50": int(np.searchsorted(cum, 0.5)) + 1,
        "pcs_for_80": int(np.searchsorted(cum, 0.8)) + 1,
        "pcs_for_95": int(np.searchsorted(cum, 0.95)) + 1,
    }

    valid = ~np.isnan(flat_loss)
    n_pcs = min(20, scores.shape[1])
    pc_position_r = []
    pc_loss_r = []
    for i in range(n_pcs):
        r_pos = float(np.corrcoef(scores[:, i], flat_pos.astype(np.float32))[0, 1])
        if valid.sum() > 10:
            r_loss = float(np.corrcoef(scores[valid, i], flat_loss[valid])[0, 1])
        else:
            r_loss = 0.0
        pc_position_r.append(round(r_pos, 4))
        pc_loss_r.append(round(r_loss, 4))

    from numpy.linalg import lstsq
    # Add intercept column for proper R² (PCA scores are zero-mean,
    # but position and loss targets are not)
    ones = np.ones((scores.shape[0], 1), dtype=np.float32)

    X_pos = np.hstack([scores[:, :n_pcs], ones])
    y_pos = flat_pos.astype(np.float32)
    coef, _, _, _ = lstsq(X_pos, y_pos, rcond=None)
    pred_pos = X_pos @ coef
    ss_res = float(((y_pos - pred_pos) ** 2).sum())
    ss_tot = float(((y_pos - y_pos.mean()) ** 2).sum())
    position_r2 = round(1.0 - ss_res / (ss_tot + 1e-10), 4)

    if valid.sum() > n_pcs:
        X_loss = np.hstack([scores[valid, :n_pcs], ones[valid]])
        y_loss = flat_loss[valid]
        coef_l, _, _, _ = lstsq(X_loss, y_loss, rcond=None)
        pred_loss = X_loss @ coef_l
        ss_res_l = float(((y_loss - pred_loss) ** 2).sum())
        ss_tot_l = float(((y_loss - y_loss.mean()) ** 2).sum())
        content_r2 = round(1.0 - ss_res_l / (ss_tot_l + 1e-10), 4)
    else:
        content_r2 = 0.0

    threshold = 0.1
    ghost_var = position_var = content_var = 0.0
    for i in range(n_pcs):
        v = float(var_exp[i])
        is_pos = abs(pc_position_r[i]) > threshold
        is_content = abs(pc_loss_r[i]) > threshold
        if is_pos:
            position_var += v
        elif is_content:
            content_var += v
        else:
            ghost_var += v
    total_top = sum(float(var_exp[i]) for i in range(n_pcs))

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "pca": pca,
        "position_r2": position_r2,
        "content_r2": content_r2,
        "ghost_fraction": round(ghost_var / (total_top + 1e-10) * 100, 1),
        "position_fraction": round(position_var / (total_top + 1e-10) * 100, 1),
        "content_fraction": round(content_var / (total_top + 1e-10) * 100, 1),
        "pc_position_r": pc_position_r,
        "pc_loss_r": pc_loss_r,
        "top_variance_pct": [round(float(v) * 100, 2) for v in var_exp[:n_pcs]],
    }


def causal_bank_causality(backend, *, seq_len: int = 256, n_tests: int = 8,
                          seed: int = 42) -> dict:
    """Finite-difference causality test for causal bank models.

    Runs full sequence vs truncated sequence. If logits[t] differ between
    full and truncated-at-t, information leaked from future positions.
    """
    rng = np.random.RandomState(seed)
    vocab_size = backend.config.vocab_size
    seq = rng.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

    full_logits = backend.forward(seq)  # [1, seq_len, vocab]

    test_positions = sorted(rng.choice(range(4, seq_len - 1), n_tests, replace=False))
    violations = []

    for t in test_positions:
        truncated = seq[:, :t + 1]
        trunc_logits = backend.forward(truncated)

        full_t = full_logits[0, t]
        trunc_t = trunc_logits[0, t]
        max_diff = float(np.abs(full_t - trunc_t).max())

        if max_diff > 1e-4:
            violations.append({
                "position": int(t),
                "max_logit_diff": round(max_diff, 6),
                "mean_logit_diff": round(float(np.abs(full_t - trunc_t).mean()), 6),
            })

    causal = len(violations) == 0
    return {
        "causal": causal,
        "seq_len": seq_len,
        "n_tests": n_tests,
        "positions_tested": [int(t) for t in test_positions],
        "violations": violations,
        "verdict": "PASS: no future information leakage" if causal
                   else f"FAIL: {len(violations)} positions leaked future info",
    }


def causal_bank_reproduce(backend, *, seq_len: int = 256, seed: int = 42) -> dict:
    """Reproducibility test: two identical forward passes should give identical logits."""
    rng = np.random.RandomState(seed)
    vocab_size = backend.config.vocab_size
    seq = rng.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

    logits_a = backend.forward(seq)
    logits_b = backend.forward(seq)

    max_diff = float(np.abs(logits_a - logits_b).max())
    mean_diff = float(np.abs(logits_a - logits_b).mean())
    identical = max_diff == 0.0

    return {
        "identical": identical,
        "max_diff": round(max_diff, 10),
        "mean_diff": round(mean_diff, 10),
        "seq_len": seq_len,
        "verdict": "PASS: bitwise identical" if identical
                   else f"FAIL: max diff {max_diff:.2e}",
    }


def tokenizer_compare(tokenizer_paths: list[str], *,
                      sample_text: str | None = None) -> dict:
    """Compare multiple sentencepiece tokenizers.

    Vocab size, compression ratio, byte fallback, token length distribution,
    overlap between tokenizers, and parameter budget impact.
    """
    import sentencepiece as spm

    results = []
    all_vocabs = []

    for path in tokenizer_paths:
        sp = spm.SentencePieceProcessor()
        sp.Load(path)
        vocab_size = sp.GetPieceSize()

        lengths = []
        byte_tokens = 0
        for i in range(vocab_size):
            piece = sp.IdToPiece(i)
            if piece.startswith("<0x") and piece.endswith(">"):
                byte_tokens += 1
                lengths.append(1)
            else:
                raw = piece.replace("\u2581", " ")
                lengths.append(len(raw.encode("utf-8")))

        lengths_arr = np.array(lengths)
        vocab_set = set(sp.IdToPiece(i) for i in range(vocab_size))
        all_vocabs.append(vocab_set)

        bpt = None
        tpb = None
        byte_fallback_pct = round(byte_tokens / vocab_size * 100, 2)
        if sample_text:
            tokens = sp.Encode(sample_text)
            text_bytes = len(sample_text.encode("utf-8"))
            bpt = round(text_bytes / len(tokens), 4) if tokens else None
            tpb = round(len(tokens) / text_bytes, 4) if text_bytes else None
            n_byte_tok = sum(1 for t in tokens if sp.IdToPiece(t).startswith("<0x"))
            byte_fallback_pct = round(n_byte_tok / max(len(tokens), 1) * 100, 2)

        length_dist = {}
        for lb in [1, 2, 3, 4, 5, 8, 12, 16]:
            count = int((lengths_arr == lb).sum()) if lb < 16 else int((lengths_arr >= lb).sum())
            length_dist[f"{lb}B"] = count

        results.append({
            "path": path,
            "vocab_size": vocab_size,
            "bytes_per_token": bpt,
            "tokens_per_byte": tpb,
            "byte_fallback_pct": byte_fallback_pct,
            "byte_tokens": byte_tokens,
            "length_distribution": length_dist,
            "mean_token_bytes": round(float(lengths_arr.mean()), 2),
        })

    overlap = {}
    for i in range(len(all_vocabs)):
        for j in range(i + 1, len(all_vocabs)):
            common = len(all_vocabs[i] & all_vocabs[j])
            total = len(all_vocabs[i] | all_vocabs[j])
            overlap[f"{i}_vs_{j}"] = {
                "common": common,
                "jaccard": round(common / (total + 1e-10), 4),
            }

    return {
        "tokenizers": results,
        "overlap": overlap,
    }


def tokenizer_difficulty(mri_path: str, *, _mri=None) -> dict:
    """Per-token difficulty from embeddings. No model needed, reads MRI arrays.

    Embedding norm correlates with prediction difficulty. Identifies
    easy vs hard tokens and measures embedding space structure.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if 'embedding' not in mri:
        return {"error": "MRI has no embedding data"}

    embedding = mri['embedding'].astype(np.float32)
    if embedding.ndim == 3:
        embedding = embedding.reshape(-1, embedding.shape[-1])

    substrate = mri.get('substrate_states')
    if substrate is not None:
        substrate = substrate.astype(np.float32)
        if substrate.ndim == 3:
            substrate = substrate.reshape(-1, substrate.shape[-1])

    n_tokens, embed_dim = embedding.shape
    embed_norm = np.linalg.norm(embedding, axis=1)

    corr = {}
    if substrate is not None:
        sub_disp = np.linalg.norm(substrate, axis=1)
        corr["embed_substrate_r"] = round(float(np.corrcoef(embed_norm, sub_disp)[0, 1]), 4)

    emb_c = embedding - embedding.mean(axis=0)
    _, S, _ = np.linalg.svd(emb_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    eff_dim = float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum()))

    quartiles = np.percentile(embed_norm, [25, 50, 75])
    edges = [0, quartiles[0], quartiles[1], quartiles[2], embed_norm.max() + 1]
    labels = ["easy", "medium-easy", "medium-hard", "hard"]
    quartile_info = []
    for i in range(4):
        mask = (embed_norm >= edges[i]) & (embed_norm < edges[i + 1])
        quartile_info.append({
            "label": labels[i],
            "n_tokens": int(mask.sum()),
            "mean_norm": round(float(embed_norm[mask].mean()), 4) if mask.any() else 0,
        })

    n_near_dup = -1
    if n_tokens <= 5000:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        emb_normed = embedding / (norms + 1e-10)
        cos = emb_normed @ emb_normed.T
        np.fill_diagonal(cos, 0)
        n_near_dup = int((cos > 0.9).sum()) // 2

    return {
        "model": meta['model'].get('name', '?'),
        "n_tokens": n_tokens,
        "embed_dim": embed_dim,
        "effective_dim": round(eff_dim, 1),
        **corr,
        "difficulty_quartiles": quartile_info,
        "near_duplicates": n_near_dup,
        "embed_norm_range": [round(float(embed_norm.min()), 4),
                             round(float(embed_norm.max()), 4)],
    }


def causal_bank_rotation_probe(mri_path: str, *, n_sample: int | None = None,
                                hidden_dim: int = 256, _mri=None) -> dict:
    """Nonlinear and rotational information probes on sequence-mode MRI.

    Linear R² misses nonlinear structure. This tool compares:
    1. Linear probe vs random-features MLP probe (position + loss prediction)
    2. Angular decomposition: mode pair phases and phase velocities
    3. What the angles encode (position, content, difficulty)

    If MLP R² >> linear R², there's information the linear probe misses.
    If angular analysis shows position-dependent phase, the rotation encodes order.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)
    loss = mri['loss']
    embedding = mri['embedding'].astype(np.float32) if 'embedding' in mri else None
    n_seqs, seq_len, n_modes = substrate.shape

    flat_sub = substrate.reshape(-1, n_modes)
    flat_loss = loss.reshape(-1)
    flat_pos = np.tile(np.arange(seq_len, dtype=np.float32), n_seqs)

    n_total = flat_sub.shape[0]
    if n_sample and n_sample < n_total:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sample, replace=False)
    else:
        idx = np.arange(n_total)
    sub_s = flat_sub[idx]
    pos_s = flat_pos[idx]
    loss_s = flat_loss[idx]
    valid_s = ~np.isnan(loss_s)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(idx))
    split = int(len(perm) * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    from numpy.linalg import lstsq
    ones_train = np.ones((len(train_idx), 1), dtype=np.float32)
    ones_test = np.ones((len(test_idx), 1), dtype=np.float32)

    def _r2(y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return round(float(1.0 - ss_res / (ss_tot + 1e-10)), 6)

    def _linear_probe(X_tr, X_te, y_tr, y_te):
        Xt = np.hstack([X_tr, np.ones((len(X_tr), 1), dtype=np.float32)])
        coef, _, _, _ = lstsq(Xt, y_tr, rcond=None)
        pred = np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float32)]) @ coef
        return _r2(y_te, pred)

    def _mlp_probe(X_tr, X_te, y_tr, y_te, hdim=hidden_dim):
        rng_mlp = np.random.RandomState(123)
        W1 = rng_mlp.randn(X_tr.shape[1], hdim).astype(np.float32) * 0.1
        b1 = rng_mlp.randn(hdim).astype(np.float32) * 0.01
        H_tr = np.maximum(0, X_tr @ W1 + b1)
        Ht = np.hstack([H_tr, np.ones((len(H_tr), 1), dtype=np.float32)])
        coef, _, _, _ = lstsq(Ht, y_tr, rcond=None)
        H_te = np.maximum(0, X_te @ W1 + b1)
        pred = np.hstack([H_te, np.ones((len(H_te), 1), dtype=np.float32)]) @ coef
        return _r2(y_te, pred)

    # PCA
    sub_c = sub_s - sub_s.mean(axis=0)
    n_pcs = min(50, n_modes)
    _, S, Vt = np.linalg.svd(sub_c, full_matrices=False)
    scores = sub_c @ Vt[:n_pcs].T

    X_tr_pca = scores[train_idx]
    X_te_pca = scores[test_idx]

    # Position probes
    y_tr_pos = pos_s[train_idx]
    y_te_pos = pos_s[test_idx]
    pos_linear_pca = _linear_probe(X_tr_pca, X_te_pca, y_tr_pos, y_te_pos)
    pos_mlp_pca = _mlp_probe(X_tr_pca, X_te_pca, y_tr_pos, y_te_pos)
    # Also probe on raw substrate (first 256 modes for speed)
    raw_dim = min(256, n_modes)
    pos_mlp_raw = _mlp_probe(sub_s[train_idx, :raw_dim], sub_s[test_idx, :raw_dim],
                              y_tr_pos, y_te_pos)

    # Loss probes
    v_tr = valid_s[train_idx]
    v_te = valid_s[test_idx]
    if v_tr.sum() > 50 and v_te.sum() > 10:
        loss_linear_pca = _linear_probe(X_tr_pca[v_tr], X_te_pca[v_te],
                                        loss_s[train_idx][v_tr], loss_s[test_idx][v_te])
        loss_mlp_pca = _mlp_probe(X_tr_pca[v_tr], X_te_pca[v_te],
                                   loss_s[train_idx][v_tr], loss_s[test_idx][v_te])
    else:
        loss_linear_pca = loss_mlp_pca = 0.0

    # --- Angular decomposition ---
    n_pairs = n_modes // 2
    angles_per_pair = min(n_pairs, 128)
    even = substrate[:, :, :angles_per_pair * 2:2]
    odd = substrate[:, :, 1:angles_per_pair * 2:2]
    angles = np.arctan2(odd, even)  # [n_seqs, seq_len, n_pairs]

    # Phase velocity
    d_angle = np.diff(angles, axis=1)
    d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi

    # Per-pair position correlation
    positions = np.arange(seq_len, dtype=np.float32)
    n_check = min(angles_per_pair, 50)
    angle_pos_corr = np.zeros(n_check)
    for p in range(n_check):
        pair_angles = angles[:, :, p].mean(axis=0)
        angle_pos_corr[p] = float(np.corrcoef(pair_angles, positions)[0, 1])

    # Velocity by position range
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    velocity_by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len - 1)
        if lo >= seq_len - 1:
            break
        chunk = d_angle[:, lo:hi, :]
        velocity_by_pos.append({
            "range": f"{lo}-{hi}",
            "mean_abs_velocity": round(float(np.abs(chunk).mean()), 4),
            "velocity_std": round(float(chunk.std()), 4),
        })

    # Velocity-difficulty correlation
    velocity_difficulty_corr = None
    if embedding is not None:
        embed_norm = np.linalg.norm(embedding, axis=-1)
        mean_vel = np.abs(d_angle).mean(axis=-1)
        velocity_difficulty_corr = round(float(np.corrcoef(
            embed_norm[:, 1:].flatten(), mean_vel.flatten()
        )[0, 1]), 4)

    # Top position-correlated pairs
    top_idx = np.argsort(np.abs(angle_pos_corr))[-5:][::-1]
    top_pairs = [{"pair": int(p), "modes": (int(p*2), int(p*2+1)),
                  "position_r": round(float(angle_pos_corr[p]), 4)}
                 for p in top_idx]

    # Angle-based position probes
    flat_angles = angles.reshape(-1, angles.shape[-1])
    ang_s = flat_angles[idx]
    ang_tr = ang_s[train_idx, :n_check]
    ang_te = ang_s[test_idx, :n_check]
    angle_pos_linear = _linear_probe(ang_tr, ang_te, y_tr_pos, y_te_pos)
    angle_pos_mlp = _mlp_probe(ang_tr, ang_te, y_tr_pos, y_te_pos)

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "n_sample": len(idx),
        "probes": {
            "position": {
                "linear_pca50": pos_linear_pca,
                "mlp_pca50": pos_mlp_pca,
                "mlp_raw256": pos_mlp_raw,
                "angle_linear": angle_pos_linear,
                "angle_mlp": angle_pos_mlp,
            },
            "loss": {
                "linear_pca50": loss_linear_pca,
                "mlp_pca50": loss_mlp_pca,
            },
        },
        "angular": {
            "n_pairs_analyzed": n_check,
            "max_pair_position_r": round(float(np.abs(angle_pos_corr).max()), 4),
            "mean_pair_position_r": round(float(np.abs(angle_pos_corr).mean()), 4),
            "top_position_pairs": top_pairs,
            "velocity_by_position": velocity_by_pos,
            "velocity_difficulty_corr": velocity_difficulty_corr,
        },
    }


def causal_bank_gate_forensics(mri_path: str, *, _mri=None) -> dict:
    """Write gate forensics from sequence-mode causal bank MRI.

    Answers: does the overwrite gate encode order (position), or is it
    just a smoother EMA? Reports gate activation by position, correlation
    with embed norm (difficulty), per-mode entropy, effective rank, and
    a direct position-dependence test.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    # Support both gated_delta write gate and overwrite gate
    gate_type = None
    gate = None
    extra_gates = {}
    if 'gated_delta_write' in mri and mri.get('gated_delta_write') is not None:
        gate = mri['gated_delta_write'].astype(np.float32)
        gate_type = "gated_delta"
        if 'gated_delta_retain' in mri and mri.get('gated_delta_retain') is not None:
            extra_gates["retain"] = mri['gated_delta_retain'].astype(np.float32)
        if 'gated_delta_erase' in mri and mri.get('gated_delta_erase') is not None:
            extra_gates["erase"] = mri['gated_delta_erase'].astype(np.float32)
    elif 'overwrite_gate' in mri and mri.get('overwrite_gate') is not None:
        gate = mri['overwrite_gate'].astype(np.float32)
        gate_type = "overwrite"
    else:
        return {"error": "MRI has no gate data (no gated_delta or overwrite_gate)"}

    n_seqs, seq_len, n_gate_dims = gate.shape

    embedding = mri['embedding'].astype(np.float32) if 'embedding' in mri else None
    loss = mri.get('loss')
    half_lives = mri.get('half_lives')
    if half_lives is not None:
        half_lives = np.asarray(half_lives, dtype=np.float32)

    # --- Gate activation by position range ---
    ranges = [(0, 4), (4, 16), (16, 64), (64, 256), (256, seq_len)]
    by_position = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk = gate[:, lo:hi, :]  # [n_seqs, range, n_modes]
        mean_gate = float(chunk.mean())
        std_gate = float(chunk.std())
        # What fraction of modes have gate > 0.5 (overwrite dominant)?
        overwrite_frac = float((chunk > 0.5).mean())
        by_position.append({
            "range": f"{lo}-{hi}",
            "mean_gate": round(mean_gate, 4),
            "std_gate": round(std_gate, 4),
            "overwrite_fraction": round(overwrite_frac, 4),
        })

    # --- Position dependence test ---
    # Correlation of mean gate activation with position index
    mean_gate_by_pos = gate.mean(axis=(0, 2))  # [seq_len]
    positions = np.arange(seq_len, dtype=np.float32)
    pos_corr = float(np.corrcoef(mean_gate_by_pos, positions)[0, 1])

    # Per-mode position correlation (which modes' gates track position?)
    mode_pos_corr = []
    for m in range(n_gate_dims):
        gate_m = gate[:, :, m].mean(axis=0)  # [seq_len]
        r = float(np.corrcoef(gate_m, positions)[0, 1])
        mode_pos_corr.append(r)
    mode_pos_corr = np.array(mode_pos_corr)

    # Top 5 most position-dependent modes
    top_pos_modes = np.argsort(np.abs(mode_pos_corr))[-5:][::-1]
    position_dependent_modes = []
    for m in top_pos_modes:
        row = {"mode": int(m), "position_r": round(float(mode_pos_corr[m]), 4)}
        if half_lives is not None and m < len(half_lives):
            row["half_life"] = round(float(half_lives[m]), 1)
        position_dependent_modes.append(row)

    # --- Correlation with embedding norm (difficulty) ---
    difficulty_corr = None
    if embedding is not None:
        embed_norm = np.linalg.norm(embedding, axis=-1)  # [n_seqs, seq_len]
        gate_mean = gate.mean(axis=-1)  # [n_seqs, seq_len]
        difficulty_corr = round(
            float(np.corrcoef(embed_norm.flatten(), gate_mean.flatten())[0, 1]), 4)

    # --- Correlation with loss ---
    loss_corr = None
    if loss is not None:
        gate_mean = gate.mean(axis=-1).flatten()
        loss_flat = loss.flatten()
        valid = ~np.isnan(loss_flat)
        if valid.sum() > 10:
            loss_corr = round(
                float(np.corrcoef(gate_mean[valid], loss_flat[valid])[0, 1]), 4)

    # --- Per-mode gate entropy (how selective is each mode?) ---
    # High entropy = gate fires similarly everywhere. Low entropy = selective.
    # Discretize gate into 10 bins and compute entropy per mode.
    n_bins = 10
    mode_entropy = []
    for m in range(n_gate_dims):
        vals = gate[:, :, m].flatten()
        hist, _ = np.histogram(vals, bins=n_bins, range=(0, 1))
        p = hist / (hist.sum() + 1e-10)
        ent = float(-(p * np.log(p + 1e-10)).sum())
        mode_entropy.append(ent)
    mode_entropy = np.array(mode_entropy)
    max_entropy = float(np.log(n_bins))

    # --- Effective rank of gate (how many independent gate patterns?) ---
    # Flatten gate to [n_seqs * seq_len, n_modes], PCA
    flat_gate = gate.reshape(-1, n_gate_dims)
    flat_gate_c = flat_gate - flat_gate.mean(axis=0)
    _, S, _ = np.linalg.svd(flat_gate_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var_exp)
    eff_rank = float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum()))

    # --- Gate by half-life quartile (only when gate dims == substrate modes) ---
    gate_by_hl = []
    if half_lives is not None and len(half_lives) == n_gate_dims:
        quartile_edges = np.percentile(half_lives, [0, 25, 50, 75, 100])
        for q in range(4):
            lo_hl, hi_hl = quartile_edges[q], quartile_edges[q + 1]
            mask = (half_lives >= lo_hl) & (half_lives <= hi_hl) if q == 3 else \
                   (half_lives >= lo_hl) & (half_lives < hi_hl)
            if not mask.any():
                continue
            gate_q = gate[:, :, mask]
            gate_by_hl.append({
                "quartile": q,
                "hl_range": f"{lo_hl:.1f}-{hi_hl:.1f}",
                "n_modes": int(mask.sum()),
                "mean_gate": round(float(gate_q.mean()), 4),
                "overwrite_fraction": round(float((gate_q > 0.5).mean()), 4),
                "mean_pos_r": round(float(np.abs(mode_pos_corr[mask]).mean()), 4),
            })

    # --- Verdict ---
    encodes_order = abs(pos_corr) > 0.3 or float(np.abs(mode_pos_corr).max()) > 0.5
    verdict = ("ENCODES ORDER: gate activation is position-dependent"
               if encodes_order else
               "SMOOTHER EMA: gate is NOT position-dependent")

    # --- Extra gates summary (for gated_delta: retain and erase) ---
    extra_gate_summary = {}
    for gate_name, gate_arr in extra_gates.items():
        mean_by_pos = gate_arr.mean(axis=(0, 2))
        pos_r = float(np.corrcoef(mean_by_pos, positions)[0, 1])
        extra_gate_summary[gate_name] = {
            "mean": round(float(gate_arr.mean()), 4),
            "position_correlation": round(pos_r, 4),
        }

    return {
        "model": meta['model'].get('name', '?'),
        "gate_type": gate_type,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_gate_dims": n_gate_dims,
        "by_position": by_position,
        "position_correlation": round(pos_corr, 4),
        "difficulty_correlation": difficulty_corr,
        "loss_correlation": loss_corr,
        "effective_rank": round(eff_rank, 1),
        "effective_rank_pcs": {
            "pcs_for_50": int(np.searchsorted(cum, 0.5)) + 1,
            "pcs_for_80": int(np.searchsorted(cum, 0.8)) + 1,
            "pcs_for_95": int(np.searchsorted(cum, 0.95)) + 1,
        },
        "mean_mode_entropy": round(float(mode_entropy.mean()), 4),
        "max_possible_entropy": round(max_entropy, 4),
        "entropy_ratio": round(float(mode_entropy.mean()) / (max_entropy + 1e-10), 4),
        "position_dependent_modes": position_dependent_modes,
        "gate_by_half_life": gate_by_hl,
        "top_gate_variance_pct": [round(float(v) * 100, 2) for v in var_exp[:10]],
        "extra_gates": extra_gate_summary,
        "verdict": verdict,
    }


def causal_bank_substrate_local(mri_path: str, *, _mri=None) -> dict:
    """Substrate vs local path balance from sequence-mode causal bank MRI."""
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)
    n_seqs, seq_len, n_modes = substrate.shape
    sub_l2 = np.linalg.norm(substrate, axis=-1)

    has_local = 'local_norm' in mri and mri.get('local_norm') is not None
    local_l2 = mri['local_norm'].astype(np.float32) if has_local else None

    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        row = {
            "range": f"{lo}-{hi}",
            "substrate_l2": round(float(sub_l2[:, lo:hi].mean()), 4),
        }
        if local_l2 is not None:
            loc = float(local_l2[:, lo:hi].mean())
            row["local_l2"] = round(loc, 4)
            row["substrate_local_ratio"] = round(
                float(sub_l2[:, lo:hi].mean()) / (loc + 1e-10), 2)
        by_pos.append(row)

    crossover = None
    if local_l2 is not None:
        mean_sub = sub_l2.mean(axis=0)
        mean_loc = local_l2.mean(axis=0)
        crosses = np.where(mean_sub > mean_loc)[0]
        if len(crosses) > 0:
            crossover = int(crosses[0])

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "has_local": has_local,
        "by_position": by_pos,
        "crossover_position": crossover,
    }


def mri_decompose(mri_path: str, *, n_sample: int = 0,
                  n_components: int = 0) -> dict:
    """PCA decomposition at every layer, saved to decomp/ for the companion viewer.

    Produces:
      decomp/L{NN}_scores.npy     — [N, K] float16 PCA scores
      decomp/L{NN}_variance.npy   — [K] float32 variance explained per PC
      decomp/L{NN}_components.npy — [K, D] float32 principal components
      decomp/all_scores.bin       — single binary blob for the companion viewer
      decomp/meta.json            — per-layer stats + sample info

    Args:
        mri_path: Path to .mri directory
        n_sample: Number of tokens to sample. 0 = full vocabulary.
        n_components: Number of PCA components. 0 = all (hidden_size).
    """
    import struct
    import sys

    mri_dir = Path(mri_path)
    meta = json.loads((mri_dir / 'metadata.json').read_text())
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')

    tokens = dict(np.load(mri_dir / 'tokens.npz', allow_pickle=True))
    n_tok = len(tokens['token_ids'])

    # Sample selection
    if n_sample <= 0 or n_sample >= n_tok:
        idx = np.arange(n_tok)
        n_sample = n_tok
    else:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n_tok, n_sample, replace=False))

    decomp_dir = mri_dir / 'decomp'
    decomp_dir.mkdir(exist_ok=True)

    hidden_size = meta['model'].get('hidden_size', 0)
    K = n_components if n_components > 0 else hidden_size
    if K <= 0:
        return {"error": "Cannot determine hidden_size from metadata"}
    print(f"  {K} components (hidden_size={hidden_size})", file=sys.stderr)
    layer_meta = []

    # BIN_K: cap for binary blob. Full scores saved per-layer to disk,
    # only BIN_K columns accumulated in memory for the viewer blob.
    _est_total_layers = n_layers + 2  # real layers + emb + lmh
    _max_blob_bytes = 100 * 1024 * 1024
    _max_k = max(3, _max_blob_bytes // (_est_total_layers * n_sample * 2))
    BIN_K = min(K, 50, _max_k)

    # Collect capped scores for the binary blob
    all_variances = []  # [n_layers, BIN_K]
    all_scores = []     # [n_layers, N, BIN_K]

    def _find_exit(li):
        """Find exit file in nested or flat layout."""
        nested = mri_dir / 'layers' / f'L{li:02d}' / 'exit.npy'
        flat = mri_dir / f'L{li:02d}_exit.npy'
        return nested if nested.exists() else flat if flat.exists() else None

    _score_mmaps = {}  # cache score mmaps within this decompose run
    def _get_score_mmap_local(dd, li, n_real):
        """Get mmap'd score array for a layer (real or virtual)."""
        if li in _score_mmaps:
            return _score_mmaps[li]
        if li < n_real:
            sp = dd / f'L{li:02d}_scores.npy'
        elif li == n_real:
            sp = dd / 'emb_scores.npy'
        elif li == n_real + 1:
            sp = dd / 'lmh_scores.npy'
        else:
            return None
        if not sp.exists():
            return None
        arr = np.load(str(sp), mmap_mode='r')
        _score_mmaps[li] = arr
        return arr

    for li in range(n_layers):
        exit_path = _find_exit(li)
        if not exit_path:
            print(f"  L{li:02d}: no exit vectors, skipping", file=sys.stderr)
            layer_meta.append({"layer": li, "pc1_pct": 0, "intrinsic_dim": 0,
                               "neighbor_stability": 0})
            all_variances.append(np.zeros(K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, BIN_K), dtype=np.float16))
            continue

        print(f"  L{li:02d}/{n_layers}  ({n_sample} tokens, {K} PCs)...",
              end="\r", file=sys.stderr)

        vecs = np.load(str(exit_path), mmap_mode='r')[idx].astype(np.float32)
        centered = vecs - vecs.mean(axis=0)
        # Randomized SVD: O(n*k*d) instead of O(n*d^2). 18x faster for 150K tokens.
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(centered, n_components=K, random_state=42)

        # Keep top K
        k = min(K, len(S))
        scores = (U[:, :k] * S[:k]).astype(np.float16)   # [N, k]
        variance = ((S[:k] ** 2) / (S ** 2).sum()).astype(np.float32)
        components = Vt[:k].astype(np.float32)            # [k, D]

        # Pad to K if fewer components available
        if k < K:
            scores = np.pad(scores, ((0, 0), (0, K - k)))
            variance = np.pad(variance, (0, K - k))
            components = np.pad(components, ((0, K - k), (0, 0)))

        np.save(decomp_dir / f'L{li:02d}_scores.npy', scores)
        np.save(decomp_dir / f'L{li:02d}_variance.npy', variance)
        np.save(decomp_dir / f'L{li:02d}_components.npy', components)

        all_variances.append(variance)           # full K (tiny, ~2KB/layer)
        all_scores.append(scores[:, :BIN_K].copy())  # capped (large)

        # Per-layer stats
        var_ratio = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(var_ratio)
        intrinsic = float(np.searchsorted(cum, 0.5)) + 1

        # Neighbor stability: fraction of 5-NN consistent between full and 3D
        nbr_stab = 0.0
        if n_sample <= 10000:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn_full = NearestNeighbors(n_neighbors=6).fit(scores.astype(np.float32))
                nn_3d = NearestNeighbors(n_neighbors=6).fit(scores[:, :3].astype(np.float32))
                _, idx_full = nn_full.kneighbors()
                _, idx_3d = nn_3d.kneighbors()
                overlaps = [len(set(a[1:]) & set(b[1:])) / 5
                            for a, b in zip(idx_full, idx_3d)]
                nbr_stab = float(np.mean(overlaps))
            except ImportError:
                nbr_stab = 0.0

        layer_meta.append({
            "layer": li,
            "pc1_pct": round(float(var_ratio[0]) * 100, 1),
            "intrinsic_dim": round(intrinsic, 1),
            "neighbor_stability": round(nbr_stab, 2),
        })

    print(f"  {n_layers} layers done.{' ' * 30}", file=sys.stderr)

    # Virtual layers: embedding and lm_head
    token_ids = tokens['token_ids']
    for vname, vpath in [('emb', 'embedding.npy'), ('lmh', 'lmhead.npy')]:
        vf = mri_dir / vpath
        if not vf.exists():
            print(f"  {vname}: not found, skipping", file=sys.stderr)
            all_variances.append(np.zeros(K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, BIN_K), dtype=np.float16))
            layer_meta.append({"layer": vname, "pc1_pct": 0, "intrinsic_dim": 0, "neighbor_stability": 0})
            continue
        print(f"  {vname} PCA...", file=sys.stderr)
        raw = np.load(str(vf), mmap_mode='r')
        vecs = raw[token_ids[idx]].astype(np.float32) if raw.shape[0] > n_sample else raw[idx].astype(np.float32)
        centered = vecs - vecs.mean(axis=0)
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(centered, n_components=K, random_state=42)
        k = min(K, len(S))
        scores = (U[:, :k] * S[:k]).astype(np.float16)
        variance = ((S[:k] ** 2) / (S ** 2).sum()).astype(np.float32)
        if k < K:
            scores = np.pad(scores, ((0, 0), (0, K - k)))
            variance = np.pad(variance, (0, K - k))
        all_variances.append(variance)              # full K
        all_scores.append(scores[:, :BIN_K].copy())  # capped
        var_ratio = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(var_ratio)
        layer_meta.append({"layer": vname, "pc1_pct": round(float(var_ratio[0]) * 100, 1),
                           "intrinsic_dim": round(float(np.searchsorted(cum, 0.5)) + 1, 1), "neighbor_stability": 0})
        np.save(decomp_dir / f'{vname}_scores.npy', scores)
        np.save(decomp_dir / f'{vname}_variance.npy', variance)

    # Precompute weight-PC alignment per layer
    align_data = []
    weights_root = mri_dir / 'weights'
    for li in range(n_layers):
        comp_path = decomp_dir / f'L{li:02d}_components.npy'
        wdir = weights_root / f'L{li:02d}'
        la = {"layer": li, "matrices": []}
        if comp_path.exists() and wdir.exists():
            Vt = np.load(str(comp_path)).astype(np.float32)  # all PCs
            for wname in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
                wp = wdir / f"{wname}.npy"
                if not wp.exists(): continue
                W = np.load(str(wp), mmap_mode='r').astype(np.float32)
                if W.shape[1] == Vt.shape[1]:
                    al = np.abs(Vt @ W.T).max(axis=1)
                elif W.shape[0] == Vt.shape[1]:
                    al = np.abs(Vt @ W).max(axis=1)
                else: continue
                al = al / (al.max() + 1e-8)
                la["matrices"].append({"name": wname, "alignment": [round(float(a), 3) for a in al]})
        align_data.append(la)
    (decomp_dir / 'weight_alignment.json').write_text(json.dumps(align_data))
    print(f"  Weight alignment: {len(align_data)} layers", file=sys.stderr)

    # Precompute gate biography summary per token (top neurons per layer)
    gate_summary = {}
    mlp_dir = mri_dir / 'mlp'
    if mlp_dir.exists():
        gate_layers = []
        for li in range(n_layers):
            gp = mlp_dir / f'L{li:02d}_gate.npy'
            if gp.exists():
                g = np.load(str(gp), mmap_mode='r')  # [N, intermediate]
                # Per-layer stats: mean_abs and max_abs per token (for fast lookup)
                gate_layers.append({"layer": li, "shape": list(g.shape)})
            else:
                gate_layers.append({"layer": li, "shape": None})
        gate_summary["layers"] = gate_layers
        gate_summary["n_tokens"] = n_sample
        (decomp_dir / 'gate_summary.json').write_text(json.dumps(gate_summary))

    total_layers = len(all_scores)
    _var_k = len(all_variances[0]) if all_variances else K
    print(f"  Blob: {total_layers} layers x {n_sample} tokens x {BIN_K} score PCs + {_var_k} var PCs = "
          f"{(total_layers * n_sample * BIN_K * 2 + total_layers * _var_k * 4) / (1024*1024):.0f}MB",
          file=sys.stderr)

    # === Precompute delta PCA (exit - entry) ===
    delta_scores_all = []
    delta_var_all = []
    for li in range(n_layers):
        exit_path = _find_exit(li)
        entry_nested = mri_dir / 'layers' / f'L{li:02d}' / 'entry.npy'
        entry_flat = mri_dir / f'L{li:02d}_entry.npy'
        entry_path = entry_nested if entry_nested.exists() else entry_flat if entry_flat.exists() else None
        if entry_path and exit_path:
            print(f"  delta L{li:02d}...", end="\r", file=sys.stderr)
            entry = np.load(str(entry_path), mmap_mode='r')[idx].astype(np.float32)
            exits = np.load(str(exit_path), mmap_mode='r')[idx].astype(np.float32)
            delta = exits - entry
            centered = delta - delta.mean(axis=0)
            from sklearn.utils.extmath import randomized_svd
            dK = min(K, BIN_K)  # delta uses viewer cap
            U, S, Vt = randomized_svd(centered, n_components=dK, random_state=42)
            k = min(dK, len(S))
            ds = (U[:, :k] * S[:k]).astype(np.float16)
            _ss = (S ** 2).sum()
            dv = ((S[:k] ** 2) / _ss).astype(np.float32) if _ss > 0 else np.zeros(k, dtype=np.float32)
            if k < dK: ds = np.pad(ds, ((0, 0), (0, dK - k))); dv = np.pad(dv, (0, dK - k))
            delta_scores_all.append(ds); delta_var_all.append(dv)
        else:
            delta_scores_all.append(np.zeros((n_sample, min(K, BIN_K)), dtype=np.float16))
            delta_var_all.append(np.zeros(min(K, BIN_K), dtype=np.float32))
    if any(v.any() for v in delta_var_all):
        dK = min(K, BIN_K)
        d_header = struct.pack('<III', n_layers, n_sample, dK)
        d_var = np.concatenate(delta_var_all).tobytes()
        d_sc = np.concatenate([s.view(np.uint16).ravel() for s in delta_scores_all]).tobytes()
        with open(decomp_dir / 'delta_scores.bin', 'wb') as f:
            f.write(d_header); f.write(d_var); f.write(d_sc)
        print(f"  Delta PCA: {len(delta_scores_all)} layers, {dK} PCs", file=sys.stderr)

    # === Gate heatmap + neuron profiles in one pass ===
    # Single sequential read per layer (no mmap — critical for NAS/USB).
    # Computes both gate heatmap (max |g*u| per token) and neuron importance
    # (mean |g*u| per neuron) from the same data.
    mlp_dir = mri_dir / 'mlp'
    if mlp_dir.exists():
        print(f"  Gate heatmap + neuron profiles...", file=sys.stderr)
        gate_heat = np.zeros((n_sample, n_layers), dtype=np.float16)
        neuron_importance = []
        for li in range(n_layers):
            gp = mlp_dir / f'L{li:02d}_gate.npy'
            up_p = mlp_dir / f'L{li:02d}_up.npy'
            if gp.exists() and up_p.exists():
                # Sequential read: one I/O per file, entire array into RAM
                g = np.load(str(gp))   # no mmap — reads whole file once
                u = np.load(str(up_p))
                _inter = g.shape[1]
                # Process in chunks to limit float32 peak (gate+up are float16 on disk)
                _heat_col = np.zeros(n_sample, dtype=np.float32)
                _contrib = np.zeros(_inter, dtype=np.float64)
                _chunk = 16384
                for _s in range(0, n_sample, _chunk):
                    _e = min(_s + _chunk, n_sample)
                    _gc = g[_s:_e].astype(np.float32)
                    _uc = u[_s:_e].astype(np.float32)
                    _gu = np.abs(_gc * _uc)
                    _heat_col[_s:_e] = _gu.max(axis=1)
                    _contrib += _gu.sum(axis=0)
                gate_heat[:, li] = _heat_col.astype(np.float16)
                contrib = _contrib / n_sample
                top50 = np.argsort(contrib)[-50:][::-1]
                neuron_importance.append({"layer": li, "top_neurons": top50.tolist(),
                    "top_contrib": contrib[top50].tolist()})
                del g, u  # free RAM before next layer
            elif gp.exists():
                g = np.load(str(gp))
                _chunk = 16384
                _col = np.zeros(n_sample, dtype=np.float32)
                for _s in range(0, n_sample, _chunk):
                    _e = min(_s + _chunk, n_sample)
                    _col[_s:_e] = np.abs(g[_s:_e].astype(np.float32)).max(axis=1)
                gate_heat[:, li] = _col.astype(np.float16)
                neuron_importance.append({"layer": li, "top_neurons": [], "top_contrib": []})
                del g
            else:
                neuron_importance.append({"layer": li, "top_neurons": [], "top_contrib": []})
            if (li + 1) % 5 == 0:
                print(f"    L{li}/{n_layers}", file=sys.stderr)
        np.save(decomp_dir / 'gate_heatmap.npy', gate_heat)
        (decomp_dir / 'neuron_importance.json').write_text(json.dumps(neuron_importance))
        print(f"  Gate heatmap: {gate_heat.shape}, neurons: {len(neuron_importance)} layers",
              file=sys.stderr)

    # === Precompute attention summary ===
    attn_dir = mri_dir / 'attention'
    if attn_dir.exists():
        print(f"  Attention summary...", file=sys.stderr)
        attn_summary = []
        for li in range(n_layers):
            wp = attn_dir / f'L{li:02d}_weights.npy'
            lp = attn_dir / f'L{li:02d}_logits.npy'
            fp = wp if wp.exists() else lp
            if fp.exists():
                a = np.load(str(fp), mmap_mode='r')  # [N, heads, seq_len]
                # Mean attention per head across all tokens
                mean_attn = np.abs(a).mean(axis=0).tolist()  # [heads, seq_len]
                attn_summary.append({"layer": li, "n_heads": a.shape[1],
                    "seq_len": a.shape[2], "mean_attn": mean_attn})
            else:
                attn_summary.append({"layer": li, "n_heads": 0, "seq_len": 0, "mean_attn": []})
        (decomp_dir / 'attention_summary.json').write_text(json.dumps(attn_summary))

    total_layers = len(all_scores)  # n_layers + 2 virtual layers

    # Build all_scores.bin (includes virtual layers)
    # Scores capped to BIN_K (50) for memory. Variance at full K (tiny).
    # Format v2: [4B magic 'HEI2'][4B n_layers][4B n_tokens][4B score_k][4B var_k]
    #            [float32 variance: n_layers * var_k]
    #            [uint16 scores: n_layers * N * score_k]
    VAR_K = len(all_variances[0]) if all_variances else K
    header = struct.pack('<4sIIII', b'HEI2', total_layers, n_sample, BIN_K, VAR_K)
    var_block = np.concatenate(all_variances).tobytes()
    score_block = np.concatenate([s.view(np.uint16).ravel()
                                  for s in all_scores]).tobytes()
    bin_path = decomp_dir / 'all_scores.bin'
    with open(bin_path, 'wb') as f:
        f.write(header)
        f.write(var_block)
        f.write(score_block)

    bin_size = len(header) + len(var_block) + len(score_block)

    # === Transposed per-token index: [N_tokens × N_layers × K] float16 ===
    # One seek per token instead of N_layers file reads. Critical for USB/NAS.
    # Token scores index: layer-by-layer build (48 sequential reads, not 3.9M page faults)
    print(f"  Token index (scores)...", file=sys.stderr)
    tok_scores_path = decomp_dir / 'token_scores.bin'
    tok_hdr = struct.pack('<4sIII', b'TOKS', n_sample, total_layers, K)
    stride = total_layers * K * 2
    # Pre-allocate output as memmap for large files
    with open(tok_scores_path, 'wb') as f:
        f.write(tok_hdr)
        f.write(b'\x00' * (n_sample * stride))  # reserve space
    out = np.memmap(str(tok_scores_path), dtype=np.float16, mode='r+',
                    offset=16, shape=(n_sample, total_layers, K))
    for li in range(total_layers):
        sp = _get_score_mmap_local(decomp_dir, li, n_layers)
        if sp is not None:
            # One sequential read of the entire layer file
            data = np.array(sp)  # force full read from mmap
            nk = min(K, data.shape[1])
            out[:data.shape[0], li, :nk] = data[:, :nk]
            del data
        if (li + 1) % 5 == 0:
            print(f"    layer {li+1}/{total_layers}", file=sys.stderr)
    out.flush(); del out
    print(f"  Token scores index: {n_sample} tokens × {total_layers} layers × {K} PCs = "
          f"{(16 + n_sample * stride) / (1024*1024):.0f}MB", file=sys.stderr)

    # === PC-major index: [K × N_layers × N_tokens] float16 ===
    # One seek per PC for cloud viewport queries. Iterates by PC, reads each layer's column.
    print(f"  PC-major index...", file=sys.stderr)
    pc_scores_path = decomp_dir / 'pc_scores.bin'
    pc_hdr = struct.pack('<4sIII', b'PCSC', total_layers, n_sample, K)
    pc_stride = total_layers * n_sample * 2  # bytes per PC
    with open(pc_scores_path, 'wb') as f:
        f.write(pc_hdr)
        for pc in range(K):
            slab = np.zeros((total_layers, n_sample), dtype=np.float16)
            for li in range(total_layers):
                sp = _get_score_mmap_local(decomp_dir, li, n_layers)
                if sp is not None and pc < sp.shape[1]:
                    slab[li] = sp[:, pc]
            f.write(slab.tobytes())
            if (pc + 1) % 50 == 0:
                print(f"    PC {pc+1}/{K}", file=sys.stderr)
    print(f"  PC-major index: {K} PCs × {total_layers} layers × {n_sample} tokens = "
          f"{(16 + K * pc_stride) / (1024*1024):.0f}MB", file=sys.stderr)

    # === Transposed per-token neuron index: [N_tokens × N_layers × intermediate] float16 ===
    # Layer-by-layer build: 48 sequential reads instead of 7.2M page faults
    if mlp_dir.exists() and n_layers > 0:
        _gp0 = mlp_dir / 'L00_gate.npy'
        if _gp0.exists():
            _inter = np.load(str(_gp0), mmap_mode='r').shape[1]
            print(f"  Token index (neurons)...", file=sys.stderr)
            tok_neurons_path = decomp_dir / 'token_neurons.bin'
            n_hdr = struct.pack('<4sIII', b'TOKN', n_sample, n_layers, _inter)
            n_stride = n_layers * _inter * 2
            # Pre-allocate output as memmap
            with open(tok_neurons_path, 'wb') as f:
                f.write(n_hdr)
                f.write(b'\x00' * (n_sample * n_stride))
            out = np.memmap(str(tok_neurons_path), dtype=np.float16, mode='r+',
                            offset=16, shape=(n_sample, n_layers, _inter))
            for li in range(n_layers):
                gp = mlp_dir / f'L{li:02d}_gate.npy'
                up_p = mlp_dir / f'L{li:02d}_up.npy'
                if not gp.exists():
                    continue
                # Sequential read: one I/O per file (not mmap)
                g = np.load(str(gp))
                u = np.load(str(up_p)) if up_p.exists() else None
                # Compute gate×up for all tokens at this layer in one vectorized op
                if u is not None:
                    out[:g.shape[0], li, :] = (g.astype(np.float32) * u.astype(np.float32)).astype(np.float16)
                else:
                    out[:g.shape[0], li, :] = g
                del g, u
                if (li + 1) % 5 == 0:
                    print(f"    layer {li+1}/{n_layers}", file=sys.stderr)
            out.flush(); del out
            print(f"  Token neuron index: {n_sample} tokens × {n_layers} layers × {_inter} neurons = "
                  f"{(16 + n_sample * n_stride) / (1024*1024):.0f}MB", file=sys.stderr)

    # Save metadata (sample_indices as "all" for full vocab to avoid huge JSON)
    meta_out = {
        "n_sample": n_sample,
        "n_components": K,
        "n_layers": total_layers,
        "n_real_layers": n_layers,
        "method": "pca",
        "virtual_layers": ["emb", "lmh"],
        "sample_indices": "all" if n_sample == n_tok else idx.tolist(),
        "layers": layer_meta,
    }
    (decomp_dir / 'meta.json').write_text(json.dumps(meta_out, indent=2))

    return {
        "model": model_name,
        "mode": mode,
        "mri_path": str(mri_dir),
        "n_tokens": n_sample,
        "n_components": K,
        "n_layers": n_layers,
        "bin_size_mb": round(bin_size / 1024 / 1024, 1),
        "layers": layer_meta,
    }
