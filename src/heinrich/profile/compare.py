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

    # PCA via SVD
    centered = all_vecs - all_vecs.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    variance = S ** 2
    total_var = float(variance.sum())

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
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        variance = S ** 2
        total_var = float(variance.sum())

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
                    if len(shared) < 5:
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
