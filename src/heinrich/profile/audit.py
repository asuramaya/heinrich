"""Direction audit: the 5-test pipeline as a library function.

Given a model and ≥1 labeled-pairs CSV (columns: statement, label where
label ∈ {0,1}), learn a mean-diff direction on the first dataset, run the
falsification battery, and report per-test outcomes.

Tests implemented:
  1. In-domain held-out classification accuracy (train/test split).
  2. Bootstrap stability (anchor cosine p5 across resamples).
  3. Permutation null (p95 of Cohen's d under label shuffling).
  4. Cross-dataset transfer (apply direction to each extra dataset).
  5. Vocab projection (top-k lm_head tokens for +/- direction halves).

Verdict is one of robust_feature / partial / falsified per the 5-test
convention in papers/lie-detection/ATTACK_PLAN.md.
"""
from __future__ import annotations
from typing import Any
import csv
import json
import time
import numpy as np


DEFAULT_TRUTH_TOKENS = [
    "true", "True", "TRUE", "correct", "Correct", "right", "yes", "Yes",
    "accurate", "factual", "real",
]
DEFAULT_FALSE_TOKENS = [
    "false", "False", "FALSE", "wrong", "Wrong", "incorrect", "Incorrect",
    "no", "No", "not", "untrue", "lie",
]


def _load_pairs(csv_path: str, n_per_class: int) -> tuple[list[str], list[int]]:
    t_s: list[str] = []
    f_s: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == "1" and len(t_s) < n_per_class:
                t_s.append(row["statement"])
            elif row.get("label") == "0" and len(f_s) < n_per_class:
                f_s.append(row["statement"])
            if len(t_s) == n_per_class and len(f_s) == n_per_class:
                break
    stmts = t_s + f_s
    labels = [1] * len(t_s) + [0] * len(f_s)
    return stmts, labels


def _capture_residuals(model, tokenizer, stmts: list[str], layer: int) -> np.ndarray:
    from heinrich.cartography.runtime import forward_pass
    first = forward_pass(model, tokenizer, stmts[0], return_residual=True, residual_layer=layer)
    hidden = first["residual"].shape[0]
    out = np.zeros((len(stmts), hidden), dtype=np.float32)
    out[0] = first["residual"]
    for i, s in enumerate(stmts[1:], start=1):
        r = forward_pass(model, tokenizer, s, return_residual=True, residual_layer=layer)
        out[i] = r["residual"]
    return out


def _train_direction(At: np.ndarray, Af: np.ndarray) -> tuple[np.ndarray | None, float]:
    d = At.mean(0) - Af.mean(0)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return None, 0.0
    u = d / n
    c = float(((At.mean(0) + Af.mean(0)) / 2) @ u)
    return u, c


def _classify(A: np.ndarray, y: np.ndarray, u: np.ndarray, c: float) -> float:
    p = A @ u
    pred = (p > c).astype(int)
    acc1 = float((pred == y).mean())
    acc2 = float(((1 - pred) == y).mean())
    return max(acc1, acc2)


def _cohen_d(At: np.ndarray, Af: np.ndarray, u: np.ndarray) -> float:
    center = (At.mean(0) + Af.mean(0)) / 2
    st = (At - center) @ u
    sf = (Af - center) @ u
    pooled = float(np.sqrt((st.var() + sf.var()) / 2))
    return float((st.mean() - sf.mean()) / (pooled + 1e-9))


def _bootstrap_p5(At: np.ndarray, Af: np.ndarray, u_anchor: np.ndarray,
                  rng: np.random.Generator, n_boot: int) -> float:
    T = len(At)
    n_anchor = float(np.linalg.norm(At.mean(0) - Af.mean(0)))
    cos_list = []
    for _ in range(n_boot):
        bi_t = rng.choice(T, T, replace=True)
        bi_f = rng.choice(T, T, replace=True)
        db = At[bi_t].mean(0) - Af[bi_f].mean(0)
        nb = float(np.linalg.norm(db))
        cos_list.append(float(db @ u_anchor / (nb + 1e-9)) if nb > 1e-8 else 0.0)
    return float(np.percentile(cos_list, 5))


def _permutation_p95(At: np.ndarray, Af: np.ndarray,
                     rng: np.random.Generator, n_perm: int) -> float:
    T = len(At)
    A_all = np.concatenate([At, Af], axis=0)
    perm_d = []
    for _ in range(n_perm):
        p = rng.permutation(2 * T)
        Atp, Afp = A_all[p[:T]], A_all[p[T:]]
        d = Atp.mean(0) - Afp.mean(0)
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            perm_d.append(0.0)
            continue
        u = d / n
        c = (Atp.mean(0) + Afp.mean(0)) / 2
        st = (Atp - c) @ u
        sf = (Afp - c) @ u
        pooled = float(np.sqrt((st.var() + sf.var()) / 2))
        perm_d.append(abs(st.mean() - sf.mean()) / (pooled + 1e-9))
    return float(np.percentile(perm_d, 95))


def _vocab_project(model, tokenizer, u: np.ndarray,
                   truth_ids: set[int], false_ids: set[int],
                   top_k: int = 10) -> dict[str, Any]:
    import mlx.core as mx
    dm = mx.array(u.astype(np.float32).reshape(1, 1, -1))
    if hasattr(model, "lm_head"):
        logits_mx = model.lm_head(dm)
    else:
        inner = getattr(model, "model", model)
        logits_mx = inner.embed_tokens.as_linear(dm)
    logits = np.array(logits_mx.astype(mx.float32)[0, 0, :])
    top_pos = np.argsort(logits)[-top_k:][::-1]
    top_neg = np.argsort(logits)[:top_k]
    pos_toks = [tokenizer.decode([int(i)]) for i in top_pos]
    neg_toks = [tokenizer.decode([int(i)]) for i in top_neg]
    pos_truth = sum(1 for i in top_pos if int(i) in truth_ids)
    pos_false = sum(1 for i in top_pos if int(i) in false_ids)
    neg_truth = sum(1 for i in top_neg if int(i) in truth_ids)
    neg_false = sum(1 for i in top_neg if int(i) in false_ids)
    vocab_pass = (
        (pos_truth > 0 and neg_false > 0) or
        (pos_false > 0 and neg_truth > 0)
    )
    return {
        "pos_top": pos_toks, "neg_top": neg_toks,
        "pos_truth": pos_truth, "pos_false": pos_false,
        "neg_truth": neg_truth, "neg_false": neg_false,
        "vocab_pass": vocab_pass,
    }


def _resolve_token_ids(tokenizer, words: list[str]) -> set[int]:
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w):
            try:
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    ids.add(int(toks[0]))
            except Exception:
                pass
    return ids


def audit_direction(
    model_id: str,
    datasets: list[str],
    layer: int,
    *,
    n_per_class: int = 80,
    train_frac: float = 0.8,
    seed: int = 42,
    n_bootstrap: int = 50,
    n_permutation: int = 200,
    truth_tokens: list[str] | None = None,
    false_tokens: list[str] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run the 5-test audit on a mean-diff direction at (model, layer).

    Parameters:
        model_id: HF model ID loadable by mlx_lm.
        datasets: list of CSV paths. First is train-on; rest are transfer targets.
            Each CSV must have columns `statement` and `label` (0 or 1).
        layer: layer index where residual is captured.
        n_per_class: statements per class per dataset.
        train_frac: fraction of first dataset used for training the direction.
        seed: RNG seed for bootstrap / permutation / split.
        n_bootstrap: bootstrap resamples for stability p5.
        n_permutation: permutations for null p95.
        truth_tokens / false_tokens: single-token words to check in lm_head top-k.
            None → DEFAULT_TRUTH_TOKENS / DEFAULT_FALSE_TOKENS.
        output_path: if set, write JSON result to this path.

    Returns dict with keys:
        in_domain_test_acc, cohens_d, boot_p5, perm_p95, snr, vocab,
        transfer (dict by dataset), verdict.
    """
    from heinrich.cartography.runtime import load_model

    rng = np.random.default_rng(seed)
    truth_tokens = truth_tokens or DEFAULT_TRUTH_TOKENS
    false_tokens = false_tokens or DEFAULT_FALSE_TOKENS

    t0 = time.time()
    model, tokenizer = load_model(model_id)
    truth_ids = _resolve_token_ids(tokenizer, truth_tokens)
    false_ids = _resolve_token_ids(tokenizer, false_tokens)

    # Capture residuals per dataset
    per_ds: dict[str, dict[str, np.ndarray]] = {}
    for csv_path in datasets:
        stmts, labels = _load_pairs(csv_path, n_per_class)
        act = _capture_residuals(model, tokenizer, stmts, layer)
        per_ds[csv_path] = {
            "act": act,
            "labels": np.array(labels),
        }

    # Train on first dataset's train split
    first = datasets[0]
    act_first = per_ds[first]["act"]
    lbls_first = per_ds[first]["labels"]
    idx = rng.permutation(len(act_first))
    n_tr = int(len(act_first) * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]
    At_tr = act_first[tr][lbls_first[tr] == 1]
    Af_tr = act_first[tr][lbls_first[tr] == 0]
    u, c = _train_direction(At_tr, Af_tr)
    if u is None:
        return {"error": "zero direction on train set"}

    # Test 1: in-domain held-out
    te_acc = _classify(act_first[te], lbls_first[te], u, c)
    # Cohen's d (on train pairs, same as audit convention)
    d = _cohen_d(At_tr, Af_tr, u)
    # Test 2: bootstrap p5
    boot_p5 = _bootstrap_p5(At_tr, Af_tr, u, rng, n_bootstrap)
    # Test 3: permutation p95
    perm_p95 = _permutation_p95(At_tr, Af_tr, rng, n_permutation)
    snr = float(abs(d) / (perm_p95 + 1e-9))
    # Test 5: vocab projection (numbered 5 in the pipeline but cheap here)
    vocab = _vocab_project(model, tokenizer, u, truth_ids, false_ids)
    # Test 4: transfer
    transfer: dict[str, float] = {}
    for csv_path in datasets[1:]:
        act = per_ds[csv_path]["act"]
        lbl = per_ds[csv_path]["labels"]
        transfer[csv_path] = _classify(act, lbl, u, c)

    # Verdict
    boot_pass = boot_p5 > 0.7
    perm_pass = abs(d) > perm_p95
    in_domain_pass = te_acc > 0.8
    transfer_pass = (
        len(transfer) == 0 or
        all(a > 0.7 for a in transfer.values())
    )
    n_pass = sum([boot_pass, perm_pass, in_domain_pass, transfer_pass, vocab["vocab_pass"]])
    if n_pass == 5:
        verdict = "robust_feature"
    elif n_pass >= 3:
        verdict = "partial"
    else:
        verdict = "falsified"

    result = {
        "model": model_id,
        "layer": layer,
        "train_dataset": first,
        "transfer_datasets": list(datasets[1:]),
        "n_per_class": n_per_class,
        "in_domain_test_acc": te_acc,
        "cohens_d": d,
        "boot_p5": boot_p5,
        "perm_p95": perm_p95,
        "snr": snr,
        "transfer": transfer,
        "vocab": vocab,
        "tests_passed": {
            "in_domain": in_domain_pass,
            "bootstrap": boot_pass,
            "permutation": perm_pass,
            "transfer": transfer_pass,
            "vocab": vocab["vocab_pass"],
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result
