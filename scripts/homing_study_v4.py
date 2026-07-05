#!/usr/bin/env python3
"""Homing study v4 — cross-model replication of the readout-commit band.

The base smollm2-135m (v3) committed its readout to the answer token in a late
band, L20-L27 (23/28). This replays the identical protocol (28 prompt pairs +
16-token background panel from v3) on other models, via the live companion's
/api/homing-run readout path, to test whether that band is a property of one
small Llama or a general phenomenon — and whether RLHF (instruct) or depth (360M)
moves it. Figure 2 of paper/heinrich_method.tex.

Reproduce: companion running (heinrich companion --mri-root web/.data), then
  python scripts/homing_study_v4.py
Writes docs/data/homing-study-v4-<model>.json per model.
"""
from __future__ import annotations
import json
import time
import urllib.request
from collections import Counter

COMPANION = "http://127.0.0.1:8377"
V3 = "docs/data/homing-study-v3-corrected.json"
MODELS = [("smollm2-135m-instruct", "raw"), ("smollm2-360m", "raw")]


def post(path: str, body: dict, timeout: float = 220.0) -> dict:
    req = urllib.request.Request(
        COMPANION + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def rank_curve_for(resp: dict, answer: str, background: list) -> tuple:
    """Per layer, the answer's rank among the background by readout value
    (0 = answer out-reads all 16 background). Returns (layers, rank_curve)."""
    by = {t["text"]: {c["l"]: c["v"] for c in t.get("readout_curve", [])}
          for t in resp.get("targets", [])}
    ans = by.get(answer, {})
    bg = [by.get(b, {}) for b in background]
    layers = sorted(ans.keys())
    rc = []
    for L in layers:
        av = ans[L]
        rc.append(sum(1 for bc in bg if bc.get(L, -1e30) > av))
    return layers, rc


def l_star_read(layers: list, rc: list):
    """First layer from which the answer out-reads the whole panel persistently."""
    if not rc or rc[-1] != 0:
        return None
    j = len(rc) - 1
    while j - 1 >= 0 and rc[j - 1] == 0:
        j -= 1
    return layers[j]


def aggregate(subset: list) -> dict:
    n = len(subset)
    hist = Counter(str(r["l_star_read"]) for r in subset if r["l_star_read"] is not None)
    fr0 = sum(1 for r in subset if r["final_rank"] == 0)
    mfr = (sum(r["final_rank"] for r in subset if r["final_rank"] is not None)
           / n) if n else 0.0
    return {"n": n, "l_star_read_hist": dict(hist),
            "final_rank0": fr0, "mean_final_rank": round(mfr, 3)}


def main() -> None:
    proto = json.load(open(V3))
    background = proto["background"]
    triples = [(r["prompt"], r["answer"], r["distractor"]) for r in proto["runs"]]

    for model, mode in MODELS:
        print(f"\n=== {model} ({mode}) ===")
        runs = []
        for i, (prompt, answer, distractor) in enumerate(triples):
            targets = [answer, distractor] + background
            try:
                resp = post("/api/homing-run", {
                    "mri_model": model, "mode": mode, "prompt": prompt,
                    "targets": targets, "readout": True})
            except Exception as e:
                print(f"  {i+1}/28 REQUEST FAILED: {e}")
                continue
            if "error" in resp:
                print(f"  {i+1}/28 ERROR: {resp['error'][:120]}")
                continue
            layers, rc = rank_curve_for(resp, answer, background)
            ls = l_star_read(layers, rc)
            pred = resp.get("prediction")
            correct = (pred == answer)
            runs.append({"prompt": prompt, "answer": answer, "distractor": distractor,
                         "prediction": pred, "correct": correct,
                         "layers": layers, "rank_curve": rc,
                         "l_star_read": ls, "final_rank": rc[-1] if rc else None})
            print(f"  {i+1}/28 {answer!r:>12} pred={pred!r:>10} "
                  f"correct={str(correct):5} l*={ls} final={rc[-1] if rc else None}")

        n_layers = len(runs[0]["layers"]) if runs else 0
        result = {
            "model": model, "mode": mode, "measure": "readout (logit lens)",
            "generated_at": time.time(), "n_layers": n_layers,
            "background": background, "runs": runs,
            "aggregate": {
                "all": aggregate(runs),
                "correct": aggregate([r for r in runs if r["correct"]]),
                "incorrect": aggregate([r for r in runs if not r["correct"]]),
            },
        }
        out = f"docs/data/homing-study-v4-{model}.json"
        json.dump(result, open(out, "w"), indent=1)
        a = result["aggregate"]["all"]
        print(f"  WROTE {out}  n_layers={n_layers} "
              f"final_rank0={a['final_rank0']}/{a['n']} l*_hist={a['l_star_read_hist']}")


if __name__ == "__main__":
    main()
