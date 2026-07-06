#!/usr/bin/env python3
"""Experiment: The Between — does the ~3/4 commit ride on an attention relation?

For each completion prompt, one forward pass (eager attention) measures, per layer, on
the LAST token: (a) the readout-lens rank of the answer among a 16-word background
panel, which locates the commit layer L* (as in the homing study); and (b) the
attention mass the last token places on the answer-LICENSING context word, against
control tokens. Tests whether the commit coincides with the last token binding,
through attention, to the token that licenses its answer — i.e. whether the decision
lives in a relation the positional frame is blind to.

Reproduce: python scripts/experiment_between.py
Writes docs/data/experiment-between-smollm2-135m.json
"""
from __future__ import annotations
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
BG = [" table", " music", " river", " doctor", " happy", " window", " garden",
      " metal", " quiet", " horse", " paper", " coffee", " stone", " engine",
      " lucky", " sugar"]
# Two prompt sets over the same 8 capitals. In ADJACENT the licenser (the country) is
# the token right before the end -> licenser == recency, a confound. In SEPARATED the
# licenser sits early, several tokens from the end, so licenser attention can be
# distinguished from a generic recency bias.
COUNTRIES = [("France", " Paris"), ("England", " London"), ("Italy", " Rome"),
             ("Spain", " Madrid"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
             ("China", " Beijing"), ("Russia", " Moscow")]
# adjacent: licenser == recency (last content token) -> recency confound.
# separated: licenser == position 0 -> attention-SINK confound (no BOS is prepended).
# middle:    licenser is neither first nor last -> isolates a genuine semantic lock.
SETS = {
    "adjacent":  [(f"The capital of {c} is", a, c) for c, a in COUNTRIES],
    "separated": [(f"{c} is a country and its capital is", a, c) for c, a in COUNTRIES],
    "middle":    [(f"We recently learned that {c} has a capital called", a, c)
                  for c, a in COUNTRIES],
}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", torch_dtype=torch.float32).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
norm = model.model.norm
lm_head = model.lm_head


def one_id(s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]


bg_ids = [one_id(w) for w in BG]


def licenser_pos(input_ids, word):
    """Index of the prompt token carrying the licenser word (last matching)."""
    toks = [tok.decode([i]) for i in input_ids]
    hit = [j for j, t in enumerate(toks) if word.lower() in t.lower().strip()]
    return hit[-1] if hit else None


def run(prompt, answer, licenser):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, output_attentions=True)
    hs = out.hidden_states           # len n_layers+1; hs[l] = state after layer l-1
    attns = out.attentions           # len n_layers; [1, heads, seq, seq]
    n = len(attns)
    ans_id = one_id(answer)
    lp = licenser_pos(ids[0].tolist(), licenser)
    seq = ids.shape[1]
    prev = seq - 2                    # recency control (token before last)
    bos = 0                           # sink control

    rank, lic, other, recency, sink, disp = [], [], [], [], [], []
    top1_final = None
    for l in range(n):
        # logit lens. Intermediate hidden_states are PRE final-norm -> apply norm once.
        # hidden_states[-1] is already POST-norm, so the final layer uses the true
        # logits directly (applying norm again double-norms -> garbage; the trap).
        if l < n - 1:
            logits = lm_head(norm(hs[l + 1][0, -1, :])).detach()
        else:
            logits = out.logits[0, -1, :].detach()
        a = float(logits[ans_id])
        rank.append(int(sum(1 for b in bg_ids if float(logits[b]) > a)))
        if l == n - 1:
            top1_final = tok.decode([int(logits.argmax())])
        # attention: last-token row, mean over heads -> dist over key positions
        arow = attns[l][0, :, -1, :].mean(0).cpu().numpy()
        lic.append(float(arow[lp]) if lp is not None else float("nan"))
        recency.append(float(arow[prev]))
        sink.append(float(arow[bos]))
        # mean over prompt tokens that are neither licenser, prev, bos, nor self
        mask = np.ones(seq, bool)
        for k in (lp, prev, bos, seq - 1):
            if k is not None:
                mask[k] = False
        other.append(float(arow[mask].mean()) if mask.any() else float("nan"))
        # position: last-token residual displacement (frame-invariant). Skip the last
        # step: hs[-1] is post-norm, so hs[-1]-hs[-2] mixes normed/unnormed states.
        disp.append(float("nan") if l == n - 1
                    else float(torch.norm(hs[l + 1][0, -1] - hs[l][0, -1])))

    # L* = first layer from which the answer out-reads the whole panel persistently
    lstar = None
    if rank[-1] == 0:
        j = n - 1
        while j - 1 >= 0 and rank[j - 1] == 0:
            j -= 1
        lstar = j
    return dict(prompt=prompt, answer=answer, licenser=licenser, n=n, lstar=lstar,
                top1_final=top1_final, licenser_pos=lp, seq=seq,
                rank=rank, lic=lic, other=other, recency=recency, sink=sink, disp=disp)


def at(curve, idxs):
    return float(np.mean([curve[i][j] for i, j in enumerate(idxs)]))


for name, prompts in SETS.items():
    rows = [run(*p) for p in prompts]
    comm = [r for r in rows if r["lstar"] is not None]
    nlay = rows[0]["n"]
    lic = [r["lic"] for r in comm]
    rec = [r["recency"] for r in comm]
    oth = [r["other"] for r in comm]
    snk = [r["sink"] for r in comm]
    ls = [r["lstar"] for r in comm]
    peak = [int(np.nanargmax(r["lic"])) for r in comm]
    lp = [r["licenser_pos"] for r in comm]
    sq = [r["seq"] for r in comm]
    coincide = sum(1 for p, s in zip(lp, sq) if p == 0 or p == s - 2)
    print(f"\n=== SET: {name}  ({len(comm)}/{len(rows)} committed) ===")
    print(f"  predictions: {[r['top1_final'] for r in comm]}")
    print(f"  licenser positions {lp} of seq {sq}  "
          f"({coincide}/{len(comm)} coincide with pos0-sink or recency)")
    print(f"  commit L*        : mean {np.mean(ls):.1f}  (rel {np.mean(ls)/nlay:.2f})")
    print(f"  licenser peak lyr: mean {np.mean(peak):.1f}  (rel {np.mean(peak)/nlay:.2f})")
    print(f"  AT commit L* : licenser {at(lic,ls):.3f}  sink(pos0) {at(snk,ls):.3f}"
          f"  recency {at(rec,ls):.3f}  other {at(oth,ls):.3f}")
    print(f"  AT peak layer: licenser {at(lic,peak):.3f}  sink(pos0) {at(snk,peak):.3f}"
          f"  recency {at(rec,peak):.3f}  other {at(oth,peak):.3f}")
    # A genuine semantic lock: licenser attention high AND distinct from BOTH the
    # position-0 sink and the recency token. Only 'middle' can show this.
    denom = max(at(snk, peak), at(rec, peak), 1e-9)
    print(f"  licenser / max(sink,recency) at peak: {at(lic,peak)/denom:.2f}x")
    json.dump({"model": MODEL, "set": name, "background": BG, "rows": rows},
              open(f"docs/data/experiment-between-{name}-smollm2-135m.json", "w"))
