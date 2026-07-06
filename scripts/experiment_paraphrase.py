#!/usr/bin/env python3
"""Are the per-answer commit neurons per-FACT or per-surface-form?

The neuron experiment found the MLP answer-write is 'per-answer' (each capital recruits a
different neuron set). Is that per-FACT (same neurons however you ask France->Paris) or
per-SURFACE-FORM (the wording picks the neurons)? At a fixed commit-band layer, take the
top answer-neurons for several PHRASINGS of the same fact and measure within-fact overlap
against between-fact overlap (control). Within >> between => per-fact (a stable lookup);
within ~ between => wording-dependent, not a clean fact write.
"""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from heinrich.profile.readout import lens_logits

M = "HuggingFaceTB/SmolLM2-135M"
LAYER = 24                                              # fixed commit-band layer (rel 0.80)
TOPK = 10
FACTS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Spain", " Madrid")]
PHRASINGS = ["The capital of {c} is",
             "We recently learned that {c} has a capital called",
             "{c}'s capital city is named",
             "The city that serves as the capital of {c} is"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
LY = model.model.layers
NW = model.model.norm.weight.detach()
WU = model.lm_head.weight.detach()
z = {}
for l, ly in enumerate(LY):
    ly.mlp.down_proj.register_forward_pre_hook(
        (lambda L: (lambda m, i: z.__setitem__(L, i[0][0, -1, :].detach())))(l))


def top_neurons(prompt, answer):
    z.clear()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    lens_logits(out, model)                            # self-check the readout
    aid = tok.encode(answer, add_special_tokens=False)[0]
    Wd = LY[LAYER].mlp.down_proj.weight.detach()
    contrib = (z[LAYER] * ((WU[aid] * NW) @ Wd)).cpu().numpy()
    return set(int(x) for x in np.argsort(contrib)[::-1][:TOPK])


tops = {}                                              # (fact_i, phrasing_j) -> set
for fi, (c, a) in enumerate(FACTS):
    for pj, tmpl in enumerate(PHRASINGS):
        tops[(fi, pj)] = top_neurons(tmpl.format(c=c), a)


def ov(s, t): return len(s & t)


# within-fact: pairwise overlap across phrasings of the SAME fact
within = []
for fi in range(len(FACTS)):
    for j in range(len(PHRASINGS)):
        for k in range(j + 1, len(PHRASINGS)):
            within.append(ov(tops[(fi, j)], tops[(fi, k)]))
# between-fact: same phrasing index, DIFFERENT facts (control)
between = []
for pj in range(len(PHRASINGS)):
    for fi in range(len(FACTS)):
        for fk in range(fi + 1, len(FACTS)):
            between.append(ov(tops[(fi, pj)], tops[(fk, pj)]))

print(f"smollm2-135m  layer L{LAYER}  top-{TOPK} answer-neurons")
print(f"  WITHIN-fact overlap (same fact, different phrasings): {np.mean(within):.2f}/{TOPK}")
print(f"  BETWEEN-fact overlap (different facts, control):      {np.mean(between):.2f}/{TOPK}")
# per-fact stable core: neurons in ALL phrasings of a fact
for fi, (c, a) in enumerate(FACTS):
    core = set.intersection(*[tops[(fi, pj)] for pj in range(len(PHRASINGS))])
    print(f"  {c:>8}->{a:>8}: {len(core)}/{TOPK} neurons shared by ALL {len(PHRASINGS)} phrasings  {sorted(core)}")
verdict = ("PER-FACT (a stable lookup survives rewording)"
           if np.mean(within) > 2 * max(np.mean(between), 0.5)
           else "wording-dependent (no stable per-fact core)")
print(f"\n=> {verdict}")
