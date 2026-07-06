#!/usr/bin/env python3
"""Experiment: is the commit's MLP write localized or distributed?

The attribution experiment showed the ~3/4 readout commit is an MLP-led sublayer WRITE.
An MLP output is a sum over intermediate neurons: neuron i contributes
``z_i * (down_proj[:, i] . answer_dir)`` to the answer readout, where z_i is its gated
activation. So at the commit layer we can rank neurons by their contribution and ask
how concentrated the write is — a handful (a legible "Paris neuron" lookup) or hundreds
(distributed). Controls: a null answer (bicycle), and cross-prompt overlap of the top
neurons (a generic capital-slot lookup shares neurons; per-answer lookups do not).

Reproduce: python scripts/experiment_neurons.py
Writes docs/data/experiment-neurons-smollm2-135m.json
"""
from __future__ import annotations
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from heinrich.profile.readout import lens_logits

MODEL = "HuggingFaceTB/SmolLM2-135M"
BG = [" table", " music", " river", " doctor", " happy", " window", " garden",
      " metal", " quiet", " horse", " paper", " coffee", " stone", " engine",
      " lucky", " sugar"]
NULL = " bicycle"
COUNTRIES = [("France", " Paris"), ("England", " London"), ("Italy", " Rome"),
             ("Spain", " Madrid"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
             ("China", " Beijing"), ("Russia", " Moscow")]
PROMPTS = [(f"We recently learned that {c} has a capital called", a)
           for c, a in COUNTRIES]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
NORM_W = model.model.norm.weight.detach()
WU = model.lm_head.weight.detach()
LAYERS = model.model.layers
INTER = LAYERS[0].mlp.down_proj.weight.shape[1]

zcap = {}                                            # layer -> down_proj input (last tok)


def mk(l):
    def pre(mod, inp):
        zcap[l] = inp[0][0, -1, :].detach()
    return pre


for l, layer in enumerate(LAYERS):
    layer.mlp.down_proj.register_forward_pre_hook(mk(l))


def one_id(s): return tok.encode(s, add_special_tokens=False)[0]


bg_ids = [one_id(w) for w in BG]


def run(prompt, answer):
    zcap.clear()
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    lens = lens_logits(out, model)
    n = len(LAYERS)
    ans_id = one_id(answer)
    rank = [int(sum(1 for b in bg_ids if float(lens[l][b]) > float(lens[l][ans_id])))
            for l in range(n)]
    lstar = None
    if rank[-1] == 0:
        j = n - 1
        while j - 1 >= 0 and rank[j - 1] == 0:
            j -= 1
        lstar = j
    if lstar is None:
        return dict(prompt=prompt, answer=answer, lstar=None)
    # neuron contributions to the answer at the commit layer
    u_ans = WU[ans_id] * NORM_W
    u_null = WU[one_id(NULL)] * NORM_W
    Wd = LAYERS[lstar].mlp.down_proj.weight.detach()          # [hidden, inter]
    z = zcap[lstar]                                           # [inter]
    contrib = (z * (u_ans @ Wd)).cpu().numpy()               # per-neuron -> answer
    contrib_null = (z * (u_null @ Wd)).cpu().numpy()         # per-neuron -> null (same z)
    return dict(prompt=prompt, answer=answer, lstar=int(lstar),
                contrib=contrib.tolist(), contrib_null=contrib_null.tolist())


rows = [run(*p) for p in PROMPTS]
comm = [r for r in rows if r["lstar"] is not None]


def concentration(c):
    c = np.asarray(c)
    pos = c[c > 0]
    tot = pos.sum()
    srt = np.sort(pos)[::-1]
    pr = (pos.sum() ** 2) / (np.square(pos).sum() + 1e-9)     # effective # of + neurons
    cum = np.cumsum(srt) / (tot + 1e-9)
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    top10 = float(srt[:10].sum() / (tot + 1e-9))
    return pr, n80, top10, tot


print(f"committed {len(comm)}/{len(rows)}   intermediate size = {INTER}")
print(f"\n{'answer':>9}{'L*':>4}{'eff#neur':>9}{'#to80%':>8}{'top10frac':>10}{'top neurons (to answer)':>28}")
top_sets = []
for r in comm:
    pr, n80, top10, tot = concentration(r["contrib"])
    top = list(np.argsort(r["contrib"])[::-1][:5])
    top_sets.append(set(int(x) for x in np.argsort(r["contrib"])[::-1][:10]))
    print(f"{r['answer']:>9}{r['lstar']:>4}{pr:>9.1f}{n80:>8}{top10:>10.2f}   {top}")

# cross-prompt overlap of the top-10 answer neurons (generic slot vs per-answer)
shared = set.intersection(*top_sets) if top_sets else set()
alln = set().union(*top_sets)
print(f"\n=== cross-prompt (8 answers) ===")
print(f"  top-10 neurons: {len(alln)} distinct across prompts; {len(shared)} shared by ALL 8")
# null specificity: do the answer's top neurons also drive the null?
spec = []
for r in comm:
    top = np.argsort(r["contrib"])[::-1][:10]
    a = float(np.asarray(r["contrib"])[top].sum())
    nl = float(np.asarray(r["contrib_null"])[top].sum())
    spec.append((a, nl))
ma = np.mean([a for a, _ in spec]); mn = np.mean([n for _, n in spec])
print(f"  top-10 answer-neurons: contribution to ANSWER {ma:+.1f}  vs to NULL {mn:+.1f}")
pr_mean = np.mean([concentration(r["contrib"])[0] for r in comm])
print(f"\n=> effective ~{pr_mean:.0f} of {INTER} neurons carry the write "
      f"({'LOCALIZED' if pr_mean < INTER*0.1 else 'DISTRIBUTED'}); "
      f"{len(shared)}/10 shared across all answers "
      f"({'generic slot' if len(shared)>=3 else 'largely per-answer'})")

json.dump({"model": MODEL, "null": NULL, "intermediate": INTER,
           "rows": [{k: v for k, v in r.items() if k != 'contrib_null'} for r in rows]},
          open("docs/data/experiment-neurons-smollm2-135m.json", "w"))
print("wrote docs/data/experiment-neurons-smollm2-135m.json")
