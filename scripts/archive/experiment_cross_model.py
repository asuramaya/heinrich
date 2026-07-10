#!/usr/bin/env python3
"""Cross-model: does the MLP-led 3/4 commit + diffuse write generalize?

For three models/families, find the commit L* (self-checking lens), then at L* report
attention-vs-MLP direct logit attribution to the answer (and a null), and the effective
number of neurons carrying the MLP write (participation ratio). Tests whether 'MLP-led
commit at ~3/4 depth, diffuse per-answer write' holds beyond smollm2-135m.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from heinrich.profile.readout import lens_logits

BG = [" table", " music", " river", " doctor", " happy", " window", " garden", " metal",
      " quiet", " horse", " paper", " coffee", " stone", " engine", " lucky", " sugar"]
NULL = " bicycle"
COUNTRIES = [("France", " Paris"), ("England", " London"), ("Italy", " Rome"),
             ("Spain", " Madrid"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
             ("China", " Beijing"), ("Russia", " Moscow")]
PROMPTS = [(f"We recently learned that {c} has a capital called", a) for c, a in COUNTRIES]
MODELS = [("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
          ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
          ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct")]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def pr(v):
    e = v ** 2
    return float((e.sum() ** 2) / ((e ** 2).sum() + 1e-12))


def analyze(name, hf):
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32).eval().to(dev)
    NW = model.model.norm.weight.detach()
    WU = model.lm_head.weight.detach()
    LY = model.model.layers
    cap = {}

    def mk(l, k):
        def h(m, i, o):
            cap[(l, k)] = (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach()
        return h

    def mkz(l):
        def pre(m, i):
            cap[(l, "z")] = i[0][0, -1, :].detach()
        return pre

    for l, ly in enumerate(LY):
        ly.self_attn.register_forward_hook(mk(l, "attn"))
        ly.mlp.register_forward_hook(mk(l, "mlp"))
        ly.mlp.down_proj.register_forward_pre_hook(mkz(l))

    def oid(s): return tok.encode(s, add_special_tokens=False)[0]
    bg = [oid(w) for w in BG]

    aa, ma, an, mn, prs, rel = [], [], [], [], [], []
    for prompt, answer in PROMPTS:
        cap.clear()
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        lens = lens_logits(out, model)
        n = len(LY)
        aid = oid(answer)
        rank = [int(sum(1 for b in bg if float(lens[l][b]) > float(lens[l][aid]))) for l in range(n)]
        if rank[-1] != 0:
            continue
        j = n - 1
        while j - 1 >= 0 and rank[j - 1] == 0:
            j -= 1
        L = j
        rel.append(L / n)
        ua = WU[aid] * NW
        un = WU[oid(NULL)] * NW
        Wd = LY[L].mlp.down_proj.weight.detach()
        z = cap[(L, "z")]
        aa.append(float(cap[(L, "attn")] @ ua)); ma.append(float(cap[(L, "mlp")] @ ua))
        an.append(float(cap[(L, "attn")] @ un)); mn.append(float(cap[(L, "mlp")] @ un))
        prs.append(pr((z * (ua @ Wd)).cpu().numpy()))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return dict(name=name, n_layers=n, inter=LY[0].mlp.down_proj.weight.shape[1],
                committed=len(rel), rel_L=float(np.mean(rel)) if rel else None,
                attn_ans=float(np.mean(aa)), mlp_ans=float(np.mean(ma)),
                attn_null=float(np.mean(an)), mlp_null=float(np.mean(mn)),
                eff_neurons=float(np.mean(prs)) if prs else None)


rows = [analyze(n, hf) for n, hf in MODELS]
print(f"\n{'model':16}{'nL':>4}{'inter':>6}{'commit':>8}{'relL':>6}"
      f"{'attn>ans':>9}{'mlp>ans':>9}{'mlp>null':>9}{'effN':>7}")
for r in rows:
    print(f"{r['name']:16}{r['n_layers']:>4}{r['inter']:>6}{r['committed']:>5}/8"
          f"{(r['rel_L'] or 0):>6.2f}{r['attn_ans']:>9.1f}{r['mlp_ans']:>9.1f}"
          f"{r['mlp_null']:>9.1f}{(r['eff_neurons'] or 0):>7.0f}")
print("\n=> MLP-led (mlp>ans exceeds attn>ans) and diffuse (effN >> a handful) across models?")
json.dump(rows, open("docs/data/experiment-cross-model.json", "w"))
print("wrote docs/data/experiment-cross-model.json")
