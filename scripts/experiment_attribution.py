#!/usr/bin/env python3
"""Experiment: Attention or MLP? — attributing the ~3/4 readout commit.

The homing commit is a *readout rotation*: the state comes to project onto the
answer's output direction, late, at ~3/4 depth. The "between" experiment showed it is
NOT carried by attention-to-content (that was recency + the position-0 sink). So which
sublayer writes the rotation? The residual stream is additive --
``h <- h + attn_out + mlp_out`` -- so we attribute the answer's readout, per layer, to
attention vs MLP by direct logit attribution: project each sublayer's LAST-token output
onto the answer's unembedding direction (folded through the final RMSNorm weight). This
avoids every attention-weight confound: it reads the sublayer *output*, not which token
it attended to.

Controls (tonight's lesson): a null answer (unrelated token) must NOT be driven at the
commit; L* comes from the blessed self-checking lens, not a hand-rolled one.

Reproduce: python scripts/experiment_attribution.py
Writes docs/data/experiment-attribution-smollm2-135m.json
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
NULL = " bicycle"                                   # unrelated null answer (control)
COUNTRIES = [("France", " Paris"), ("England", " London"), ("Italy", " Rome"),
             ("Spain", " Madrid"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
             ("China", " Beijing"), ("Russia", " Moscow")]
PROMPTS = [(f"We recently learned that {c} has a capital called", a)
           for c, a in COUNTRIES]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
NORM_W = model.model.norm.weight.detach()           # final RMSNorm gain [D]
WU = model.lm_head.weight.detach()                  # unembedding [vocab, D]
LAYERS = model.model.layers


def one_id(s): return tok.encode(s, add_special_tokens=False)[0]


bg_ids = [one_id(w) for w in BG]

# hooks: capture each layer's self_attn and mlp LAST-token output (the additive
# contributions to the residual stream).
cap = {}


def mk(l, kind):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cap[(l, kind)] = o[0, -1, :].detach()
    return hook


for l, layer in enumerate(LAYERS):
    layer.self_attn.register_forward_hook(mk(l, "attn"))
    layer.mlp.register_forward_hook(mk(l, "mlp"))


def dla_dir(tid):
    return WU[tid] * NORM_W                          # direct-logit-attribution direction


def run(prompt, answer):
    cap.clear()
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    lens = lens_logits(out, model)                  # [n, vocab], self-checked
    n = len(LAYERS)
    ans_id, null_id = one_id(answer), one_id(NULL)
    ua, un = dla_dir(ans_id), dla_dir(null_id)
    rank = [int(sum(1 for b in bg_ids if float(lens[l][b]) > float(lens[l][ans_id])))
            for l in range(n)]
    lstar = None
    if rank[-1] == 0:
        j = n - 1
        while j - 1 >= 0 and rank[j - 1] == 0:
            j -= 1
        lstar = j
    A = lambda kind, u: [float(cap[(l, kind)] @ u) for l in range(n)]
    return dict(prompt=prompt, answer=answer, n=n, lstar=lstar,
                top1_final=tok.decode([int(lens[-1].argmax())]), rank=rank,
                attn_ans=A("attn", ua), mlp_ans=A("mlp", ua),
                attn_null=A("attn", un), mlp_null=A("mlp", un))


rows = [run(*p) for p in PROMPTS]
comm = [r for r in rows if r["lstar"] is not None]
n = rows[0]["n"]


def stk(k): return np.array([r[k] for r in comm], float)


aa, ma = stk("attn_ans"), stk("mlp_ans")
an, mn = stk("attn_null"), stk("mlp_null")
ls = [r["lstar"] for r in comm]

print(f"predictions: {[r['top1_final'] for r in comm]}")
print(f"committed {len(comm)}/{len(rows)}   mean L* {np.mean(ls):.1f} (rel {np.mean(ls)/n:.2f})")
print("\ndirect logit attribution to the ANSWER, mean over prompts, by depth:")
print(f"{'L':>3}{'relD':>6}{'attn->ans':>10}{'mlp->ans':>9}{'attn->null':>11}{'mlp->null':>10}")
for l in range(0, n, 2):
    print(f"{l:>3}{l/n:>6.2f}{aa[:,l].mean():>10.2f}{ma[:,l].mean():>9.2f}"
          f"{an[:,l].mean():>11.2f}{mn[:,l].mean():>10.2f}")


def at(c, idx): return float(np.mean([c[i][j] for i, j in enumerate(idx)]))


print(f"\n=== at the commit L* ===")
print(f"  attn -> answer: {at(aa,ls):+.2f}     mlp -> answer: {at(ma,ls):+.2f}")
print(f"  attn -> null:   {at(an,ls):+.2f}     mlp -> null:   {at(mn,ls):+.2f}   (control: ~0)")
apk = [int(np.argmax(aa[i])) for i in range(len(comm))]
mpk = [int(np.argmax(ma[i])) for i in range(len(comm))]
print(f"  peak attn->ans layer: mean {np.mean(apk):.1f} (rel {np.mean(apk)/n:.2f})")
print(f"  peak mlp ->ans layer: mean {np.mean(mpk):.1f} (rel {np.mean(mpk)/n:.2f})")
drv = "MLP" if at(ma, ls) > at(aa, ls) else "attention"
print(f"\n=> at the commit, the answer readout is driven by: {drv}")

json.dump({"model": MODEL, "null": NULL, "background": BG, "rows": rows},
          open("docs/data/experiment-attribution-smollm2-135m.json", "w"))
print("wrote docs/data/experiment-attribution-smollm2-135m.json")
