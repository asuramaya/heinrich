#!/usr/bin/env python3
"""Is the position-0 attention sink a null park, or the routing locus?

The between experiment found the last token dumps 0.6-0.9 of its attention on position 0.
Canonical test: an attention sink is a slot the model attends to in order to attend to
NOTHING — it carries a low-norm value, so despite huge attention weight it writes little.
Measure per layer: the last token's attention to the sink, the value-vector norm at the
sink vs content positions, and the sink's actual write magnitude (attn x |value|) vs
content's. If the sink's value is small and its write << content's despite huge
attention, it is a null park (routing happens via the small content attention), not the
routing locus.
"""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = ["We recently learned that France has a capital called",
           "We recently learned that Japan has a capital called",
           "The capital of Italy is", "Grass on a summer day is usually the color"]
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, attn_implementation="eager",
                                             torch_dtype=torch.float32).eval()
LY = model.model.layers
vcap = {}
for l, ly in enumerate(LY):
    ly.self_attn.v_proj.register_forward_hook(
        (lambda L: (lambda m, i, o: vcap.__setitem__(L, o[0].detach())))(l))

n = len(LY)
A0, VR, WR = [], [], []                         # sink attn; value ratio; write ratio — per layer, over prompts
for p in PROMPTS:
    vcap.clear()
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    a0, vr, wr = [], [], []
    for l in range(n):
        a = out.attentions[l][0, :, -1, :].mean(0)          # last-token attn over positions
        vn = vcap[l].norm(dim=-1)                           # value norm per position
        sink_a = float(a[0]); sink_v = float(vn[0])
        cont_v = float(vn[1:].mean()) if len(vn) > 1 else 1e-9
        sink_w = sink_a * sink_v
        cont_w = float((a[1:] * vn[1:]).sum()) if len(vn) > 1 else 1e-9
        a0.append(sink_a); vr.append(sink_v / (cont_v + 1e-9)); wr.append(sink_w / (cont_w + 1e-9))
    A0.append(a0); VR.append(vr); WR.append(wr)

A0, VR, WR = np.array(A0), np.array(VR), np.array(WR)
print(f"smollm2-135m — mean over {len(PROMPTS)} prompts, by depth:")
print(f"{'L':>3}{'relD':>6}{'sink_attn':>10}{'valnorm sink/content':>22}{'write sink/content':>20}")
for l in range(0, n, 3):
    print(f"{l:>3}{l/n:>6.2f}{A0[:,l].mean():>10.2f}{VR[:,l].mean():>22.2f}{WR[:,l].mean():>20.2f}")
print(f"\naverages:  sink_attn {A0.mean():.2f}   value sink/content {VR.mean():.2f}   "
      f"write sink/content {WR.mean():.2f}")
print("=> null park if: sink_attn high, value ratio < 1, write ratio < 1 "
      "(huge attention, little written — routing is via content, not the sink)")
