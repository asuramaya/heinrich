"""Model profiling instruments.

  frt     → tokenizer atoms: raw bytes, merge ranks, decoded text
  shrt    → displacement from silence: vectors + KL at primary layer
  sht     → output distribution change (absorbed into shrt v0.3)
  trd     → per-head attribution via o_proj projection
  capture → total state: entry + exit at every layer, naked or template

  compare → all analysis: PCA, coherence, directions, safety rank, survey
  basin   → attractor mapping, first-token profiling, lm_head decomposition

The capture stores state. The analysis interprets it. Separate concerns.
"""
from __future__ import annotations
