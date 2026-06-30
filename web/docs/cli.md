# CLI reference

Every command is `heinrich <verb>`. Analysis verbs that read a `.mri`/`.npz` need **no
model**; capture/eval/audit need weight access.

## Capture (the primary workflow)

```bash
heinrich mri --model X --mode raw --n-index 2000 --output X.mri    # single-mode capture
heinrich mri --model X.checkpoint.pt --output X.mri                # causal-bank MRI (auto-detects)
heinrich mri-scan --model X --output DIR                           # full workup: 3 modes + health + analysis
heinrich mri-backfill --model X --mri X.mri                        # fill missing weights/norms/embedding
heinrich mri-health --dir /Volumes/sharts                          # deep health (shapes, NaN, gates, attn)
heinrich mri-status --dir /Volumes/sharts                          # what's complete / incomplete / running
heinrich mri-verify --model X                                      # 5-token smoke test
heinrich mri-decompose --mri X.mri --n-components 0                # PCA + transposed indexes + precomputes
```

## MRI analysis (reads `.mri`, no model)

```bash
heinrich profile-layer-deltas --mri X.mri      # per-layer delta norms and amplification
heinrich profile-logit-lens   --mri X.mri      # per-layer predictions (the logit lens)
heinrich profile-gates        --mri X.mri      # MLP gate diversity, concentration, routing
heinrich profile-attention    --mri X.mri      # self vs prefix vs suffix attention
heinrich profile-pca-depth    --mri X.mri      # per-layer PCA structure
heinrich profile-pca-anatomy  --shrt S --frt F # name the unnamed PCA axes
```

## Profile pipeline (the lighter, no-full-capture path)

```bash
heinrich frt-profile   --tokenizer X                  # .frt: vocab, bytes/token, script detection
heinrich shart-profile --model X --n-index 3000       # .shrt: residual displacement vs silence
heinrich shart-profile --model X --n-index 500 --layers all   # all-layer sweep
heinrich sht-profile   --model X --n-index 3000       # .sht: output KL divergence vs silence
```

```bash
heinrich profile-chain    --frt F --shrt S --sht T    # three-stage correlation
heinrich profile-cross    --a S1 --b S2 --frt F        # two-model comparison
heinrich profile-survey   --shrt S1 S2 --frt F1 F2     # multi-model concordance
heinrich profile-mismatch --shrt S --frt F             # tokenizer–weight gap
heinrich profile-depth    --shrt S1 S2 --frt F1 F2     # layer trajectory (needs --layers all)
```

## Direction discovery (needs model + DB prompts)

```bash
heinrich profile-discover-direction --model X                       # safety direction → .npy
heinrich profile-safety-rank  --shrt X.shrt.npz --direction X.npy   # rank all tokens by safety
heinrich profile-first-token  --model X --direction X.npy           # first-token logit gap
heinrich profile-basin        --model X --direction X.npy --layer N # attractor map
```

::: warning Discover natively — never cross-map
Direction `.npy` files are per-model. Cross-mapped directions from the DB can be inverted
(Phi-3 Cyrillic was wrong). Always `discover-direction` for the specific model.
:::

## Causal-bank tools (reads `sequence.mri`, no model)

```bash
heinrich profile-cb-loss      --mri X.mri   # per-position loss decomposition
heinrich profile-cb-routing   --mri X.mri   # sequence-level expert routing
heinrich profile-cb-temporal  --mri X.mri   # temporal attention forensics
heinrich profile-cb-decompose --mri X.mri   # manifold decomposition (position/content/ghost)
heinrich profile-cb-causality --model X.checkpoint.pt   # finite-difference causality (needs model)
```

## Eval, publish, infrastructure

```bash
heinrich run     --model X --prompts harmbench --scorers word_match,qwen3guard
heinrich eval    --model X --prompts simple_safety --scorers word_match
heinrich audit   <model_id>                  # full behavioral security audit

heinrich publish --mri X.mri --bucket heinrich-mri    # lean subset → R2 (S3 API)
heinrich companion --mri-root DIR             # the local 3D viewer (the maximal node)
heinrich serve                                # MCP stdio server
heinrich db summary                           # database stats
```

The full command surface (40+ MCP tools, every flag) lives in the repo's
[`CLAUDE.md`](https://github.com/asuramaya/heinrich/blob/main/CLAUDE.md).
