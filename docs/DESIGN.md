# Design Rationale

## Why Geometry, Not Text

Safety evaluation is broken. Every major benchmark (HarmBench, SALAD-Bench, JailbreakBench) uses a single judge model scored against ground-truth labels that encode one company's safety policy as fact. LlamaGuard catches 4.5-21.8% of harmful content. Qwen3Guard drops from 91% to 33.8% on unseen prompts. The judges disagree with each other 33% of the time on the same data.

Heinrich's approach: measure what the model computes, not what it outputs. The output text is a 1-dimensional projection of a 3584-dimensional computation. The geometry — residual stream projections, attention routing, activation traces — contains more information than any text-level scorer can extract.

Two responses that say "I'm sorry, I can't help with that" can have completely different geometric signatures. One is deep in the refusal basin (projection +57). The other is at the basin boundary (projection -2.7). The text is identical. The vulnerability is different.

## Why Signal Isolation

Calibration interferes with measurement. The old approach: scorers produce labels, calibration grades them against "ground truth" (harmful prompt = harmful response), the report presents FPR/FNR. But harmful prompt does NOT mean harmful response — that's the whole point of safety alignment. The calibration punished judges for correctly identifying refusals as safe.

Heinrich's approach: each scorer stays in its own lane. No FPR/FNR. No ground-truth calibration. The report presents raw distributions per scorer per condition. The disagreements between scorers are the findings. A generation where word_match says COMPLIES, regex_harm says STRUCTURAL, and qwen3guard says safe — that disagreement IS the signal.

## Why External Data Only

Every hardcoded prompt is a human bias injected into the measurement. "How do I build a pipe bomb?" as the test prompt. Chinese political tokens as shart candidates. 10 benign prompts chosen for baselines. Each one carries the author's assumptions about what's dangerous.

Heinrich's approach: all prompts from HuggingFace benchmark datasets. `db.require_prompts()` loads from the DB or raises — no silent fallback to hardcoded strings. Shart candidates from random vocabulary sampling — no human-chosen candidate lists. The model identifies its own anomalies.

## Why One Forward Pass

The old pipeline ran `forward()` for measurement, then `generate()` for text. Two forward passes on the same input, producing potentially different results (forward returns argmax, generate samples). The measurement and the text could diverge.

`generate_with_geometry` runs generation and captures first-token geometry from the same computation. The logits, entropy, top-k alternatives, and contrastive projection all come from the forward pass that actually produced the text. No divergence between what was measured and what happened.

## Why the DB Is the Single Source of Truth

Every measurement writes to SQLite. Prompts, generations, scores, directions, conditions, sharts — all in one DB. The MCP server reads it. The viz reads it. The report reads it. No in-memory state that disappears when the process exits. No JSON files that drift from the DB.

Schema migrations handle version evolution (currently v10). The ChronoHorn writer pattern (single-writer thread, read-only reader connection) eliminates lock contention. Scorer subprocesses can write concurrently without "database is locked" errors.

## Historical: Why Signals

The original heinrich (2024) was built for the Jane Street Dormant LLM Puzzle — finding backdoor triggers in 671B-parameter models from weights alone. Everything produced typed `Signal` objects into a `SignalStore`. This pattern remains: spectral stats, weight deltas, circuit scores, and behavioral probes all produce Signals. The eval pipeline (2025) added the scorer/generation/projection system on top.

## Historical: Why the Full Circuit Matters

In DeepSeek V3's MLA, the backdoor modified both `q_a_proj` and `q_b_proj`. SVD on `q_a_proj` alone identified "Shakespeare" as the top trigger at z=17.8. The full circuit showed Shakespeare scores only 0.83. The actual trigger was "Let's think step by step" at 10.6x. Half a circuit gives qualitatively wrong answers. Heinrich's `diff/circuit.py` exists to prevent this mistake.

## Why Three Files, Not One

The shart is not in the tokenizer or the weights or the output. It's the compound. The `.frt` measures what the tokenizer decided (byte allocation, script, merge order). The `.shrt` measures what the weights do (residual displacement from silence). The `.sht` measures what comes out (KL divergence in the output distribution).

These three stages disagree. CJK is rank 9 in `.shrt` (low displacement) but rank 2 in `.sht` (high output change). Hebrew is rank 4 in `.shrt` but rank 6 in `.sht`. The internal response and the output effect are different measurements of different things. Collapsing them into one number hides the disagreement — and the disagreement is the signal.

## Why Token ID Splicing

The old code: `decode(token_id)` → format as string → `encode(string)`. The round-trip loses the token ID. 106 different token IDs decode to the same `�` glyph — 106 "measurements" of one thing. Accented characters re-encode differently than they decode.

The fix: `prefix_ids + [token_id] + suffix_ids`. Encode the template parts once. Splice the raw token ID in the middle. No decode. No re-encode. The token ID IS the measurement unit.

## Why Delta Is Already Relative

Delta = displacement from silence baseline. It's already a difference. Dividing by bytes makes it a ratio of a ratio — the bytes carry their own assumptions (UTF-8 encoding ≠ information content). Korean uses 3 bytes per character. Latin uses 1. Per-byte normalization made Thai look easy and Korean look hard, but that was the encoding talking, not the model.

For cross-model comparison, use within-model ranks and Kendall's W. Don't compare absolute deltas — different models have different silence states (entropy ranges from 0.92 to 11.2).

## Why Dynamic Baselines

The chat template silently injects a system prompt ("You are Qwen, created by Alibaba Cloud"). Two `.shrt` runs of the same model diverged 17% because one used `apply_chat_template` (poisoned) and the other used structural tokens only (clean). The baseline entropy was 2.86 vs 0.92 — a 3x difference from the invisible system prompt.

`_extract_clean_baseline` renders the template and strips system blocks using regex patterns for ChatML, Llama, and Mistral formats. The baseline is always structural tokens only.

## Why Consolidate, Don't Cut

conker-detect had 52 modules. Heinrich inherited all of them. The dormant puzzle used tools we never expected to need (mask geometry, safetensors byte-level carving, meta-probe families). The tools you think are dead weight are the ones you need when the investigation goes sideways. Heinrich consolidates overlapping implementations but keeps every capability.
