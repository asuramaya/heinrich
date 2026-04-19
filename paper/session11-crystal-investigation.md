# The Crystal Token: 뀔 (Qwen-0.5B)

Investigation of why one token — 뀔 (U+B014 HANGUL SYLLABLE GGWIL) —
dominates PCA at every layer in raw/naked mode while its Korean syllable
neighbors in the vocabulary do not.

## What 뀔 is

- Vocab id: 144928 of 151643 (96th percentile of Qwen-0.5B's vocabulary)
- Script: Korean single-syllable BPE token
- UTF-8: 3 bytes (`eb 80 94`)
- Merge rank: 144928 (learned late, rare)

## Magnitudes don't explain the crystal

| Token | L0 norm | L3 norm | L5 norm |
|-------|---------|---------|---------|
| 뀔    |   6.92  |   20.91 |   32.01 |
| 씌    |   8.98  | 1025.15 | 1051.74 |
| 킵    |   6.55  | 1315.02 | 1341.50 |
| 뜯    |   7.22  | 1189.47 | 1215.99 |

Every other Korean syllable at the same vocab-rank zone blows up 100–200×
at L3. 뀔 grows only ~3×. If the crystal were "explosive growth", 뀔
would be the least crystal-like token, not the most.

## L3 MLP gate neurons

| Token | Top gate×up neurons | Magnitudes |
|-------|--------------------|-----------|
| 뀔    | [4500, 3687, 3499, 195, 430] | 4.1, 3.4, 2.7, 2.6, 2.5 |
| 씌    | [2247, 3016, 61, 3499, 4597] | 858, 585, 1.2, 1.2, 0.9 |
| 킵    | [2247, 3016, 61, 3499, 4597] | 876, 593, 1.2, 1.2, 0.9 |
| 뜯    | [2247, 3016, 3499, 61, 4597] | 867, 590, 1.2, 1.2, 0.9 |

The other Korean syllables share two dominant neurons (2247, 3016) that
fire at ~800–900× magnitude. 뀔 uniquely does NOT fire those neurons.
Its top neurons are small, distributed.

So the crystal is an **absence**: 뀔 is the token that L3 doesn't know
how to process with the Korean-rare-token circuit. It bypasses the
amplifier and ends up isolated in residual space.

## PC163 at L10: "isolated weird-unicode axis"

Top-10 tokens loading on PC163 at L10:

| Rank | Token | PC163 |
|------|-------|-------|
| 1    | 뀔    | 34.91 |
| 2    | ` ucwords`           | 10.00 |
| 3    | `;;;;;;;;;;;;;;;;`   | 5.21 |
| 4    | `)NULL`              | 4.42 |
| 5    | `^^^^`               | 3.89 |
| 6    | `!!!!!!!!`           | 3.59 |
| 7    | `oğ`                 | 3.22 |
| 8    | `ivityManager`       | 2.80 |
| 9    | `══`                 | 2.69 |
| 10   | ` \u200b\u200b`      | 2.56 |

The axis represents "isolated rare/code/repeat tokens seen with no
context". 뀔 is **3.5× farther on this axis than the next token** — it
is genuinely unique in latent geometry, not merely at the tail.

## Layer trajectory

Crystal is born at L3. Before L3 it is a normal low-z outlier.

| Layer | max \|z\| | dominant PC | share of z² |
|-------|-----------|-------------|-------------|
| L00   |   3.56    | PC219       | 0.012 |
| L01   |   4.76    | PC854       | 0.017 |
| L02   |   5.16    | PC0         | 0.017 |
| L03   |  86.16    | PC173       | 0.079 |
| L04   |  80.91    | PC167       | 0.069 |
| L05   |  93.33    | PC172       | 0.091 |
| L06   |  86.40    | PC166       | 0.078 |
| L07–L20 | 66–89   | PC163 (stable) | ~0.06 |
| L21–L23 | 42–58   | PC252/259/303  | ~0.02 |

The crystal axis is not one PC — it rotates slightly in early layers
and stabilises on PC163 by L7. The crystal decays in the last three
layers as the representation assembles for output.

## Template mode kills the crystal

| Mode     | 뀔 max \|z\| at L10 |
|----------|--------------------|
| raw      | 82.46 |
| naked    | 82.47 |
| template | 2.50  |

The baseline-stripping operation (raw mode) and the BOS-only baseline
(naked mode) both preserve the crystal. Adding any system-prompt
context destroys it: the token attends to the prefix, the MLP gets a
multi-signal input, and the unique-isolation handling is bypassed.
Matches the Session 6 finding that template mode prevents
crystallisation by giving attention a prefix to look at.

## Lmhead self-loop

Top-10 predictions at the final layer when the crystal is the input:

| Rank | Prediction | Logit | Note |
|------|-----------|-------|------|
| 1    | 뀔         | 1192.66 | itself |
| 2    | `contracts` | 1168.12 | |
| 3    | `jets`      | 1153.01 | |
| 4    | ` annum`    | 1134.44 | |
| 5    | ` GlobalKey`| 1092.44 | |
| 6    | `滚`        | 1092.08 | Chinese for "roll/scram" |
| …    |            |       | |

The model's #1 prediction when 뀔 is the only input is to produce 뀔
again. Self-loop: `뀔 → 뀔 → 뀔 …`.

The remaining predictions are also rare / code-like tokens. Consistent
with training data in which 뀔 only appeared in repetitive / corrupted
contexts — likely a webpage with the character repeated, or a broken
stream of bytes that landed in the pretraining set.

## Interpretation

The crystal is **not** a tokenizer bug or a training artifact in the
sense of being wrong. It is a learned behaviour:

1. The model has been told "when 뀔 appears alone, expect more 뀔".
2. The MLP amplification circuit (neurons 2247 + 3016 at L3) handles
   the common cases of rare Korean syllables — amplify and push toward
   a shared cluster for generic rare-token handling.
3. 뀔 bypasses that circuit because it has a self-predictive signal
   strong enough to route it differently.
4. Isolated, 뀔 sits alone in residual space at a unique geometric
   point. That point dominates PCA variance because no other token is
   near it.
5. In context, the self-loop is bypassed, the crystal vanishes.

The PCA dominance is a consequence of a rare, highly-specific learned
behaviour that the training data encoded, made visible only when the
model is fed exactly one token with nothing else to look at. It is a
fingerprint of the training corpus, not of the tokenizer or the
architecture.

## Implications for the microscope

- **Always exclude extreme-z outliers before searching PCs for
  "features".** A single pathological self-predictive token can
  hijack every other discovered direction in raw mode. Done in
  `_auto_discover_directions` (see Session 10 fix #36).
- **Don't interpret raw-mode crystals as a real feature of the model's
  normal processing.** They surface only when the isolation destroys
  context.
- **Template mode is the honest mode for concept discovery.** Raw and
  naked modes are useful for diagnosing rare-token anomalies but not
  for claims about how the model normally computes.
