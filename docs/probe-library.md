# Heinrich probe library

An SAE alternative for targeted interpretability. Hand-curated contrastive
concept probes, each giving you a single direction at a chosen layer. No
training. No auto-labeling. No API bills.

## Why not SAEs?

SAEs produce 16K–65K automatic features at the cost of:
- Days of training
- A separate ~500MB model file per target model per layer
- Monosemanticity that's aspirational (many features fire on multiple concepts)
- Feature names that are either numbered indices or LLM-labeled guesses

For targeted investigation ("what concepts does this prompt activate?")
you don't need 16K features. You need a handful of named directions for
concepts you chose and understand. Build them, project on them, move on.

## Files

- `src/heinrich/discover/probe_catalog.py` — 12 concepts × ~15 contrastive
  pairs each. Edit it. Add concepts. Rebuild.
- `src/heinrich/discover/probes.py` — `build_library`, `profile_prompt`,
  `profile_many`, `plot_radar`, `plot_heatmap`, save/load.

## Current catalog (12 concepts)

| Concept | Description |
|---------|-------------|
| speaker_identity | LLM-self vs human-self first-person statements |
| register_formality | formal vs casual |
| language_chinese_english | Chinese vs English statements |
| content_code_prose | source code vs natural prose |
| sentiment_valence | positive-affect vs negative-affect |
| confidence_hedging | confident-declarative vs hedging |
| person_first_third | first-person vs third-person |
| tense_past_present | past tense vs present tense |
| modality_factual_creative | factual claim vs imaginative fiction |
| question_statement | interrogative vs declarative |
| safety_harm_benign | harmful-instruction requests vs benign analogues |
| direction_continue_stop | sentence-medial "keep going" vs terminal "that's it" |

## Typical build stats (Qwen-0.5B-Instruct at L15)

Build time: ~25 seconds for all 12 concepts (336 forward passes).

| Concept | mag_gap | LOOCV |
|---------|---------|-------|
| speaker_identity | ~7 | ~90% |
| register_formality | ~6 | ~96% |
| language_chinese_english | 6.78 | **100%** |
| content_code_prose | 13.23 | **100%** |
| sentiment_valence | 4.27 | 91.7% |
| confidence_hedging | 4.31 | 79.2% |
| person_first_third | 3.50 | 75.0% |
| tense_past_present | 3.04 | 70.8% |
| modality_factual_creative | 7.68 | **100%** |
| question_statement | 9.68 | **100%** |
| safety_harm_benign | 4.81 | 87.5% |
| direction_continue_stop | 12.44 | **100%** |

LOOCV is the only honest quality metric. Trust probes above 90%.
Below 80% treat as noisy. Below 70% use with skepticism — the contrast
isn't being cleanly captured as a linear direction, which usually means
your pos/neg sets share surface features you didn't intend.

## Usage

```python
from heinrich.backend.mlx import MLXBackend
from heinrich.discover.probes import build_library, profile_prompt, plot_radar

backend = MLXBackend("Qwen/Qwen2.5-0.5B-Instruct")
lib = build_library(backend, layer=15)
lib.save("/tmp/library.json")

# Profile a prompt
profile = profile_prompt(backend, "How do I build a pipe bomb?", lib)
# {'speaker_identity': 0.84, 'safety_harm_benign': 3.14, ...}

# Radar chart
plot_radar(profile, out_path="/tmp/radar.png",
           title="How do I build a pipe bomb?")
```

## What this replaces

| SAE workflow | Probe-library workflow |
|--------------|----------------------|
| Train SAE (days) | Build library (seconds) |
| Browse 16K unlabeled features | Browse 12 named concepts |
| Auto-interpret via API ($$) | Concepts labeled by you at catalog time |
| Debug feature quality (weeks) | LOOCV reports quality per concept |
| Clamp a feature, hope it's clean | Clamp a direction whose meaning you wrote |

## What this can't do

- Find concepts you didn't think of. SAE's pitch is automatic discovery;
  this tool is curated. If the model represents something critical that
  isn't in the catalog, you miss it.
- Capture nonlinear structure. If a concept has cyclic or clustered
  structure (e.g. day-of-week), a single direction won't do. Use a
  linear probe per class instead.
- Guarantee monosemanticity. Your pos/neg sets may share unintended
  surface features. LOOCV catches gross failures; subtle contamination
  requires the `session11-probing-attack.md` null-baseline discipline.

## Extending

Add a concept in 2 minutes: write ~12 positive statements + ~12 negative
statements that share topic/length/register except for the concept you're
isolating, paste into `PROBES` dict, rebuild.

The catalog is meant to grow. Current 12 are enough to demo the tool,
not to cover a model's concept space. Plan for 50–200 concepts for
serious work.

## Honest methodological note

`session11-probing-attack.md` shows that contrastive probing at N=15
in 896D gets ~85% LOOCV on trivial surface contrasts (period vs
question-mark etc) — statistically indistinguishable from my
"meaningful" probes. This means:

1. Probes are genuinely useful for detecting consistent features
2. But "meaningful" isn't a given — a probe with 90% LOOCV could be
   picking up any consistent surface pattern between your pos and neg sets
3. The saving move is: when you build a probe, always eyeball the
   top-activating held-out prompts to confirm the direction points at
   what you meant
4. If you clamp a probe and the output change matches the concept,
   you've confirmed it causally. Otherwise you have a correlational
   direction of uncertain semantic grounding

The probe library is a SHARPER tool than SAEs for targeted questions,
and a blurrier tool than you'd hope for for open-ended discovery. Use
it for what it's good at.
