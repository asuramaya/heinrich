---
layout: home
hero:
  name: Heinrich
  text: A model-forensics instrument
  tagline: It measures what a language model computes — the internal residual geometry — not what it says. This is the documentation for the instrument. For the art, enter the Observatory.
  actions:
    - theme: brand
      text: Install & quick start
      link: /guide
    - theme: alt
      text: GitHub →
      link: https://github.com/asuramaya/heinrich
    - theme: alt
      text: Enter the Observatory →
      link: https://hcirnieh.com/observatory
features:
  - title: 1 · Capture
    details: MRI any model with weight access (MLX, HuggingFace, or causal-bank backends). Full-vocabulary residual state, every layer, every measurement in its own lane. No hardcoded prompts — all data from HF benchmarks.
    link: /guide
    linkText: The pipeline
  - title: 2 · Decompose & analyse
    details: PCA the residual stream, build O(1) transposed indexes, and run the model-free profile-* family — directions, gates, attention, neuron fields, the logit lens. Reads a recording; needs no model.
    link: /cli
    linkText: CLI reference
  - title: 3 · Publish to the edge
    details: heinrich publish ships a lean consumer subset to R2 (the .mri is the contract). A thin Worker serves it with zero paid compute, and the same SPA renders it in any browser.
    link: /architecture
    linkText: Architecture
---

## What it is

Heinrich captures a model's internal geometry — residual-stream projections, attention
patterns, activation traces — alongside language-level signals, and presents them as an
**isolated signal stack**: each measurement in its own lane, no ground-truth calibration to
round off the disagreements, interpretation left to the reader.

It runs on any model with weight access. It produces `.mri` artifacts. The
[Observatory](https://hcirnieh.com/observatory) is one consumer of those artifacts — a
browser you can fly a real model's residual stream through. This documentation is the other
half: how the instrument works, how to point it at your own model, and what it found.

## The two halves

| | Producer | Consumer |
| --- | --- | --- |
| **Runs** | where the models are (GPU + weights) | anywhere — including a browser at the edge |
| **Does** | `mri` · `decompose` · `eval` · `audit` · the MCP suite | the viewer + the model-free `profile-*` analyses |
| **You install** | `pip install heinrich` | nothing — open [hcirnieh.com](https://hcirnieh.com/) |

The `.mri` artifact is the API between them. You don't port the producer to the edge — you
give it a publish target. → [Architecture](/architecture)

## Three principles

> **Measurement and calibration interfere.** There is no FPR/FNR calibration. Each scorer
> produces raw labels; the report presents distributions and disagreements. The disagreement
> *is* the signal.

> **One forward pass, not two.** `generate_with_geometry` captures text *and* first-token
> geometry from the same computation.

> **The DB is the single source of truth.** All prompts from HF benchmarks via
> `require_prompts()` — no hardcoded fallbacks. All directions, sharts, generations, and
> scores in SQLite with schema migrations.

## Start here

- New to it → [**Install & quick start**](/guide)
- Want the command surface → [**CLI reference**](/cli) · [**MCP tools**](/mcp)
- Want the format / the edge → [**The `.mri` artifact**](/artifact) · [**Architecture**](/architecture)
- Want the results → [**Findings**](/findings)
- Want the whole strange story → the [**Grand Unified Theory of Sharts**](https://hcirnieh.com/book) (the book — rendered from the live `.tex`)
