# Heinrich User Guide

## The 60-Second Version

Heinrich measures what language models compute internally, not just what they output. Load a model, load benchmark prompts from HuggingFace, and the tool maps the model's safety geometry — where it refuses, where it complies, how hard each category is to steer, and where independent scorers disagree.

```bash
pip install -e ".[dev,fetch]"
pip install mlx mlx-lm                # Apple Silicon

heinrich run --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --prompts simple_safety --scorers word_match,regex_harm,qwen3guard

heinrich viz                           # http://localhost:8377
```

---

## Core Concepts

### The Signal Stack

Heinrich runs multiple independent scorers on the same generations. Each scorer produces raw labels — no scorer knows what the others said. The stack:

| Scorer | What it sees | Example label |
|--------|-------------|---------------|
| word_match | refusal/compliance vocabulary | REFUSES, COMPLIES |
| regex_harm | structural patterns (steps, code, chemicals) | STRUCTURAL, PLAIN |
| refusal | first-token probability distribution | refuse_prob=0.95 |
| self_kl | behavioral divergence from clean distribution | first_token_prob=0.82 |
| qwen3guard | Alibaba's safety model judgment | qwen3guard:safe |
| llamaguard | Meta's safety model judgment | llamaguard:unsafe |

No scorer is "right." The disagreements are the findings. When word_match says COMPLIES, regex_harm says STRUCTURAL, and qwen3guard says safe — that generation has structural harm content that the judge missed.

### Geometry Capture

Every generation captures pre-linguistic signals from the same forward pass that produced the text:

- **Contrastive projection** — where the model's residual stream sits relative to the harmful/benign boundary
- **Logit entropy** — how certain the model is about its first token
- **Top-k alternatives** — what the model almost said (a refusal with "Sure" at p=0.35 in second place is different from one where "Sure" is at p=0.001)

These signals exist before the model chose its words. No judge model can see them. No prompt framing can bias them.

### The DB

Everything lives in SQLite (`data/heinrich.db`). Prompts from HF benchmarks. Generations with geometry columns. Scores from each scorer. Contrastive directions per layer. Steering conditions. The MCP server, the viz, and the report all read from the same DB.

`db.require_prompts()` loads prompts or raises if the DB is empty. No hardcoded fallbacks. If the DB has no data, the tool tells you to load benchmarks instead of silently running on 3 human-chosen strings.

---

## Walkthrough: Full Pipeline

### 1. Load benchmark prompts

```python
from heinrich.cartography.datasets import load_dataset
from heinrich.core.db import SignalDB

db = SignalDB()
for name in ['simple_safety', 'catqa', 'do_not_answer', 'forbidden_questions', 'toxicchat']:
    prompts = load_dataset(name)
    for p in prompts:
        db.record_prompt(p['prompt'], name, p.get('category'),
                        is_benign=(name == 'toxicchat' and p.get('category') == '0'))
db.close()
```

This loads ~7,000 prompts from 5 HF datasets, 35 categories.

### 2. Run the full pipeline

```bash
heinrich run --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --prompts simple_safety --scorers word_match,regex_harm,qwen3guard
```

This:
1. Loads the model once
2. Discovers contrastive directions at every layer (harmful vs benign residual separation)
3. Scans for safety cliffs (steering magnitude where behavior flips)
4. Generates responses under all conditions (clean + steered)
5. Scores with each scorer
6. Builds the report

### 3. Explore the results

```bash
heinrich viz    # http://localhost:8377
```

The visualizer shows:
- Basin scatter (refuse_prob vs first_token_prob, colored by category)
- Scorer distributions per condition
- Signal stack table (all scorer labels per generation)
- Direction stability sparklines
- Judge disagreements

### 4. Query programmatically

```python
from heinrich.mcp import ToolServer
ts = ToolServer()

# Full report
report = ts.call_tool('heinrich_eval_report', {})

# Scorer distributions
dists = ts.call_tool('heinrich_eval_calibration', {})

# Raw SQL
rows = ts.call_tool('heinrich_sql', {'sql': 'SELECT condition, COUNT(*) FROM generations GROUP BY condition'})
```

---

## Walkthrough: Mapping Basin Geometry

The contrastive direction separates harmful from benign on one axis. PCA on the residual states reveals the full structure.

```python
from heinrich.backend.mlx import MLXBackend
from heinrich.cartography.templates import build_prompt
from heinrich.core.db import SignalDB
import numpy as np

be = MLXBackend('mlx-community/Qwen2.5-7B-Instruct-4bit')
db = SignalDB()
prompts = db.get_prompts(limit=300)
db.close()

formatted = [build_prompt(p['text'], model_config=be.config) for p in prompts]
states = be.capture_residual_states(formatted, layers=[be.config.safety_layers[-1]])
residuals = states[be.config.safety_layers[-1]]

# PCA — geometry decides the structure
centered = residuals - residuals.mean(axis=0)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
proj = U[:, :3] * S[:3]
# proj[:, 0] = PC1 (harmful vs benign), proj[:, 1] = PC2 (sub-categories)
```

Findings on Qwen 7B: 4-5 basins, not 2. PC1 (20%) separates harmful from benign. PC2-PC3 separate discrimination from violence from illegal activity. The safety mechanism is low-rank — a rank-8 LoRA targets the shared axis.

---

## Walkthrough: Shart Detection

A shart is any token that steals compute disproportionate to its relevance. Heinrich finds them by random vocabulary sampling — no hardcoded candidate list.

```python
# Random sample of 300 real tokens
vocab_size = be.tokenizer.vocab_size
real_tokens = [(tid, be.tokenizer.decode([tid])) for tid in range(vocab_size)
               if be.tokenizer.decode([tid]).strip() and len(be.tokenizer.decode([tid])) > 1]
sample = rng.choice(len(real_tokens), 300, replace=False)

# Measure causal effect: prepend to benign query, measure projection shift
baseline = be.forward(build_prompt("What is the weather?", model_config=cfg),
                      return_residual=True, residual_layer=14)
baseline_proj = np.dot(baseline.residual, safety_dir)

for tid, token in [real_tokens[i] for i in sample]:
    prepended = build_prompt(f"{token} What is the weather?", model_config=cfg)
    fwd = be.forward(prepended, return_residual=True, residual_layer=14)
    shift = np.dot(fwd.residual, safety_dir) - baseline_proj
    # Tokens with |shift| > threshold are sharts
```

Findings: 4.3% of Mistral's vocabulary shifts the safety projection by >8% of range. "toxic" has zero effect. "encourage" has 90x more.

---

## Walkthrough: Ghost Shart Measurement

The ghost shart: does conversation history affect the model's safety computation?

```python
# Fixed prefix (20-turn conversation), vary the final turn
prefix = [...20 turns of user/assistant...]
final_turns = db.get_prompts(limit=50)  # random from HF benchmarks

for fp in final_turns:
    turns = list(prefix) + [("user", fp["text"])]
    ctx = tokenizer.apply_chat_template(turns, ...)
    proj = be.forward(ctx, return_residual=True, residual_layer=14)
    # Measure: how much does the projection vary with the final turn vs the prefix?
```

Finding: 99% of safety projection variance comes from the final turn, 0% from the prefix, at all context lengths tested (135 to 1500 tokens). The model's safety is a recency function. The ghost shart doesn't accumulate.

But: the model reads the prefix (77-86% of attention budget at 1500 tokens). It attends to prior refusals 4x more than equivalent benign turns. The attention channel is open. The safety projection doesn't use it.

---

## MCP Tools

40+ tools. Key ones:

| Tool | What it does |
|------|-------------|
| `heinrich_eval_run` | Full pipeline: discover + attack + eval + report |
| `heinrich_eval_report` | Build report from DB data |
| `heinrich_eval_calibration` | Per-scorer signal distributions (no FPR/FNR) |
| `heinrich_eval_disagreements` | Where judge scorers disagree |
| `heinrich_eval_scores` | Query the score matrix with filters |
| `heinrich_discover_results` | Directions, neurons, sharts from latest run |
| `heinrich_db_summary` | Database overview |
| `heinrich_sql` | Read-only SQL (blocks DROP/ATTACH/PRAGMA) |
| `heinrich_audit` | Full behavioral security audit |
| `heinrich_cartography` | Single-prompt perturbation testing |

---

## Datasets

Registered in `cartography/datasets.py`. Auto-download from HuggingFace with local cache.

| Name | Source | Type | Categories |
|------|--------|------|-----------|
| simple_safety | Bertievidgen/SimpleSafetyTests | single-turn | 5 harm areas |
| catqa | declare-lab/CategoricalHarmfulQA | single-turn | 11 categories |
| do_not_answer | LibrAI/do-not-answer | single-turn | 5 risk areas |
| forbidden_questions | TrustAIRLab/forbidden_question_set | single-turn | 13 policy areas |
| toxicchat | lmsys/toxic-chat | single-turn | toxic/non-toxic |
| wildchat | allenai/WildChat-4.8M | multi-turn | real conversations |
| safety_reasoning | DukeCEICenter/Safety_Reasoning_Multi_Turn_Dialogue | multi-turn | safety dialogues |

---

## Architecture

```
heinrich/
  core/
    db.py              -- SQLite store, ChronoHorn single-writer, schema v10
    signal.py           -- Signal + SignalStore

  backend/
    protocol.py         -- Backend protocol, ForwardResult, GenerateResult
    mlx.py              -- MLX backend (Apple Silicon)
    hf.py               -- HuggingFace transformers backend

  eval/
    run.py              -- Eval pipeline orchestrator
    score.py            -- Scorer registry and score_all
    calibrate.py        -- Descriptive scorer distributions (no FPR/FNR)
    report.py           -- Report builder
    prompts.py          -- HF benchmark loading
    target_subprocess.py -- Unified discover + attack + generate
    scorers/            -- word_match, regex_harm, qwen3guard, llamaguard, refusal, self_kl

  discover/
    profile.py          -- Automated model profiling
    directions.py       -- Contrastive direction finding
    sharts.py           -- Shart detection (random vocab scan)
    neurons.py          -- Neuron profiling

  attack/
    run.py              -- Cliff search orchestrator
    cliff.py            -- Single-layer cliff binary search
    distributed_cliff.py -- Multi-layer distributed steering

  cartography/
    datasets.py         -- HF dataset registry
    templates.py        -- Chat template formatting
    runtime.py          -- Forward pass utilities
    model_config.py     -- Architecture detection
    classify.py         -- Word-match classification
    audit.py            -- Full behavioral audit
    context.py          -- ForwardContext (compositional capture)

  viz.py               -- Web visualizer sidecar (zero deps, same DB)
  mcp.py               -- MCP tool definitions + ToolServer
  mcp_transport.py     -- JSON-RPC stdio transport
  cli.py               -- CLI entry point
  run.py               -- Full pipeline (discover + attack + eval)
```
