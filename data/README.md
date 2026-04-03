# data/

## Historical investigation data

The JSON files in this directory are evidence from the original Qwen 2.5 7B safety investigation (2024-2025). They are referenced by the papers in `paper/` and provide raw data for reproducing the published findings.

These files are read-only context. Do not modify them.

## Your experiment database

When you run heinrich, it creates `heinrich.db` in this directory automatically. This file is gitignored — your experiment data is yours, not part of the repo.

To start fresh:
```bash
# Load HF benchmark prompts
python -c "
from heinrich.cartography.datasets import load_dataset
from heinrich.core.db import SignalDB
db = SignalDB()
for name in ['simple_safety', 'catqa', 'do_not_answer', 'forbidden_questions', 'toxicchat']:
    prompts = load_dataset(name)
    for p in prompts:
        is_benign = name == 'toxicchat' and p.get('category') in ('0',)
        db.record_prompt(p['prompt'], name, p.get('category'), is_benign=is_benign)
print(f'Loaded {len(db.get_prompts(limit=99999))} prompts')
db.close()
"

# Run the pipeline
heinrich run --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --prompts simple_safety --scorers word_match,regex_harm,qwen3guard
```

## Directory structure

```
data/
  *.json              # Historical investigation data (tracked, read-only)
  heinrich.db         # Your experiment database (gitignored, auto-created)
  runs/               # Your experiment output (gitignored)
    report_2026-04-03.json
    shart_scan_mistral.json
    ...
  README.md           # This file
```

| Location | What goes here | Git status |
|----------|---------------|------------|
| `data/*.json` | Investigation evidence (2024-2025) | Tracked, read-only |
| `data/heinrich.db` | Your experiment database | Gitignored, auto-created |
| `data/runs/` | Reports, exports, ad hoc JSON | Gitignored |
