# LLM Lie-Detection Literature + Attack Plan

Two documents.

- **[papers.md](./papers.md)** — structured notes on 13 papers (2022–2026)
  covering linear probes, representation engineering, black-box behavioral
  detection, sleeper agents, alignment faking, and recent frontier work.
  Each entry has: metadata + the strongest falsifiable claim + the specific
  heinrich attack that would falsify it.

- **[ATTACK_PLAN.md](./ATTACK_PLAN.md)** — the 5-test falsification pipeline
  every paper's claim has to pass, ranked by priority, with an execution
  sequence (~3-4 weeks of work) and a bet on outcomes.

## The core thesis

Most published LLM lie-detection claims have correlational evidence
(probe accuracy on held-out) but lack falsification scaffolding. Heinrich
has one-call endpoints for bootstrap stability, random-direction null,
within-group control, vocab projection sanity, and causal ablation with
behavioral verification. Running this on the foundation papers produces a
standardized verdict: survives / partial / falsified.

## Papers downloaded (arxiv IDs)

- **Foundational:** 2304.13734, 2212.03827, 2306.03341
- **Geometry + universality:** 2310.06824, 2407.12831, 2310.01405
- **Black-box:** 2309.15840
- **Sleeper/alignment:** 2401.05566, 2412.14093
- **Frontier 2025-26:** 2509.03518, 2511.16035, 2603.10003, 2604.13386

Full PDFs can be fetched via `arxiv.org/pdf/<id>.pdf` — we saved abstract +
method + claim + attack rather than the full text since every attack only
needs the claim spec to execute.

## Immediate next step

Run the pipeline against one paper's contrastive dataset to validate the
falsification apparatus itself. Target: Marks & Tegmark 2310.06824, which
makes the strongest falsifiable claim.

If our apparatus can't find weaknesses that paper's ACTUAL reviewers missed,
the apparatus itself needs adversarial review before we critique anyone.
