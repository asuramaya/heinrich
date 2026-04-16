# Mechanistic Interpretability Open Problems — Attack Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement.

**Goal:** Build tools for the 5 partially-addressed problems from Sharkey et al. 2025.

---

## Problem 1: Behavioral Control (3.2.2) — HIGHEST PRIORITY

Steer along a direction and visualize the residual stream change in real-time.

**What exists:** `attack-steer` backend, `_direction_steer_test` endpoint (generates text), direction projection coloring, full MRI data.

**What to build:**
- `/api/direction-steer-mri` endpoint: Given tokens A/B and a layer, steer the model and capture the STEERED residual stream at every layer. Return per-layer projections for the steered state alongside the clean state.
- rv0 (Direction Depth): Show BOTH clean and steered lines. The delta between them = the causal effect of steering.
- Separation panel: Show steered text alongside clean text (already done via steer-test button).

**Implementation:**
1. New function `_direction_steer_capture(mri_path, a, b, layer, alpha, prompt)`:
   - Load direction from PCA components (already done in `_direction_steer_test`)
   - Run model forward on prompt with steering enabled
   - Capture residual at each layer
   - Project steered residual onto the concept direction
   - Return per-layer clean vs steered projections
2. New endpoint `POST /api/direction-steer-mri`
3. Update rv0 to show steered overlay (green dashed line alongside red/blue)

**Data flow:** MRI scores (clean state, pre-captured) + live model forward pass (steered state) → comparison.

---

## Problem 2: Circuit Discovery (2.3) — HIGH PRIORITY

Show which attention heads write to and read from the concept direction at each layer.

**What exists:** Weight matrices stored per layer (`weights/L{NN}/*.npy`): q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. PCA components (`L{NN}_components.npy`). Attention weights per token (`attention/L{NN}_weights.npy`).

**What to build:**
- `/api/direction-circuit` endpoint: For a given direction (from pinned tokens), compute per-head attribution at each layer.
- Per-head attribution = how much does head H's output project onto the concept direction?
  - `attribution[layer][head] = ||o_proj[head] @ V @ attention_weights||` projected onto direction
  - Simpler approximation: `weight_alignment[layer][head] = |o_proj[head].T @ direction_hidden|`
- New viewport or overlay in flowers showing per-head contribution as colored bars.

**Implementation:**
1. New function `_direction_circuit(mri_path, a, b, layer)`:
   - Load direction in hidden space (via PCA components)
   - For each layer, load o_proj weights, compute per-head projection onto direction
   - Return per-layer per-head attribution scores
2. New endpoint `GET /api/direction-circuit/<model>/<mode>?a=N&b=N`
3. Display in rv2 or as overlay on existing flower viewports

---

## Problem 3: Weight-Space vs PCA-Space (2.1.1)

Bridge the gap between PCA decomposition and the model's native computation.

**What exists:** PCA components map PC space ↔ hidden space. Weight matrices in hidden space.

**What to build:**
- `/api/direction-weights` endpoint: For a direction, show how each weight matrix (Q/K/V/O/gate/up/down) at each layer aligns with it in HIDDEN space (not PC space).
- This is what the flowers SHOULD show when a direction is active — currently flowers show alignment with the current viewport PCs, not the concept direction.

**Implementation:**
1. New function `_direction_weight_alignment(mri_path, a, b)`:
   - Compute direction in hidden space
   - For each layer, for each weight matrix, compute alignment
   - Return per-layer per-matrix alignment scores
2. When direction is active, flowers auto-switch to show direction alignment instead of PC alignment

---

## Problem 4: Contextual Probing (2.2.2)

Move beyond single-token probing to multi-context concept measurement.

**What exists:** Template mode captures tokens in one context (system prompt). The eval pipeline captures generations from multiple prompts.

**What to build:**
- `/api/direction-contextual` endpoint: Take a concept (two anchor tokens) and a SET of prompts. Run the model on each prompt, capture the residual at a target layer, project onto the concept direction.
- Returns: per-prompt, per-position projection. Shows how the concept manifests in different contexts.
- This bridges the gap between "the model separates tokens X and Y in isolation" and "the model uses this separation during actual text processing."

**Implementation:**
1. New function `_direction_contextual(model_id, a_text, b_text, prompts, layer)`:
   - For each prompt: run forward, capture residual at layer
   - Project each position's residual onto the a→b direction
   - Return per-prompt per-position projections
2. New endpoint `POST /api/direction-contextual`
3. Visualization: heatmap of concept activation across prompt positions

---

## Problem 5: Microscope AI (3.5)

Extract latent knowledge by finding directions the model uses that humans haven't labeled.

**What exists:** PCA decomposition gives unlabeled axes. The top PCs capture maximum variance but we don't know what they represent.

**What to build:**
- `/api/direction-auto-discover` endpoint: For a given layer, find the top-N directions that MOST strongly split the vocabulary into bimodal distributions. These are the model's "natural features" — directions it actually uses, regardless of human labels.
- For each discovered direction: report bimodality, top tokens per side (these reveal what the direction encodes), concentration.
- This is automated feature discovery WITHOUT SAEs — using PCA + bimodality as the selection criterion.

**Implementation:**
1. New function `_auto_discover_directions(mri_path, layer, n_candidates)`:
   - The PCA components ARE candidate directions
   - For each PC: project all tokens, measure bimodality
   - Rank PCs by bimodality score
   - For top-N bimodal PCs: report top tokens per side
2. New endpoint `GET /api/direction-discover/<model>/<mode>?layer=L&n=20`
3. Display: list of discovered features with "what it separates" descriptions
4. Click to navigate viewer to that direction

---

## Execution Order

1. **Problem 2 (Circuit Discovery)** — pure computation from existing weight data, no model loading needed, fills the biggest analysis gap
2. **Problem 3 (Weight-Space)** — extends circuit discovery, makes flowers direction-aware
3. **Problem 5 (Microscope AI)** — automated discovery from existing PCA data, no model needed
4. **Problem 1 (Behavioral Control)** — requires live model, most complex
5. **Problem 4 (Contextual Probing)** — requires live model + multiple forward passes
