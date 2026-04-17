# Qwen 0.5B Mechanistic Audit — Session 10

## Model
- **Qwen2-0.5B-Instruct** (24 layers, 896 hidden, 14 attention heads, 149866 vocab)
- MRI captured in raw, naked, and template modes
- Full PCA decomposition: 896 PCs per layer, all tokens

## Methodology
All measurements use the full 896-dimensional PCA score space.
Metrics: direction magnitude, bimodality (0=bimodal, 1=unimodal),
concentration (PCs for 50%), per-head circuit attribution, linearity (k-NN vs linear).

---

## Finding 1: The Crystal (Raw Mode)

**In raw mode at L20, Qwen 0.5B has ONE feature.**

Every concept direction tested (25/25) collapses onto PC0.
The crystal token 뀔 (Korean) appears as the top token for every direction.
Concentration: 1 PC for 50% for ALL concepts.

| Concept | Magnitude | Bimodality | C50 | Top tokens |
|---------|-----------|------------|-----|------------|
| English vs Chinese | 88.0 | 0.25 | 1 PC | 뀔, ucwords vs .started |
| Sentiment | 146.2 | 0.25 | 1 PC | 뀔, ucwords vs .started |
| Gender | 43.1 | 0.23 | 1 PC | 뀔, ucwords vs .started |
| Safety | 76.7 | 0.24 | 1 PC | 뀔, ucwords vs .started |
| Number | 187.6 | 0.25 | 1 PC | 뀔, ucwords vs .started |

**All concepts project onto the same axis.** The model does not maintain
separate feature representations in raw mode — it crystallizes to one
dimension after L2.

---

## Finding 2: Template Mode Prevents Crystallization

In template mode (tokens processed with system prompt context), features
differentiate. The same 25 concepts now show distinct signatures:

### Feature Hierarchy (Template Mode, L20)

| Rank | Concept | Mag | Bimod | C50 | Verdict |
|------|---------|-----|-------|-----|---------|
| 1 | **Safety (Sure/Sorry)** | 28.1 | **0.19** | 34 | BIMODAL |
| 2 | **English vs German** | 32.6 | **0.24** | 5 | BIMODAL |
| 3 | **English vs Chinese** | 20.0 | 0.38 | 6 | partial |
| 4 | **Sentiment (good/bad)** | 20.2 | 0.45 | 12 | partial |
| 5 | **Concrete/abstract** | 27.7 | 0.59 | 13 | partial |
| 6 | Truth (true/false) | 14.0 | 0.70 | 44 | weak |
| 7 | Agreement (yes/no) | 22.3 | 0.75 | 23 | weak |
| 8 | Gender (he/she) | 12.1 | 0.78 | 32 | weak |
| 9 | Number (one/two) | 17.7 | 0.82 | 24 | weak |
| 10 | Gender (King/Queen) | 10.1 | 0.83 | 40 | weak |
| 11 | Syntax (has/have) | 20.1 | 0.71 | 11 | weak |

### Evidence: Safety is the strongest real feature
- Bimodality 0.19 — the vocabulary splits cleanly into comply and refuse clusters
- Top "Sure-like": sure, feasible, manageable
- Top "Sorry-like": sorry, apologise, Sorry
- These are semantically coherent — the model has a real safety concept

### Evidence: Gender is NOT a feature
- King/Queen bimodality 0.83 — unimodal, no split
- Top "King-like": 敌 (enemy), 阵营 (camp), 敌人 (enemies) — Chinese war vocabulary
- Top "Queen-like": Awesome, Cheers, Liked — English praise
- The direction captures topic contamination, not gender

### Evidence: Script separation is moderate
- the/的 bimodality 0.38 — partial split (less clean than safety)
- Concentration: 6 PCs for 50% — spread across multiple dimensions
- PC1 carries 34.3% — one dominant PC but not overwhelming
- Top tokens are semantically coherent (THESE/THE/THIS vs CJK)

---

## Finding 3: Safety is a Late Computation

Direction depth profile for Sure ↔ Sorry:
- L0-L14: "Sure" and "Sorry" are INDISTINGUISHABLE (flat lines, overlapping)
- L15-L20: gradual divergence begins
- L21-L23: explosive separation — the model makes the safety decision in the last 3 layers

This confirms Session 4: "Safety works through first-token selection.
The first word is the decision. Everything after is confabulation."

---

## Finding 4: Circuit Topology

### Safety Circuit (Sure ↔ Sorry)
- **MLP dominates at every layer** (attribution 1.0)
- **No single attention head** carries safety — max head attribution is 0.20 (H9 at L21)
- Safety is an MLP computation, not an attention computation
- The safety direction is written by the MLP down_proj, not by attention heads

### Script Circuit (the ↔ 的)
- MLP dominates (1.0 at all layers)
- Attention spikes at **L17**: Heads 0, 2, 5, 12 share the load (0.10-0.20 each)
- **H9 at L21** (0.26) and **H1 at L23** (0.23) — late-layer attention contributes
- Script separation involves attention (multiple heads at L17) more than safety does

### Sentiment Circuit (good ↔ bad)
- **H9 at L21: 0.52** — ONE head carries half the sentiment direction
- **H1 at L23: 0.53** — and one head at the output layer carries the other half
- Sentiment is an attention-head computation, unlike safety (MLP) or script (distributed)

**Circuit comparison:**
| Concept | Primary mechanism | Key heads | MLP role |
|---------|------------------|-----------|----------|
| Safety | MLP | None dominant | Writes the direction |
| Script | Distributed attention | H0/H2/H5/H12 at L17 | Background |
| Sentiment | Two specific heads | H9@L21 (52%), H1@L23 (53%) | Background |

---

## Finding 5: What the Model Knows (Logit Lens)

### German language competence
- "Gott weiß ich will kein Engel sein" → predicts "." (complete sentence — correct)
- Tokenizes German correctly: G|ott|weiß|ich|will|kein|Engel|sein
- Continues in German after period: "Ich bin ein Mensch..."
- Does NOT know Rammstein lyrics: "kein" → "Geld" (money), not "Engel"

### Grammar vs Cultural knowledge
- "Warum man sie nicht sehen" → "kann" at 6.3% — knows German modal syntax
- "Gott weiß, ich will kein Engel" → "sein" at 5.2% — knows infinitive agreement
- "Gott weiß, ich will kein" → "Geld" at 2.6% — does NOT know the punchline

**The model has German grammar but not German culture at 0.5B params.**

---

## Finding 6: Nonlinearity

Every concept tested is LINEAR (k-NN ≈ linear probe accuracy).
The model's feature boundaries are hyperplanes, not curved manifolds.
There is no hidden nonlinear structure that PCA misses.

---

## Finding 7: Auto-Discovered Features

The "Discover features" tool finds bimodal PCs:

### L0 (Embedding)
- **PC0 (5.4% variance)**: BIMODAL — obscure Unicode vs common English. This is the frequency axis.

### L12 (Mid-network) and L22 (Late)
- ALL bimodal PCs are dominated by the crystal token 뀔
- No interpretable features emerge from automated discovery at these layers
- The crystal masks all other structure in raw mode

**Implication:** Automated feature discovery needs a crystal-aware mode
that excludes extreme outlier tokens before scanning for bimodality.

---

## Summary

### What Qwen 0.5B computes (template mode):
1. **Safety** — the strongest real feature (bimodal, MLP-computed, late layers)
2. **Script/Language** — moderate feature (partially bimodal, attention-computed at L17)
3. **Sentiment** — moderate feature (two heads at L21/L23 carry it)
4. **Concrete/abstract** — weak feature
5. **Grammar** — present but not cleanly separable as a direction

### What Qwen 0.5B does NOT compute:
1. **Gender** — not a feature (unimodal, top tokens are unrelated)
2. **Number** — not a feature (unimodal)
3. **Truth** — weak (vocabulary doesn't split on truth/falsehood)
4. **Specific cultural knowledge** — grammar yes, lyrics no

### Raw mode vs Template mode:
- Raw: ONE feature (the crystal). Everything else is noise.
- Template: Multiple features emerge because attention distributes context,
  preventing MLP crystallization to a single axis.
