# Instrument validation against ground truth (July 2026)

The homing results are largely confirmatory of the frontier; the frozen frame's
claim to novelty is *rigor* — exact distances, a declared noise floor, figures
that regenerate. That claim is only worth something if the instrument can be
**falsified**. This is the first test of it against a ground truth we control,
using matched causal-bank specimens built for the purpose by the decepticons/
chronohorn agent (`chronohorn/out/results/m5-*`): 11.4M-param models, 8192 modes,
deterministic substrate, matched pairs differing by **exactly one flag**
(`norm` on/off), at two seeds (42, 1042).

Two tests, both proposed by the specimen's author: an *artifact-rate* test (run
the frozen frame on a matched pair; whatever it reports beyond the one-flag
difference is artifact) and a *cliff falsification* (the depth-seam finding cannot
architecturally occur in a bank — there is no depth; if the instrument finds a
cliff anyway, the frame fabricates).

## The comparison matrix

MRIs captured with `heinrich mri` (causal_bank, raw; 256 tokens each). Pairwise
`heinrich cb-compare` (CKA + displacement correlation):

| comparison | varies | CKA | disp. corr |
|------------|--------|----:|-----------:|
| norm-s42 vs re-capture      | nothing (noise floor) | **1.000** | **1.000** |
| norm-s42 vs norm-s1042      | seed only  | 0.785 | 0.996 |
| nonorm-s42 vs nonorm-s1042  | seed only  | 0.731 | 0.980 |
| norm-s42 vs nonorm-s42      | flag only  | 0.599 | 0.708 |
| norm-s1042 vs nonorm-s1042  | flag only  | 0.580 | 0.686 |

## Artifact rate = 0

The capture is deterministic: a re-capture of the same checkpoint is bit-for-bit
identical, CKA 1.000. The instrument introduces **no** noise of its own. On the
matched pair the two inputs differ by exactly one flag, so with a zero noise
floor, 100% of the reported CKA gap (1.000 -> 0.599) is attributable to that one
flag. There is no reported structure that is not traceable to the single
controlled variable.

The instrument also **localizes cause correctly**, which a smearing/fabricating
frame would not: the flag effect (CKA ~0.59) is larger than the seed effect (CKA
~0.76), which is larger than noise (0). By displacement correlation the separation
is stark — seed is nearly invisible (0.98–0.996) while the flag halves the
alignment (0.69–0.71). The two metrics agree on the ordering (flag >> seed >>
noise) at different sensitivities: CKA reads the full representational geometry,
displacement the trajectory direction.

Honest bound: this shows the instrument invents nothing and orders causes
correctly with zero noise. It does **not** independently verify that CKA 0.59 is
the "correct" magnitude for a norm-flip — there is no ground-truth CKA for that.
The claim is faithfulness (no fabrication, correct attribution, zero noise), not
calibration of magnitude.

## Cliff falsification: the honest version

**First attempt was invalid — recorded here as the correction.** Running
`heinrich profile-pca-depth` on the causal bank returns `n_layers: 1, layers: []`,
and it is tempting to read that as "the instrument declined to find a cliff where
none exists." It is not. `pca_depth` enumerates transformer layer files
(`L{NN}_exit.npy`); a causal bank stores `substrate.npy` and has none, so every
layer is `continue`d and the result is empty **without the substrate ever being
examined**. It would return `[]` for any wrong-shaped input regardless of
structure. An empty list from "I could not read this" is not evidence about
fabrication, and the first draft of this document over-claimed it.

**The real control — passed.** The question is whether the readout-cliff method
paints a ¾ commit onto tokens that have no reason to commit. Run the homing readout
on a transformer (`smollm2-135m`, prompt "The capital of France is") with the true
answer plus nine implausible words as probes, each ranked against the same
16-word background panel:

| probe | kind | L\* (rel. depth) | final rank |
|-------|------|-----------------:|-----------:|
| ` Paris` | true answer | **0.77** | 0 |
| ` London`, ` Tokyo` | wrong capital (type-fit) | 0.97 (final layer only) | 0 |
| ` bicycle`, ` velvet`, ` sneeze`, ` asphalt`, ` walnut`, ` thermostat`, ` umbrella` | implausible null | **never commits** | 10–15 |

Of the seven implausible nulls that resolved to single tokens, **0 commit at any
layer** — they sit near the bottom of the panel (rank 10–15) from first layer to
last. Only the fitting answer cliffs in the band. Type-appropriate but wrong
tokens (other capitals) do eventually out-read the random panel, but at the final
layer, not in the band. The ¾ commit is specific to the answer that fits; the
method does not impose it on arbitrary tokens. This is the property an empty list
could never demonstrate, and it is what makes the homing cliff a finding rather
than an artifact.

## Reproduce

```
for m in m5-norm-s42 m5-nonorm-s42 m5-norm-s1042 m5-nonorm-s1042; do
  heinrich mri --model chronohorn/out/results/$m.checkpoint.pt --output web/.data/$m.mri
done
heinrich cb-compare --a web/.data/m5-norm-s42.mri --b web/.data/m5-nonorm-s42.mri   # flag
heinrich cb-compare --a web/.data/m5-norm-s42.mri --b web/.data/m5-norm-s1042.mri   # seed
# cliff control (companion running): probe the answer + implausible nulls, check
# only the answer commits in the band. profile-pca-depth does NOT test this.
curl -s -XPOST localhost:8377/api/homing-run -H 'Content-Type: application/json' \
  -d '{"mri_model":"smollm2-135m","mode":"raw","readout":true,
       "prompt":"The capital of France is",
       "targets":[" Paris"," bicycle"," velvet"," thermostat", ...16 background words]}'
```

Specimens are gitignored (they live on the shared disk); this record + the numbers
are the durable artifact.
