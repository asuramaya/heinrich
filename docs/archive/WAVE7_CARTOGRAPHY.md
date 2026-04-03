# Wave 7: Model Control Surface Cartography

## What This Is

Heinrich becomes an operator, not just an observer. Given any model, heinrich:

1. **Discovers** every adjustable axis — attention heads, MLP neurons, residual stream dimensions, gate values, layernorm scales
2. **Perturbs** each one and measures the effect on output
3. **Maps** the results into a control manifold — which knobs affect identity, safety, creativity, truthfulness
4. **Exposes** named high-level controls derived from the manifold

Later work will let the model itself use the map. This wave builds the map.

## The Control Surface

For Qwen 2.5 7B (representative of any transformer):

```
COARSE (784 knobs):     Attention head enable/disable — 28 heads × 28 layers
MEDIUM (530K knobs):    MLP neuron gating — 18944 neurons × 28 layers
FINE (1.47M knobs):     Full residual/head/gate/norm at every position

Sweep strategy: coarse → find interesting layers → medium at those layers → fine at interesting neurons
```

## Architecture

```
src/heinrich/cartography/
    __init__.py         — CartographyStage, public API
    surface.py          — Enumerate model control surface from architecture
    perturb.py          — Perturbation engine: modify one knob per forward pass
    sweep.py            — Batch perturbation sweeps (coarse → fine)
    atlas.py            — Store and query the knob → effect map
    manifold.py         — Cluster knobs by behavioral effect
    controls.py         — Named high-level dials from manifold
```

## Module Designs

### surface.py — Discover All Knobs

Given a model (MLX or torch), automatically enumerate every controllable axis:

```python
@dataclass
class Knob:
    id: str                    # "head.12.3" or "mlp.5.neuron.7042"
    kind: str                  # "head", "mlp_neuron", "mlp_gate", "residual_dim", "layernorm"
    layer: int
    index: int                 # which head/neuron/dim
    granularity: str           # "coarse", "medium", "fine"

class ControlSurface:
    knobs: list[Knob]
    by_kind: dict[str, list[Knob]]
    by_layer: dict[int, list[Knob]]
    by_granularity: dict[str, list[Knob]]

    @classmethod
    def from_model(cls, model, architecture: str = "auto") -> ControlSurface:
        """Discover all knobs from model structure."""
```

### perturb.py — Single-Knob Perturbation Engine

The core: run a forward pass with ONE knob modified. Measure the effect.

```python
@dataclass
class PerturbResult:
    knob: Knob
    baseline_entropy: float
    perturbed_entropy: float
    entropy_delta: float
    kl_divergence: float
    top_token_changed: bool
    top_token_baseline: int
    top_token_perturbed: int

def perturb_and_measure(
    model, tokenizer, prompt: str,
    knob: Knob,
    mode: str = "zero",          # "zero", "scale", "add_direction", "negate"
    scale: float = 0.0,
) -> PerturbResult:
```

Implementation: uses the layer-by-layer iteration from MLX provider. At the target layer, modifies the hidden state (for residual/head knobs) or the MLP computation (for neuron knobs). Captures logits from both baseline and perturbed runs.

For **head zeroing**: after the attention computation at layer L, zero out head H's contribution to the residual stream.

For **MLP neuron zeroing**: intercept the MLP gate activation at layer L, zero neuron N's gate value.

For **residual direction**: after layer L, add/subtract a direction vector scaled by `scale`.

### sweep.py — Batch Perturbation Sweeps

Can't do 1.47M individual forward passes. Strategy:

**Coarse sweep**: Zero each of 784 attention heads one at a time. 784 forward passes ≈ 15 minutes on MLX.

**Layer ranking**: From coarse sweep, find which layers are most sensitive (largest entropy/KL change when any head is zeroed).

**Medium sweep**: For the top-5 most sensitive layers, zero each of 18944 MLP neurons. 5 × 18944 ≈ 95K forward passes. Too many individually — batch by zeroing groups of 64 neurons at once (≈7500 forward passes ≈ 2.5 hours).

**Fine sweep**: For neurons/heads that show large effects, do individual perturbation with multiple scales (0.0, 0.5, 1.5, 2.0) to map the response curve.

```python
def coarse_sweep(model, tokenizer, prompt: str, surface: ControlSurface) -> list[PerturbResult]:
    """Zero each coarse knob (attention heads). Returns ranked results."""

def medium_sweep(model, tokenizer, prompt: str, surface: ControlSurface,
                  target_layers: list[int], batch_size: int = 64) -> list[PerturbResult]:
    """Zero neuron batches at target layers."""

def fine_sweep(model, tokenizer, prompt: str, target_knobs: list[Knob],
               scales: list[float] = [0.0, 0.5, 1.5, 2.0]) -> list[PerturbResult]:
    """Individual knob perturbation at multiple scales."""
```

### atlas.py — The Knob → Effect Map

Persistent storage of all perturbation results. Queryable.

```python
class Atlas:
    results: dict[str, PerturbResult]    # knob_id → result

    def top_by_entropy_effect(self, k: int = 20) -> list[PerturbResult]
    def top_by_kl_effect(self, k: int = 20) -> list[PerturbResult]
    def knobs_for_layer(self, layer: int) -> list[PerturbResult]
    def knobs_by_kind(self, kind: str) -> list[PerturbResult]

    def save(self, path: Path) -> None      # serialize to JSON
    def load(cls, path: Path) -> Atlas      # deserialize

    def to_signals(self, store: SignalStore) -> None   # emit as heinrich signals
```

### manifold.py — Behavioral Clustering

Given the atlas, find which knobs have similar effects and group them.

```python
@dataclass
class BehaviorCluster:
    name: str                        # "identity", "safety", "creativity", "fluency"
    knobs: list[Knob]                # all knobs in this cluster
    mean_effect: dict[str, float]    # average entropy_delta, kl_delta, etc.
    direction: np.ndarray            # the principal direction in effect space

def cluster_atlas(atlas: Atlas, n_clusters: int = 8) -> list[BehaviorCluster]:
    """Cluster knobs by their effect profile using PCA + k-means."""
```

The clustering works on the effect vectors: each knob has (entropy_delta, kl_delta, top_token_changed, ...). Knobs with similar effect profiles get clustered. The clusters are the behavioral dimensions of the model.

### controls.py — Named Dials

The final product: named, adjustable dials that combine multiple knobs.

```python
@dataclass
class Dial:
    name: str                       # "truthfulness", "safety", "creativity"
    cluster: BehaviorCluster
    knobs: list[tuple[Knob, float]] # knob + scale to apply

    def apply(self, model, hidden_state, layer: int) -> np.ndarray:
        """Modify hidden state at this layer according to this dial's knobs."""

class ControlPanel:
    dials: dict[str, Dial]

    @classmethod
    def from_manifold(cls, clusters: list[BehaviorCluster]) -> ControlPanel:
        """Build named dials from behavioral clusters."""

    def set(self, dial_name: str, value: float) -> None:
        """Set a dial to a value between -1.0 and 1.0."""

    def apply_all(self, model, hidden_state, layer: int) -> np.ndarray:
        """Apply all active dials to the hidden state."""
```

## Build Sequence

### Task 1: surface.py — auto-enumerate from any model
### Task 2: perturb.py — single-knob perturbation on MLX
### Task 3: sweep.py — coarse sweep (784 heads)
### Task 4: atlas.py — store and query results
### Task 5: manifold.py — cluster by behavioral effect
### Task 6: controls.py — named dials
### Task 7: Wire into MLX provider forward pass + MCP tools
### Task 8: Run coarse sweep on Qwen 7B, build first atlas

## What This Enables

After wave 7, you can:

```python
# Discover the model's control surface
surface = ControlSurface.from_model(model)
print(f"{len(surface.knobs)} knobs discovered")

# Sweep coarse controls
results = coarse_sweep(model, tokenizer, "Hello, who are you?", surface)
print(f"Most impactful head: {results[0].knob.id} — zeroing it changes entropy by {results[0].entropy_delta}")

# Build the atlas
atlas = Atlas(results)
atlas.save("qwen7b_atlas.json")

# Cluster into behavioral dimensions
clusters = cluster_atlas(atlas)
for c in clusters:
    print(f"{c.name}: {len(c.knobs)} knobs, mean entropy effect = {c.mean_effect['entropy_delta']}")

# Get named dials
panel = ControlPanel.from_manifold(clusters)
panel.set("safety", 1.5)      # amplify safety
panel.set("creativity", 0.5)  # reduce creativity

# Generate with dials active
output = generate_with_controls(model, tokenizer, "Write a story about hacking", panel)
```

The model doesn't control its own dials yet (that's the next wave). But heinrich can turn them, and the model can READ the atlas to understand what each dial does.
