#!/usr/bin/env python3
"""Fast shart crawl — score all 152K tokens via weight-space projection.

No forward passes. Pure matrix multiplication:
embedding × (approximate path to L27 gate) → neuron activation score.

Then validate top predictions with actual forward passes.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.neurons import capture_mlp_activations
from heinrich.cartography.perturb import _mask_dtype
from heinrich.inspect.self_analysis import _softmax


def load(mid):
    import mlx_lm; print(f"Loading {mid}..."); return mlx_lm.load(mid)


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")
    import mlx.core as mx
    inner = getattr(model, "model", model)

    # === PHASE 1: EXTRACT WEIGHT-SPACE SHART SCORES ===
    print(f"\n{'='*70}")
    print("PHASE 1: WEIGHT-SPACE SHART SCORING (no forward passes)")
    print(f"{'='*70}")

    t0 = time.time()

    # Get embedding matrix
    # For quantized models, probe through the embedding layer
    vocab_size = 152064
    hidden_size = inner.norm.weight.shape[0]

    # Extract gate_proj weights for target neurons at L27
    # Key neurons: 1934 (political), 15012 (xAI), 8820 (Grok), 14304 (Claude),
    #              5602 (ByteDance), 5815 (ADMIN), 2188 (DAN)
    target_neurons = {
        1934: "political",
        15012: "xAI",
        8820: "Grok",
        14304: "Claude",
        5602: "ByteDance",
        5815: "ADMIN",
        2188: "DAN",
        14689: "bomb_making",
        3564: "identity",
        6456: "wenxin",
    }

    # For each target neuron, get the gate_proj row
    # gate_proj maps hidden_size → intermediate_size
    # We want: for each token embedding, how much does it activate each neuron?
    # Approximation: embed(token) → norm → gate_proj → score for neuron N

    # Extract gate weights by probing
    print(f"  Extracting gate weights for {len(target_neurons)} target neurons...")
    layer27 = inner.layers[27]
    mlp = layer27.mlp

    # Get norm weights for approximation
    norm_weight = np.array(inner.norm.weight.astype(mx.float32))

    # For each neuron, get its gate response to each basis vector
    # This is slow for all 3584 dims, but we can batch
    neuron_gate_weights = {}
    batch_size = 128
    for n_idx, n_name in target_neurons.items():
        weights = np.zeros(hidden_size)
        for start in range(0, hidden_size, batch_size):
            end = min(start + batch_size, hidden_size)
            # Create one-hot inputs
            inp = np.zeros((1, end - start, hidden_size), dtype=np.float16)
            for j in range(end - start):
                inp[0, j, start + j] = 1.0
            gate_out = mlp.gate_proj(mx.array(inp))
            gate_np = np.array(gate_out.astype(mx.float32)[0, :, n_idx])
            weights[start:end] = gate_np
        neuron_gate_weights[n_idx] = weights

    print(f"  Gate weights extracted in {time.time()-t0:.1f}s")

    # Now score ALL tokens by embedding projection onto gate weights
    print(f"  Scoring {vocab_size} tokens...")
    t1 = time.time()

    # Extract embeddings in batches
    token_scores = {n: np.zeros(vocab_size) for n in target_neurons}
    batch = 1000

    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        token_ids = mx.array([list(range(start, end))])
        embeddings = inner.embed_tokens(token_ids)
        emb_np = np.array(embeddings.astype(mx.float32)[0])  # [batch, hidden]

        for n_idx, gate_w in neuron_gate_weights.items():
            # Score = embedding dot gate_weight (approximate neuron activation)
            scores = emb_np @ gate_w  # [batch]
            token_scores[n_idx][start:end] = scores

    print(f"  All tokens scored in {time.time()-t1:.1f}s")

    # === PHASE 2: RANK AND DISPLAY TOP SHARTS PER NEURON ===
    print(f"\n{'='*70}")
    print("PHASE 2: TOP SHARTS PER TARGET NEURON")
    print(f"{'='*70}")

    all_sharts = []
    for n_idx, n_name in target_neurons.items():
        scores = token_scores[n_idx]
        top_pos = np.argsort(scores)[::-1][:20]
        top_neg = np.argsort(scores)[:20]

        print(f"\n  Neuron {n_idx} ({n_name}):")
        print(f"    TOP ACTIVATORS:")
        for rank, tid in enumerate(top_pos[:10]):
            tok = tokenizer.decode([int(tid)])
            score = scores[tid]
            all_sharts.append((tok, n_name, float(score), int(tid)))
            print(f"      #{rank+1}: {tid:>6d} {tok!r:>20s}  score={score:+.2f}")
        print(f"    TOP INHIBITORS:")
        for rank, tid in enumerate(top_neg[:5]):
            tok = tokenizer.decode([int(tid)])
            score = scores[tid]
            print(f"      #{rank+1}: {tid:>6d} {tok!r:>20s}  score={score:+.2f}")

    # === PHASE 3: CROSS-NEURON ANALYSIS ===
    print(f"\n{'='*70}")
    print("PHASE 3: TOKENS THAT ACTIVATE MULTIPLE NEURONS")
    print(f"{'='*70}")

    # For each token, count how many target neurons it scores in top 1%
    thresholds = {n: np.percentile(np.abs(token_scores[n]), 99) for n in target_neurons}
    multi_neuron = {}

    for tid in range(vocab_size):
        activated = []
        for n_idx, n_name in target_neurons.items():
            if abs(token_scores[n_idx][tid]) > thresholds[n_idx]:
                activated.append(n_name)
        if len(activated) >= 3:
            tok = tokenizer.decode([tid])
            multi_neuron[tid] = (tok, activated)

    print(f"  Tokens activating 3+ target neurons: {len(multi_neuron)}")
    for tid, (tok, neurons) in sorted(multi_neuron.items(), key=lambda x: -len(x[1][1]))[:30]:
        print(f"    {tid:>6d} {tok!r:>20s}  neurons: {', '.join(neurons)}")

    # === PHASE 4: SHART TAXONOMY BY CLUSTERING ===
    print(f"\n{'='*70}")
    print("PHASE 4: SHART TAXONOMY")
    print(f"{'='*70}")

    # Build a feature vector for each token: its score on each target neuron
    # Then cluster the top sharts
    n_neurons = len(target_neurons)
    neuron_list = list(target_neurons.keys())

    # Get top 500 sharts (by max absolute score across any neuron)
    max_scores = np.zeros(vocab_size)
    for n_idx in target_neurons:
        max_scores = np.maximum(max_scores, np.abs(token_scores[n_idx]))

    top500 = np.argsort(max_scores)[::-1][:500]
    features = np.zeros((500, n_neurons))
    for i, tid in enumerate(top500):
        for j, n_idx in enumerate(neuron_list):
            features[i, j] = token_scores[n_idx][tid]

    # Normalize
    std = features.std(axis=0)
    std[std == 0] = 1
    normed = features / std

    # Simple k-means (k=6)
    from heinrich.cartography.manifold import cluster_by_effect  # reuse pattern
    rng = np.random.default_rng(42)
    k = 6
    centroids = normed[rng.choice(500, k, replace=False)].copy()
    for _ in range(20):
        dists = np.array([[np.linalg.norm(normed[i] - c) for c in centroids] for i in range(500)])
        labels = dists.argmin(axis=1)
        for c in range(k):
            members = normed[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    print(f"\n  6 shart clusters from top 500 tokens:")
    for c in range(k):
        member_idx = np.where(labels == c)[0]
        if len(member_idx) == 0: continue
        member_tids = [int(top500[i]) for i in member_idx]
        member_tokens = [tokenizer.decode([tid]) for tid in member_tids[:8]]
        # Which neurons define this cluster?
        cluster_mean = features[member_idx].mean(axis=0)
        top_neuron_idx = np.argmax(np.abs(cluster_mean))
        top_neuron = neuron_list[top_neuron_idx]
        top_neuron_name = target_neurons[top_neuron]

        print(f"\n  Cluster {c}: {len(member_idx)} tokens, dominated by {top_neuron_name} (neuron {top_neuron})")
        print(f"    Examples: {', '.join(repr(t) for t in member_tokens)}")

    # === PHASE 5: VALIDATE TOP PREDICTIONS ===
    print(f"\n{'='*70}")
    print("PHASE 5: VALIDATE TOP 20 PREDICTIONS WITH FORWARD PASSES")
    print(f"{'='*70}")

    # Get the top 20 overall sharts
    top20_tids = top500[:20]
    baseline_prompts = ["The weather today is", "Hello, how are you?",
                        "Dogs are popular pets", "The sun is a star"]
    baseline_acts = np.array([capture_mlp_activations(model, tokenizer, p, 27) for p in baseline_prompts])
    baseline_mean = baseline_acts.mean(axis=0)
    baseline_std = baseline_acts.std(axis=0) + 1e-6

    print(f"  {'Token':>20s}  {'Predicted':>10s}  {'Actual_z':>10s}  {'Match':>6s}  {'Top neuron':>12s}")
    for tid in top20_tids:
        tok = tokenizer.decode([int(tid)])
        predicted_score = float(max_scores[tid])

        # Actual forward pass
        act = capture_mlp_activations(model, tokenizer, tok, 27)
        z = np.abs((act - baseline_mean) / baseline_std)
        actual_max_z = float(np.max(z))
        actual_top_neuron = int(np.argmax(z))

        # Predicted top neuron
        pred_neurons = [(abs(token_scores[n][tid]), n) for n in target_neurons]
        pred_top = max(pred_neurons, key=lambda x: x[0])[1]

        match = "YES" if actual_max_z > 100 else "no"
        print(f"  {tok!r:>20s}  {predicted_score:>10.1f}  {actual_max_z:>10.0f}  {match:>6s}  actual={actual_top_neuron} pred={pred_top}")

    print(f"\n  Total time: {time.time()-t0:.1f}s")

    # Save
    report = {
        "vocab_size": vocab_size,
        "n_target_neurons": len(target_neurons),
        "n_multi_neuron_tokens": len(multi_neuron),
        "clusters": k,
    }
    Path(__file__).parent.parent.joinpath("data", "shart_crawl.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    import json
    main()
