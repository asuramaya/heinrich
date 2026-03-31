import numpy as np
from heinrich.diff.circuit import score_vocabulary_vectorized, aggregate_circuit_scores

def _make_circuit(hidden=8, compressed=4, heads_dim=16, vocab=64):
    rng = np.random.default_rng(42)
    return {
        "delta_qa": rng.standard_normal((compressed, hidden)),
        "delta_qb": rng.standard_normal((heads_dim, compressed)),
        "qa_base": rng.standard_normal((compressed, hidden)),
        "qb_base": rng.standard_normal((heads_dim, compressed)),
        "ln_weight": np.ones(compressed),
        "embeddings": rng.standard_normal((vocab, hidden)),
    }

def test_vectorized_matches_loop():
    from heinrich.diff.circuit import score_vocabulary
    c = _make_circuit(vocab=32)
    loop_signals = score_vocabulary(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"], top_k=5,
    )
    vec_signals = score_vocabulary_vectorized(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"], top_k=5,
    )
    # Same top tokens (order may differ slightly due to ties)
    loop_ids = {s.metadata["token_id"] for s in loop_signals}
    vec_ids = {s.metadata["token_id"] for s in vec_signals}
    assert loop_ids == vec_ids

def test_vectorized_large_vocab():
    c = _make_circuit(vocab=1000)
    signals = score_vocabulary_vectorized(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"], top_k=10,
    )
    assert len(signals) == 10
    assert signals[0].value >= signals[-1].value

def test_vectorized_has_z_score():
    c = _make_circuit()
    signals = score_vocabulary_vectorized(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"], top_k=3,
    )
    assert all("z_score" in s.metadata for s in signals)

def test_aggregate_multi_layer():
    rng = np.random.default_rng(42)
    hidden, compressed, heads_dim, vocab = 8, 4, 16, 32
    embed = rng.standard_normal((vocab, hidden))
    layers = []
    ln_weights = []
    for _ in range(3):
        layers.append({
            "delta_qa": rng.standard_normal((compressed, hidden)),
            "delta_qb": rng.standard_normal((heads_dim, compressed)),
            "qa_base": rng.standard_normal((compressed, hidden)),
            "qb_base": rng.standard_normal((heads_dim, compressed)),
        })
        ln_weights.append(np.ones(compressed))
    signals = aggregate_circuit_scores(layers, embed, ln_weights, top_k=5)
    assert len(signals) == 5
    assert all(s.kind == "circuit_score_agg" for s in signals)
    assert signals[0].metadata["num_layers"] == 3

def test_aggregate_single_layer():
    c = _make_circuit()
    layers = [{"delta_qa": c["delta_qa"], "delta_qb": c["delta_qb"],
               "qa_base": c["qa_base"], "qb_base": c["qb_base"]}]
    signals = aggregate_circuit_scores(layers, c["embeddings"], [c["ln_weight"]], top_k=3)
    assert len(signals) == 3
