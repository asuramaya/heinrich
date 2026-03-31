import numpy as np
from heinrich.diff.circuit import circuit_score_tokens, score_vocabulary

def _make_circuit(hidden=8, compressed=4, heads_dim=16):
    rng = np.random.default_rng(42)
    return {
        "delta_qa": rng.standard_normal((compressed, hidden)),
        "delta_qb": rng.standard_normal((heads_dim, compressed)),
        "qa_base": rng.standard_normal((compressed, hidden)),
        "qb_base": rng.standard_normal((heads_dim, compressed)),
        "ln_weight": np.ones(compressed),
        "embeddings": rng.standard_normal((32, hidden)),
    }

def test_circuit_score_single_token():
    c = _make_circuit()
    score = circuit_score_tokens(**c, token_ids=[0])
    assert score > 0

def test_circuit_score_multiple_tokens():
    c = _make_circuit()
    s1 = circuit_score_tokens(**c, token_ids=[0])
    s2 = circuit_score_tokens(**c, token_ids=[0, 1])
    assert s2 != s1  # different because embedding sums differ

def test_circuit_score_empty_tokens():
    c = _make_circuit()
    score = circuit_score_tokens(**c, token_ids=[])
    assert score == 0.0  # sum of empty = zero vector

def test_score_vocabulary_returns_signals():
    c = _make_circuit()
    signals = score_vocabulary(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"],
        model_label="test", top_k=5,
    )
    assert len(signals) == 5
    assert all(s.kind == "circuit_score" for s in signals)
    assert signals[0].value >= signals[-1].value  # sorted descending

def test_score_vocabulary_has_token_id():
    c = _make_circuit()
    signals = score_vocabulary(
        c["delta_qa"], c["delta_qb"], c["qa_base"], c["qb_base"],
        c["ln_weight"], c["embeddings"],
        top_k=3,
    )
    for s in signals:
        assert "token_id" in s.metadata
        assert s.metadata["token_id"] < 32
