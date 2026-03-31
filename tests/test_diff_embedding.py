import numpy as np
from heinrich.diff.embedding import project_delta_onto_embeddings, score_phrase

def test_project_returns_signals():
    rng = np.random.default_rng(42)
    delta = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    signals = project_delta_onto_embeddings(delta, embed, top_k=5)
    assert len(signals) == 5
    assert all(s.kind == "token_activation" for s in signals)
    assert signals[0].value >= signals[-1].value

def test_project_has_z_score():
    rng = np.random.default_rng(42)
    delta = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    signals = project_delta_onto_embeddings(delta, embed, top_k=3)
    assert all("z_score" in s.metadata for s in signals)

def test_score_phrase_single():
    rng = np.random.default_rng(42)
    delta = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    s = score_phrase(delta, embed, [0])
    assert s > 0

def test_score_phrase_multiple():
    rng = np.random.default_rng(42)
    delta = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    s1 = score_phrase(delta, embed, [0])
    s2 = score_phrase(delta, embed, [0, 1])
    assert s2 != s1

def test_score_phrase_empty():
    rng = np.random.default_rng(42)
    delta = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    assert score_phrase(delta, embed, []) == 0.0
