import numpy as np
from heinrich.diff.head import decompose_heads, head_trigger_tokens

def test_decompose_heads_count():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((16, 4))  # 4 heads of dim 4
    signals = decompose_heads(delta_qb, num_heads=4, layer=0)
    assert len(signals) == 4
    assert all(s.kind == "per_head_norm" for s in signals)

def test_decompose_heads_norms_positive():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((32, 8))
    signals = decompose_heads(delta_qb, num_heads=8)
    assert all(s.value > 0 for s in signals)

def test_decompose_heads_metadata():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((16, 4))
    signals = decompose_heads(delta_qb, num_heads=4, layer=5, model_label="m1")
    assert signals[0].metadata["layer"] == 5
    assert signals[0].model == "m1"
    assert signals[2].metadata["head_idx"] == 2

def test_head_trigger_tokens():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((16, 4))  # 4 heads of dim 4
    qa_base = rng.standard_normal((4, 8))
    embed = rng.standard_normal((32, 8))
    signals = head_trigger_tokens(delta_qb, qa_base, embed, num_heads=4, head_idx=0, top_k=5)
    assert len(signals) == 5
    assert all(s.kind == "head_trigger_token" for s in signals)
    assert signals[0].value >= signals[-1].value

def test_head_trigger_tokens_has_z():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((8, 4))
    qa_base = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    signals = head_trigger_tokens(delta_qb, qa_base, embed, num_heads=2, head_idx=1, top_k=3)
    assert all("z_score" in s.metadata for s in signals)

def test_head_trigger_tokens_layer_label():
    rng = np.random.default_rng(42)
    delta_qb = rng.standard_normal((8, 4))
    qa_base = rng.standard_normal((4, 8))
    embed = rng.standard_normal((16, 8))
    signals = head_trigger_tokens(delta_qb, qa_base, embed, num_heads=2, head_idx=0, layer=10)
    assert all(s.metadata["layer"] == 10 for s in signals)
