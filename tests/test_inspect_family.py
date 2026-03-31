from heinrich.inspect.family import classify_tensor_family, summarize_families

def test_classify_mlp_gate():
    assert classify_tensor_family("model.layers.0.mlp.gate_proj.weight") == "mlp_gate_proj"
def test_classify_mlp_expert():
    assert classify_tensor_family("model.layers.10.mlp.experts.5.gate_proj.weight") == "mlp_expert_gate"
def test_classify_attn_q_a():
    assert classify_tensor_family("model.layers.0.self_attn.q_a_proj.weight") == "attn_q_a"
def test_classify_attn_o():
    assert classify_tensor_family("model.layers.5.self_attn.o_proj.weight") == "attn_o"
def test_classify_layernorm():
    assert classify_tensor_family("model.layers.0.input_layernorm.weight") == "input_layernorm"
def test_classify_embed():
    assert classify_tensor_family("model.embed_tokens.weight") == "embed"
def test_classify_lm_head():
    assert classify_tensor_family("lm_head.weight") == "lm_head"
def test_classify_unknown():
    assert classify_tensor_family("something.weird") == "other"
def test_summarize_families():
    items = [
        {"name": "model.layers.0.self_attn.q_a_proj.weight", "spectral": {"sigma1": 10.0, "fro_norm": 50.0}},
        {"name": "model.layers.1.self_attn.q_a_proj.weight", "spectral": {"sigma1": 20.0, "fro_norm": 80.0}},
        {"name": "model.layers.0.mlp.gate_proj.weight", "spectral": {"sigma1": 5.0, "fro_norm": 30.0}},
    ]
    families = summarize_families(items)
    family_names = {f["family"] for f in families}
    assert "attn_q_a" in family_names and "mlp_gate_proj" in family_names
    attn = [f for f in families if f["family"] == "attn_q_a"][0]
    assert attn["count"] == 2 and attn["max_sigma1"] == 20.0
