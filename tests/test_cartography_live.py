import pytest
try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_coarse_sweep_real_model():
    """Run coarse head sweep on first 2 layers of Qwen 7B."""
    from heinrich.cartography.surface import ControlSurface
    from heinrich.cartography.sweep import coarse_head_sweep
    from heinrich.cartography.atlas import Atlas
    from heinrich.probe.mlx_provider import MLXProvider

    provider = MLXProvider({"model": "Qwen/Qwen2.5-7B-Instruct"})
    provider._ensure_loaded()

    # Only sweep first 2 layers (56 heads) to keep test fast
    surface = ControlSurface.from_config(n_layers=2, n_heads=28, head_dim=128, intermediate_size=18944, hidden_size=3584)

    prompt = provider._tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}], tokenize=False, add_generation_prompt=True)

    results = coarse_head_sweep(provider._model, provider._tokenizer, prompt, surface)

    assert len(results) > 0
    atlas = Atlas()
    atlas.add_all(results)

    # The most impactful head should have KL > 0
    top = atlas.top_by_kl(1)
    assert top[0].kl_divergence > 0
    print(f"Top head: {top[0].knob.id} KL={top[0].kl_divergence:.4f} changed_top={top[0].top_token_changed}")
