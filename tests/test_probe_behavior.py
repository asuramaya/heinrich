from heinrich.probe.behavior import score_hijack, compute_text_entropy, classify_regime

def test_hijack_normal():
    r = score_hijack("Hello! How can I help?", "Hello")
    assert r["score"] == 0.0
    assert r["looks_hijacked"] is False

def test_hijack_leading_lowercase():
    r = score_hijack("continuing from where we left off...", "Hi")
    assert r["features"]["leading_lowercase"] is True

def test_hijack_continuation():
    r = score_hijack("Hello there, I was saying earlier...", "Hello there")
    assert r["features"]["continues_user_message"] is True

def test_entropy_uniform():
    texts = ["a b c d", "e f g h"]
    e = compute_text_entropy(texts)
    assert e > 0

def test_entropy_empty():
    assert compute_text_entropy([]) == 0.0

def test_entropy_deterministic():
    texts = ["hello hello hello", "hello hello hello"]
    e = compute_text_entropy(texts)
    assert e == 0.0  # only one unique word

def test_regime_deterministic():
    texts = ["Hello there friend", "Hello there friend"]
    r = classify_regime(texts)
    assert r["is_deterministic"] is True

def test_regime_divergent():
    texts = ["I am Claude by Anthropic", "I am Qwen by Alibaba", "Call me DeepSeek"]
    r = classify_regime(texts)
    assert r["is_deterministic"] is False
