from __future__ import annotations

import json
from types import SimpleNamespace

from heinrich import companion as c


def _make_live_mri(tmp_path, model_name: str = "test-model") -> str:
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model": {"name": model_name},
        "capture": {"mode": "raw"},
    }))
    return str(tmp_path)


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}:{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(int(tid)) for tid in token_ids)


class _FakeBackend:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.closed = False

    def close(self):
        self.closed = True

    def generate(self, prompt, *, max_tokens=30):
        return "fallback"

    def generate_with_geometry(self, prompt, *, max_tokens=30, top_k=5):
        return SimpleNamespace(
            text="ok",
            top_k=[(87, "W", 0.7), (88, "X", 0.2)],
            first_token="W",
            entropy=1.25,
        )


def test_live_session_status_reports_cached_backend(tmp_path):
    mri_path = _make_live_mri(tmp_path)
    backend = _FakeBackend()

    with c._steer_backend_lock:
        c._steer_backend_cache.clear()
        c._steer_backend_cache["test-model"] = backend

    status = c._live_session_status(mri_path)

    assert status["resolved_model_id"] == "test-model"
    assert status["loaded"] is True
    assert status["backend_type"] == "_FakeBackend"
    assert status["supports_generate"] is True
    assert status["supports_geometry"] is True

    with c._steer_backend_lock:
        c._steer_backend_cache.clear()


def test_live_backend_unload_closes_backend(tmp_path):
    mri_path = _make_live_mri(tmp_path)
    backend = _FakeBackend()

    with c._steer_backend_lock:
        c._steer_backend_cache.clear()
        c._steer_backend_cache["test-model"] = backend

    result = c._live_backend_unload(mri_path)

    assert result["ok"] is True
    assert result["unloaded"] == ["test-model"]
    assert result["loaded"] is False
    assert backend.closed is True


def test_live_chat_uses_loaded_backend_geometry(tmp_path, monkeypatch):
    mri_path = _make_live_mri(tmp_path)
    backend = _FakeBackend()
    monkeypatch.setattr(c, "_get_steer_backend", lambda model_id: backend)

    result = c._live_chat(
        mri_path,
        "hello",
        history=[{"role": "assistant", "content": "prev"}],
        model_id="explicit-model",
        max_tokens=12,
    )

    assert result["reply"] == "ok"
    assert result["resolved_model_id"] == "explicit-model"
    assert result["backend_type"] == "_FakeBackend"
    assert result["top_k"][0]["token"] == "W"
    assert result["reply_token_texts"] == ["o", "k"]
    assert "assistant:" in result["prompt_used"]
