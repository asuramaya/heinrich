"""Backend for decepticons causal bank models.

Wraps decepticons.loader.CausalBankInference to expose the interface
Heinrich needs for MRI capture and analysis.

Usage:
    from heinrich.backend.decepticon import DecepticonBackend
    backend = DecepticonBackend("path/to/checkpoint.pt")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DecepticonConfig:
    """Minimal config that matches what Heinrich expects."""
    model_type: str
    n_layers: int  # 1 for causal bank (no layer iteration)
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    chat_format: str
    quantization: str | None

    # Causal bank specific
    n_modes: int
    n_experts: int
    n_bands: int
    embed_dim: int

    @property
    def config_hash(self):
        import hashlib
        s = f"{self.model_type}_{self.n_modes}_{self.n_experts}_{self.vocab_size}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @property
    def last_layer(self):
        return 0

    @property
    def all_layers(self):
        return [0]


class DecepticonBackend:
    """Heinrich backend for decepticons causal bank models."""

    def __init__(self, checkpoint_path: str, *,
                 result_json: str | None = None,
                 tokenizer_path: str | None = None,
                 device: str = "cpu"):
        from decepticons.loader import load_checkpoint

        self.model = load_checkpoint(
            checkpoint_path,
            result_json=result_json,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        cfg = self.model.config
        self.tokenizer = self.model.tokenizer

        self.config = DecepticonConfig(
            model_type="causal_bank",
            n_layers=1,
            hidden_size=cfg["n_modes"],  # substrate state is the "hidden state"
            intermediate_size=cfg.get("hidden_dim", 0),
            n_heads=cfg.get("n_experts", 1),
            n_kv_heads=cfg.get("n_experts", 1),
            head_dim=cfg["n_modes"] // max(cfg.get("n_experts", 1), 1),
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg.get("seq_len", 512),
            chat_format="base",
            quantization=None,
            n_modes=cfg["n_modes"],
            n_experts=cfg.get("n_experts", 1),
            n_bands=cfg.get("n_bands", 1),
            embed_dim=cfg["embed_dim"],
        )

    def forward_captured(self, token_ids: np.ndarray) -> dict:
        """Run forward pass with full internal state capture."""
        return self.model.forward_captured(token_ids)

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Get embeddings."""
        return self.model.embed(token_ids)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Get logits."""
        return self.model.forward(token_ids)

    def weights(self) -> dict:
        """All model weights as numpy arrays."""
        return self.model.weights()

    def tokenize(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.Encode(text)
        return []

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.Decode(token_ids)
        return ""
