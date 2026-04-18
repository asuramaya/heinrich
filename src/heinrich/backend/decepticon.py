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


# Chronohorn shard format (see chronohorn/data/build_byte_shards.py):
# First 256 int32 = 1024 bytes of header.
# If header[0] == TOKEN_SHARD_MAGIC, header[2] gives token count.
# Payload is uint16 regardless of vocab (sp tokens or bytes stored in uint16 slots).
_SHARD_MAGIC = 20240520
_SHARD_VERSION = 1
_SHARD_HEADER_BYTES = 256 * 4  # 256 int32


def load_val_sequences(
    path: str,
    *,
    seq_len: int = 512,
    n_seqs: int = 50,
    seed: int = 42,
    byte_level: bool = False,
) -> np.ndarray:
    """Load validation data from a .bin file, return [n_seqs, seq_len] int64.

    Takes non-overlapping random slices from the token stream.
    Returns fewer sequences if data is too short.

    Chronohorn shards store tokens as uint16 after a 1024-byte header,
    regardless of vocab size — byte-level shards hold byte values (0-255)
    in uint16 slots. If `byte_level=True` and a shard header is present,
    the header is skipped and payload is still read as uint16.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    has_header = False
    token_count = None
    if raw.size >= _SHARD_HEADER_BYTES:
        header = raw[:_SHARD_HEADER_BYTES].view(np.int32)
        if int(header[0]) == _SHARD_MAGIC and int(header[1]) == _SHARD_VERSION:
            has_header = True
            token_count = int(header[2])

    if has_header:
        payload = raw[_SHARD_HEADER_BYTES:].view(np.uint16)
        if token_count is not None and token_count <= len(payload):
            payload = payload[:token_count]
        tokens = payload.astype(np.int64)
    elif byte_level:
        # Legacy raw-bytes file: each uint8 is one byte token.
        tokens = raw.astype(np.int64)
    else:
        tokens = raw.view(np.uint16).astype(np.int64)

    n_total = len(tokens)
    max_seqs = n_total // seq_len
    actual_seqs = min(n_seqs, max_seqs)
    if actual_seqs == 0:
        raise ValueError(f"Val data too short: {n_total} tokens < seq_len {seq_len}")

    if byte_level:
        # Guard against value-range mismatch.
        sample = tokens[:min(n_total, 10_000_000)]
        warnings_issued = []

        if sample.max() > 255:
            warnings_issued.append(
                f"payload max value is {int(sample.max())} > 255 — file is "
                f"likely sp-tokenized, not byte-level")

        # UTF-16LE detection: English ASCII text in UTF-16LE has ~50% zero
        # bytes (the high byte of each ASCII character). Real UTF-8 text has
        # ~1–5% zero bytes. Flag zero-fraction > 30% as suspicious.
        zero_frac = float((sample == 0).mean())
        if zero_frac > 0.30:
            warnings_issued.append(
                f"{zero_frac:.1%} of bytes are zero — file may be UTF-16 or "
                f"padded; byte-level models expect UTF-8")

        # Printable-ASCII floor: real English byte streams are ~60–95% printable
        # ASCII (bytes 32–126). < 25% printable suggests binary or wrong encoding.
        printable_frac = float(((sample >= 32) & (sample < 127)).mean())
        if sample.max() <= 255 and printable_frac < 0.25 and zero_frac < 0.30:
            warnings_issued.append(
                f"only {printable_frac:.1%} printable ASCII — file may be "
                f"binary or non-text; expected ~60% for English web data")

        # Entropy check: uniform-random bytes give ~8 bits/byte. English text
        # gives ~4–5 bits/byte. Flag > 7.5 as likely random or encrypted data.
        hist = np.bincount(sample.astype(np.int64) & 0xFF, minlength=256).astype(np.float64)
        hist = hist[hist > 0] / hist.sum()
        ent = float(-(hist * np.log2(hist)).sum())
        if ent > 7.5 and sample.max() <= 255:
            warnings_issued.append(
                f"byte entropy {ent:.2f} bits/byte is near-uniform — data "
                f"may be random or encrypted, not natural language")

        if warnings_issued:
            import warnings
            msg = f"load_val_sequences(byte_level=True) on {path}: " + \
                  "; ".join(warnings_issued) + \
                  ". Consider fetching a verified byte shard."
            warnings.warn(msg, stacklevel=2)

    rng = np.random.RandomState(seed)
    all_starts = np.arange(max_seqs) * seq_len
    chosen = rng.choice(len(all_starts), actual_seqs, replace=False)
    chosen.sort()
    starts = all_starts[chosen]

    seqs = np.stack([tokens[s:s + seq_len] for s in starts])
    return seqs


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

    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank models do not support text generation")

    def generate_with_geometry(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank models do not support text generation")

    def capture_residual_states(self, prompts, **kwargs):
        raise NotImplementedError("Causal bank: use forward_captured() for substrate states")

    def capture_mlp_activations(self, prompt, layer):
        raise NotImplementedError("Causal bank: no MLP layers")

    def capture_all_positions(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank: use forward_captured() for substrate states")

    def capture_mlp_detail(self, prompt, layer):
        raise NotImplementedError("Causal bank: no MLP layers")

    def forward_with_neuron_mask(self, prompt, layer, neuron_indices, **kwargs):
        raise NotImplementedError("Causal bank: no neurons to mask")

    def perturb_head(self, prompt, layer, head, **kwargs):
        raise NotImplementedError("Causal bank: no attention heads")

    def weight_projection(self, layer, neuron_index):
        raise NotImplementedError("Causal bank: no gate_proj weights")

    def capture_attention_patterns(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank: no attention mechanism")

    def capture_per_layer_delta(self, prompt):
        raise NotImplementedError("Causal bank: single-layer architecture")

    def steer_and_generate(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank: no steering support")

    def instrumented_forward(self, prompt, **kwargs):
        raise NotImplementedError("Causal bank: use forward_captured() instead")

    def capabilities(self):
        return {"mri": True, "generate": False, "steer": False, "attention": False}

    def forward_context(self):
        raise NotImplementedError("Causal bank: no ForwardContext support")

    def generation_context(self, prompt):
        raise NotImplementedError("Causal bank: no GenerationContext support")
