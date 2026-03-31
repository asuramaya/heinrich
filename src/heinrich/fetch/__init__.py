"""Fetch stage — acquire model data without loading weights."""

from .hf import fetch_hf_model
from .local import fetch_local_model

__all__ = ["fetch_local_model", "fetch_hf_model"]
