"""Backward compatibility -- imports from heinrich.backend.

All backend code has been extracted to the heinrich.backend package.
This module re-exports everything so existing ``from .backend import ...``
and ``from heinrich.cartography.backend import ...`` continue to work.
"""
from heinrich.backend.protocol import Backend, ForwardResult, load_backend  # noqa: F401
from heinrich.backend.mlx import MLXBackend  # noqa: F401
from heinrich.backend.hf import HFBackend  # noqa: F401
