"""Autoencoder/self-supervised model family exports."""

from .dgi import DGI
from .vgae import VGAE

__all__ = ["VGAE", "DGI"]
