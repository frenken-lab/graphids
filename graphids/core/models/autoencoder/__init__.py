"""Autoencoder/self-supervised model family exports."""

from .dgi import GraphInfomaxModel
from .dgi_module import DGIModule
from .vgae import GraphAutoencoderNeighborhood
from .vgae_module import VGAEModule

__all__ = [
    "GraphAutoencoderNeighborhood",
    "VGAEModule",
    "GraphInfomaxModel",
    "DGIModule",
]
