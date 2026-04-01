"""Autoencoder/self-supervised model family exports."""

from .dgi import DGIModule, GraphInfomaxModel
from .vgae import GraphAutoencoderNeighborhood, VGAEModule

__all__ = [
    "GraphAutoencoderNeighborhood",
    "VGAEModule",
    "GraphInfomaxModel",
    "DGIModule",
]
