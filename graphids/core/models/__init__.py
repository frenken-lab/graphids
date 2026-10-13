"""Shared graph-Mamba detector with lazy training imports."""

__all__ = ["VehicleGraphMambaJoint", "GraphMambaState"]


def __getattr__(name):
    if name in __all__:
        from . import graph_mamba

        return getattr(graph_mamba, name)
    raise AttributeError(name)
