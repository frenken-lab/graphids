"""Shared graph-Mamba detector with lazy training imports."""

__all__ = ["EventGraphMamba", "EventGraphMambaState"]


def __getattr__(name):
    if name in __all__:
        from . import event_graph_mamba

        return getattr(event_graph_mamba, name)
    raise AttributeError(name)
