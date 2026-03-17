"""GraphIDS: Graph-based Intrusion Detection System.

Public API:
    from graphids import PipelineConfig, resolve, checkpoint_path
    import graphids.core  # lazy-loaded
    import graphids.pipeline  # lazy-loaded
"""

__version__ = "1.0.0"

from graphids.config import PipelineConfig, checkpoint_path, resolve

_lazy_submodules = {"core", "pipeline"}


def __getattr__(name):
    if name in _lazy_submodules:
        import importlib

        return importlib.import_module(f"graphids.{name}")
    raise AttributeError(f"module 'graphids' has no attribute {name!r}")


def __dir__():
    return [*_lazy_submodules, "PipelineConfig", "resolve", "checkpoint_path", "__version__"]
