"""Runtime environment checks for compute jobs."""

from __future__ import annotations

from importlib import import_module


def assert_mamba_backend() -> None:
    """Reject missing optimized kernels rather than silently selecting a loop."""
    try:
        import_module("selective_scan_cuda")
        convolution = import_module("causal_conv1d")
        if not callable(getattr(convolution, "causal_conv1d_fn", None)):
            raise ImportError("causal_conv1d_fn missing")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Graph-Mamba requires the pinned CUDA selective scan and causal-conv1d. "
            "Install the mamba extra in an Ascend/Cardinal compute allocation; "
            "backend='reference' is an explicit CPU diagnostic, not a production fallback."
        ) from exc
