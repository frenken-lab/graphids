"""Multiprocessing start-method guard.

Must run before any CUDA-touching DataLoader worker is spawned, per
critical-constraints.md — fork + CUDA is a silent segfault. Called
once at the entry of each training command (``cli/training.py::fit``
and ``orchestrate/run.py::run_pipeline``).
"""

from __future__ import annotations

import multiprocessing

import torch.multiprocessing

_SPAWN_SET = False


def ensure_spawn() -> None:
    """Set mp start method to ``spawn`` + tensor IPC to ``file_system``.

    Idempotent — subsequent calls are a no-op.
    """
    global _SPAWN_SET  # noqa: PLW0603
    if _SPAWN_SET:
        return
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")
    _SPAWN_SET = True
