"""CPU thread-pool configuration.

A training process sees a hard CPU quota set by the SLURM cgroup (or the
local machine's core count). PyTorch doesn't read that quota — by default
it sizes intra-op threads to ``os.cpu_count()`` (node-wide, ignoring
affinity) and interop threads to ``cpu_count/2``. On a SLURM node that
gives you ``--cpus-per-task=16``, torch will happily try to spawn 64+
threads across NUMA domains; BLAS backends fight over cores and
throughput drops.

``configure_cpu_threads()`` fixes this:

- **intra-op** (``torch.set_num_threads``) = N = ``SLURM_CPUS_PER_TASK``
  (or ``os.cpu_count()`` fallback). BLAS / matmul / any single-op
  parallelism can use every allocated core.
- **interop** (``torch.set_num_interop_threads``) = 1. Cross-op fork-join
  parallelism double-subscribes when intra-op already fills the quota;
  on CPU-bound training (fusion bandit/dqn/mlp) this consistently hurts
  more than it helps.
- ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` env vars pinned to N so
  BLAS backends that read env (not torch) stay in sync.

Must run before any ``torch`` op launches a worker — once parallel work
starts, ``set_num_interop_threads`` raises. The function is idempotent
via a module flag; subsequent calls are no-ops.
"""

from __future__ import annotations

import os

from graphids._otel import get_logger

log = get_logger(__name__)

_CONFIGURED = False


def allocated_cpus() -> int:
    """CPU quota visible to this process. SLURM cgroup > os.cpu_count()."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return (int(slurm) if slurm and slurm.isdigit() else None) or os.cpu_count() or 1


def configure_cpu_threads() -> int:
    """Pin torch + OMP thread counts to the SLURM CPU quota. Idempotent.

    Returns the intra-op thread count it installed.
    """
    global _CONFIGURED  # noqa: PLW0603
    import torch

    if _CONFIGURED:
        return torch.get_num_threads()

    n = allocated_cpus()
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some torch op already launched — log and continue with whatever
        # interop count is now locked in. Not a crash: intra-op is still
        # applied, and that's where CPU-bound ops get most of their wins.
        log.warning("cpu_threads_interop_locked", intra_op=n)
    _CONFIGURED = True
    log.info("cpu_threads_configured", intra_op=n, interop=1)
    return n
