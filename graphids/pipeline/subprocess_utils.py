"""Shared CLI command builder for subprocess dispatch.

All subprocess-based stage invocations go through build_cli_cmd() to ensure
consistent argument formatting. The returned list is suitable for subprocess.run()
or subprocess.Popen(). SLURM wrappers join the result into a string and wrap
with sbatch separately.

Emits Hydra override grammar: ``stage=X model=Y dataset=Z key=value``.
"""

from __future__ import annotations

import sys


def build_cli_cmd(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    *,
    seed: int | None = None,
    auxiliaries: str = "none",
    overrides: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Build a CLI command list for ``python -m graphids.cli``.

    Emits Hydra override grammar (``key=value``).

    Parameters
    ----------
    stage : str
        Pipeline stage (autoencoder, curriculum, fusion, evaluation, ...).
    model : str
        Model type (vgae, gat, dqn, ...).
    scale : str
        Scale variant (large, small).
    dataset : str
        Dataset name.
    seed : int | None
        Single seed value.
    auxiliaries : str
        Auxiliary loss modifier (default ``"none"``).
    overrides : list[tuple[str, str]] | None
        Config overrides as ``[("key", "value"), ...]`` pairs.

    Returns
    -------
    list[str]
        Command list ready for ``subprocess.run(cmd)``.
    """
    cmd = [
        sys.executable,
        "-m",
        "graphids.cli",
        f"stage={stage}",
    ]

    # Compound model name for Hydra config group (e.g., model=vgae_large)
    if model and scale:
        cmd.append(f"model={model}_{scale}")

    cmd.append(f"dataset={dataset}")

    if auxiliaries != "none":
        cmd.append(f"auxiliary={auxiliaries}")

    if seed is not None:
        cmd.append(f"seed={seed}")

    if overrides:
        for key, value in overrides:
            cmd.append(f"{key}={value}")

    return cmd
