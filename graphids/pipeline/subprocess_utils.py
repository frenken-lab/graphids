"""Shared CLI command builder for subprocess dispatch.

All subprocess-based stage invocations go through build_cli_cmd() to ensure
consistent argument formatting. The returned list is suitable for subprocess.run()
or subprocess.Popen(). SLURM wrappers (coordinator) join the result into a string
and wrap with sbatch separately.
"""

from __future__ import annotations

import sys


def build_cli_cmd(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    seed: int | None = None,
    seeds: str | None = None,
    auxiliaries: str = "none",
    overrides: list[tuple[str, str]] | None = None,
    sweep_id: str | None = None,
    ckpt_path: str | None = None,
) -> list[str]:
    """Build a CLI command list for ``python -m graphids.pipeline.cli``.

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
        Single seed value (``--seed``).
    seeds : str | None
        Seeds string for multi-seed (``--seeds``).
    auxiliaries : str
        Auxiliary loss modifier (default ``"none"``).
    overrides : list[tuple[str, str]] | None
        Config overrides as ``[("-O", "key", "value"), ...]`` style pairs.
        Each tuple is ``(key, value)`` and is emitted as ``-O key value``.
    sweep_id : str | None
        Sweep identifier (``--sweep-id``).

    Returns
    -------
    list[str]
        Command list ready for ``subprocess.run(cmd)``.
    """
    cmd = [
        sys.executable,
        "-m",
        "graphids.pipeline.cli",
        stage,
        "--model",
        model,
        "--scale",
        scale,
        "--dataset",
        dataset,
    ]

    if auxiliaries != "none":
        cmd.extend(["--auxiliaries", auxiliaries])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if seeds is not None:
        cmd.extend(["--seeds", str(seeds)])

    if overrides:
        for key, value in overrides:
            cmd.extend(["-O", key, str(value)])

    if sweep_id is not None:
        cmd.extend(["--sweep-id", sweep_id])

    if ckpt_path is not None:
        cmd.extend(["--ckpt-path", str(ckpt_path)])

    return cmd
