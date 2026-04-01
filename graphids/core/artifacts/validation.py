"""Validation helpers for analysis artifact generation."""

from __future__ import annotations

from pathlib import Path


def validate_inputs(
    *,
    ckpt_path: str,
    cka: bool,
    cka_teacher_ckpt: str,
    fusion_policy: bool,
    vgae_ckpt_path: str,
    gat_ckpt_path: str,
) -> None:
    """Fail-loud validation for analyzer runtime inputs."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if cka and not cka_teacher_ckpt:
        raise ValueError("cka=true requires cka_teacher_ckpt")
    if cka and not Path(cka_teacher_ckpt).exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {cka_teacher_ckpt}")

    if fusion_policy and not vgae_ckpt_path:
        raise ValueError("fusion_policy=true requires vgae_ckpt_path")
    if fusion_policy and not gat_ckpt_path:
        raise ValueError("fusion_policy=true requires gat_ckpt_path")
