"""Execution helpers for orchestrated training assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphids.config import CKPT_SUBPATH, COMPLETE_MARKER, LAST_CKPT_SUBPATH, run_dir
from graphids.core.contracts import TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.slurm import sacct_query


def artifact_paths(
    cfg: StageConfig,
    *,
    lake_root: str,
    user: str,
    dataset: str,
    seed: int,
) -> tuple[str, Path, Path, Path]:
    """Build run directory, checkpoint, and completion marker paths for one partition."""
    rd = run_dir(
        lake_root,
        user,
        dataset,
        cfg.model_type,
        cfg.scale,
        cfg.stage,
        cfg.identity,
        cfg.kd_tag,
        seed,
    )
    rd_path = Path(rd)
    ckpt_file = rd_path / CKPT_SUBPATH
    complete = rd_path / COMPLETE_MARKER
    return rd, rd_path, ckpt_file, complete


def training_spec(
    cfg: StageConfig,
    *,
    dataset: str,
    seed: int,
    run_directory: str,
    run_directory_path: Path,
    upstream_ckpts: dict[str, str],
) -> TrainingSpec:
    """Create one training contract spec, including optional resume checkpoint."""
    spec = TrainingSpec(
        stage=cfg.stage,
        model_family=cfg.model_type,
        scale=cfg.scale,
        dataset=dataset,
        seed=seed,
        run_dir=run_directory,
        config_files=cfg.config_files,
        model_init_overrides=cfg.model_init_overrides,
        upstream_ckpt_paths=upstream_ckpts,
        upstream_model_families=cfg.upstream_model_families,
    )
    if cfg.kd_overrides:
        import json
        spec = spec.model_copy(update={
            "runtime_overrides": {
                **spec.runtime_overrides,
                "model.init_args.auxiliaries": json.dumps([cfg.kd_overrides]),
            }
        })

    resume = run_directory_path / LAST_CKPT_SUBPATH
    if not resume.exists():
        return spec

    return spec.model_copy(
        update={
            "runtime_overrides": {
                **spec.runtime_overrides,
                "ckpt_path": str(resume),
            }
        }
    )


def slurm_accounting_metadata(job_id: int) -> dict[str, Any]:
    """Extract wall time and peak RSS from sacct output."""
    out = sacct_query([job_id], "JobID,Elapsed,MaxRSS", units="G")
    wall, rss = "", ""
    if out:
        for line in out.strip().split("\n"):
            fields = line.split("|")
            if len(fields) < 3:
                continue
            jid_field = fields[0].strip()
            if "." not in jid_field:
                wall = fields[1].strip()
            elif jid_field.endswith(".batch"):
                rss = fields[2].strip()
    return {"job_id": job_id, "wall_time": wall, "peak_rss": rss}
