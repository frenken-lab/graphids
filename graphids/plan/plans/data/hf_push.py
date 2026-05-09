"""Push teacher checkpoints, cache, states, logs, and analysis to HuggingFace Hub.

Prerequisites:
  - GRAPHIDS_LAKE_ROOT must be set (cache_dir/lake_root called at render time)
  - HF_TOKEN must be set in .env (sourced by the SLURM node at exec time)
  - Analysis push rows require analyze rows to have completed first

Usage:
    gx run data.hf_push -d hcrl_sa -s 42 -o rendered/hcrl_sa/data/hf_push/seed_42.json
    gx plans submit --plan rendered/hcrl_sa/data/hf_push/seed_42.json -C pitzer --filter 'push_ckpt*'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphids.paths import (
    PREPROCESSING_VERSION,
    best_ckpt,
    cache_dir,
    lake_root,
    run_dir,
    states_dir,
)
from graphids.plan import hf_push

CHECKPOINT_REPO = "buckeyeguy/graphids-checkpoints"
DATA_REPO = "buckeyeguy/graphids-data"

_TEACHER_VARIANTS = [
    ("teacher", "teacher_vgae"),
    ("teacher", "teacher_gat"),
]

_ABLATION_GAT_VARIANTS = [
    ("gat_loss", "ce"),
    ("gat_loss", "focal"),
    ("gat_loss", "weighted_ce"),
    ("gat_sampling", "curriculum_random"),
    ("gat_sampling", "curriculum_vgae"),
    ("gat_sampling", "none"),
    ("id_encoding", "hash"),
    ("id_encoding", "lookup"),
]


def _log_path(dataset: str, group: str, variant: str, seed: int) -> str:
    scripts_dir = Path(run_dir(dataset, group, variant, seed)) / ".slurm_scripts"
    logs = sorted(scripts_dir.glob("*.stderr"))
    if not logs:
        raise FileNotFoundError(f"No .stderr log found in {scripts_dir}")
    return str(logs[-1])


def _analysis_dir(dataset: str, group: str, variant: str, seed: int) -> str:
    return str(Path(run_dir(dataset, group, variant, seed)) / "artifacts")


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for group, variant in _TEACHER_VARIANTS:
        ckpt_dir = str(Path(best_ckpt(dataset, group, variant, seed)).parent)

        rows.append(
            hf_push(
                name=f"push_ckpt_{variant}_{dataset}",
                artifact_type="checkpoints",
                repo_id=CHECKPOINT_REPO,
                repo_type="model",
                local_path=ckpt_dir,
                path_in_repo=f"{dataset}/{variant}/seed_{seed}",
            )
        )

        rows.append(
            hf_push(
                name=f"push_log_{variant}_{dataset}",
                artifact_type="logs",
                repo_id=DATA_REPO,
                repo_type="dataset",
                local_path=_log_path(dataset, group, variant, seed),
                path_in_repo=f"logs/{dataset}/{group}/{variant}/seed_{seed}/job.stderr",
            )
        )

        rows.append(
            hf_push(
                name=f"push_analysis_{variant}_{dataset}",
                artifact_type="analysis",
                repo_id=DATA_REPO,
                repo_type="dataset",
                local_path=_analysis_dir(dataset, group, variant, seed),
                path_in_repo=f"analysis/{dataset}/{group}/{variant}/seed_{seed}",
            )
        )

    for group, variant in _ABLATION_GAT_VARIANTS:
        rows.append(
            hf_push(
                name=f"push_analysis_{variant}_{dataset}",
                artifact_type="analysis",
                repo_id=DATA_REPO,
                repo_type="dataset",
                local_path=_analysis_dir(dataset, group, variant, seed),
                path_in_repo=f"analysis/{dataset}/{group}/{variant}/seed_{seed}",
            )
        )

    rows.append(
        hf_push(
            name=f"push_cache_{dataset}",
            artifact_type="cache",
            repo_id=DATA_REPO,
            repo_type="dataset",
            local_path=str(cache_dir(lake_root(), dataset)),
            path_in_repo=f"cache/v{PREPROCESSING_VERSION}/{dataset}",
            length="long",
        )
    )

    rows.append(
        hf_push(
            name=f"push_states_kd_{dataset}",
            artifact_type="states",
            repo_id=DATA_REPO,
            repo_type="dataset",
            local_path=str(states_dir(dataset, seed, "kd")),
            path_in_repo=f"states/{dataset}/kd/seed_{seed}",
        )
    )

    return rows
