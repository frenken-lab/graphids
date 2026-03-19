"""Optuna-based HPO for KD-GAT stages.

Subprocess-based objective: each trial spawns `cli.py` for CUDA isolation.
Optuna's built-in SQLite storage provides free resume across SLURM restarts.

Usage:
    from graphids.pipeline.orchestration.optuna_sweep import run_sweep, run_sweep_pipeline
    run_sweep("autoencoder", dataset="hcrl_sa", num_samples=20)
    run_sweep_pipeline("hcrl_sa", "large", num_samples=20)
"""

from __future__ import annotations

import json
import structlog
import os
import subprocess
from pathlib import Path

import optuna
import yaml

from graphids.config import (
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    STAGE_MODEL_MAP,
    resolve,
    sweep_result_path,
)
from graphids.storage import StorageGateway
from ..subprocess_utils import build_cli_cmd

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Search spaces (loaded from YAML: graphids/config/search_spaces/)
# ---------------------------------------------------------------------------

_SEARCH_SPACES_DIR = Path(__file__).resolve().parents[2] / "config" / "search_spaces"


def _load_search_spaces() -> dict[str, dict[str, tuple]]:
    """Load search space definitions from YAML files."""
    spaces: dict[str, dict[str, tuple]] = {}
    for yaml_path in _SEARCH_SPACES_DIR.glob("*.yaml"):
        raw = yaml.safe_load(yaml_path.read_text())
        parsed: dict[str, tuple] = {}
        for param_name, spec in raw.items():
            stype = spec["type"]
            if stype == "choice":
                parsed[param_name] = ("choice", spec["values"])
            elif stype in ("uniform", "loguniform"):
                parsed[param_name] = (stype, spec["low"], spec["high"])
            else:
                raise ValueError(f"Unknown search space type '{stype}' for {param_name}")
        spaces[yaml_path.stem] = parsed
    return spaces


_SEARCH_SPACES = _load_search_spaces()

# Ordered stages for the sweep pipeline (derived from STAGE_MODEL_MAP)
_PIPELINE_STAGES = [
    (stage, model)
    for stage, model in [("autoencoder", "vgae"), ("curriculum", "gat"), ("fusion", "dqn")]
    if stage in STAGE_MODEL_MAP
]


def _suggest_params(trial: optuna.Trial, stage: str) -> list[tuple[str, str]]:
    """Suggest params from search space YAML -> Hydra override tuples."""
    model = STAGE_MODEL_MAP[stage]
    overrides: list[tuple[str, str]] = []
    for param, spec in _SEARCH_SPACES[model].items():
        if spec[0] == "loguniform":
            val = trial.suggest_float(param, spec[1], spec[2], log=True)
        elif spec[0] == "uniform":
            val = trial.suggest_float(param, spec[1], spec[2])
        elif spec[0] == "choice":
            val = trial.suggest_categorical(param, spec[1])
        else:
            raise ValueError(f"Unknown spec type: {spec[0]}")
        overrides.append((param, str(val)))
    return overrides


# ---------------------------------------------------------------------------
# Objective function (subprocess isolation preserved)
# ---------------------------------------------------------------------------


def _objective(
    trial: optuna.Trial,
    stage: str, dataset: str, scale: str,
    max_epochs: int, patience: int,
) -> float:
    """Optuna objective: suggest params -> subprocess train -> read val_loss."""
    overrides = _suggest_params(trial, stage)
    if max_epochs > 0:
        overrides.append(("training.max_epochs", str(max_epochs)))
    if patience > 0:
        overrides.append(("training.patience", str(patience)))

    model = STAGE_MODEL_MAP[stage]
    cmd = build_cli_cmd(stage, model, scale, dataset, overrides=overrides)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        log.warning("trial_failed", trial=trial.number, stderr=(result.stderr or "unknown")[-500:])
        return float("inf")

    cfg = resolve(model, scale, dataset=dataset)
    gw = StorageGateway(cfg=cfg)
    manifest_path = gw.resolve(stage, "_manifest.json")
    if manifest_path.exists():
        metrics = gw.read_json(manifest_path).get("metrics", {})
        return metrics.get("val_loss", metrics.get("best_val_loss", float("inf")))
    return float("inf")


# ---------------------------------------------------------------------------
# Warm-start + sweep DB
# ---------------------------------------------------------------------------


def _enqueue_warm_start(
    study: optuna.Study, stage: str, source_dataset: str, scale: str
) -> None:
    """Enqueue prior sweep results as initial trial for warm-starting."""
    path = sweep_result_path(stage, source_dataset, scale)
    if not path.exists():
        log.info("no_prior_sweep_results", path=str(path))
        return
    payload = yaml.safe_load(path.read_text())
    log.info("warm_starting", path=str(path), val_loss=round(payload["val_loss"], 6))
    study.enqueue_trial(payload["config"])


def _sweep_db_path() -> Path:
    """Path to Optuna SQLite database."""
    db_dir = PROJECT_ROOT / ".cache" / "kd-gat"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "optuna_sweeps.db"


# ---------------------------------------------------------------------------
# Public API: single-stage sweep
# ---------------------------------------------------------------------------


def run_sweep(
    stage: str,
    dataset: str = "hcrl_sa",
    scale: str = "large",
    *,
    num_samples: int = 20,
    max_epochs: int = 50,
    patience: int = 15,
    warm_start_from: str | None = None,
    pruner: bool = True,
) -> dict:
    """Run Optuna HPO for a single stage. Returns best params dict."""
    if stage not in STAGE_MODEL_MAP:
        raise ValueError(f"No search space defined for stage '{stage}'")

    study_name = f"{stage}_{dataset}_{scale}"
    storage = f"sqlite:///{_sweep_db_path()}"

    callbacks = []

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=(
            optuna.pruners.MedianPruner(n_startup_trials=5)
            if pruner
            else optuna.pruners.NopPruner()
        ),
    )

    if warm_start_from:
        _enqueue_warm_start(study, stage, warm_start_from, scale)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = num_samples - len(completed)

    if remaining <= 0:
        log.info("study_already_complete", study=study_name, completed_trials=len(completed))
    else:
        log.info("running_trials", remaining=remaining, study=study_name, already_completed=len(completed))
        study.optimize(
            lambda trial: _objective(trial, stage, dataset, scale, max_epochs, patience),
            n_trials=remaining,
            callbacks=callbacks,
        )

    best = study.best_trial
    _export_best_config(best, stage, dataset, scale)

    log.info("best_config", stage=stage, params=best.params, val_loss=round(best.value, 6))
    return best.params


def _export_best_config(
    best: optuna.trial.FrozenTrial, stage: str, dataset: str, scale: str
) -> None:
    """Export best trial config to YAML."""
    out_path = sweep_result_path(stage, dataset, scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage, "dataset": dataset, "scale": scale,
        "val_loss": float(best.value), "config": best.params,
    }
    out_path.write_text(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))
    log.info("best_config_saved", path=str(out_path))
    cli_parts = [f"-O {k} {v}" for k, v in sorted(best.params.items())]
    log.info("cli_overrides", overrides=" \\\n  ".join(cli_parts))


def load_best_config(stage: str, dataset: str, scale: str) -> dict:
    """Read best HP config from a completed sweep's YAML output."""
    path = sweep_result_path(stage, dataset, scale)
    if not path.exists():
        raise FileNotFoundError(f"No sweep results found at {path}")
    return yaml.safe_load(path.read_text())["config"]


# ---------------------------------------------------------------------------
# Full pipeline: sequential 3-stage sweep + train + eval
# ---------------------------------------------------------------------------


def run_sweep_pipeline(
    dataset: str,
    scale: str,
    *,
    num_samples: int = 20,
    max_epochs: int = 50,
    patience: int = 15,
    warm_start_from: str | None = None,
    dry_run: bool = False,
    multi_seed: bool = False,
) -> None:
    """Sequential sweep -> train-best -> evaluate for all 3 stages."""
    if dry_run:
        log.info("sweep_dry_run", dataset=dataset, scale=scale, stages=len(_PIPELINE_STAGES), trials_each=num_samples)
        return

    for stage, model in _PIPELINE_STAGES:
        if not sweep_result_path(stage, dataset, scale).exists():
            ws = warm_start_from
            if ws is None and dataset != "set_01":
                if sweep_result_path(stage, "set_01", scale).exists():
                    ws = "set_01"
                    log.info("auto_warm_start", stage=stage, source="set_01")
            run_sweep(stage, dataset, scale, num_samples=num_samples,
                      max_epochs=max_epochs, patience=patience, warm_start_from=ws)
        else:
            log.info("sweep_already_complete", stage=stage)

        cfg = resolve(model, scale, dataset=dataset)
        gw = StorageGateway(cfg=cfg)
        if not gw.exists(stage, "best_model.pt"):
            best_config = load_best_config(stage, dataset, scale)
            env = {**os.environ, "KD_GAT_SWEEP_ID": f"tune_{stage}_{dataset}_{scale}"}
            cmd = build_cli_cmd(stage, model, scale, dataset, overrides=list(best_config.items()))
            log.info("training_best_config", stage=stage, cmd=" ".join(cmd))
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Training step '{stage}' failed (exit {result.returncode})")
        else:
            log.info("checkpoint_exists_skipping", stage=stage)

    log.info("Running final evaluation")
    result = subprocess.run(build_cli_cmd("evaluation", "vgae", scale, dataset), cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed (exit {result.returncode})")
    log.info("sweep_pipeline_complete", dataset=dataset, scale=scale)

    if multi_seed:
        _run_multi_seed_final(dataset, scale)


def _run_multi_seed_final(dataset: str, scale: str) -> None:
    """Re-train best config with all DEFAULT_SEEDS for statistical significance."""
    log.info("multi_seed_training_start", dataset=dataset, scale=scale, seeds=DEFAULT_SEEDS)
    for stage, model in _PIPELINE_STAGES:
        config = load_best_config(stage, dataset, scale)
        for seed in DEFAULT_SEEDS:
            cmd = build_cli_cmd(stage, model, scale, dataset, seed=seed, overrides=list(config.items()))
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                raise RuntimeError(f"Multi-seed '{stage}' seed={seed} failed (exit {result.returncode})")

    for seed in DEFAULT_SEEDS:
        result = subprocess.run(build_cli_cmd("evaluation", "vgae", scale, dataset, seed=seed), cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Multi-seed evaluation seed={seed} failed")
    log.info("multi_seed_training_complete", dataset=dataset, scale=scale)
