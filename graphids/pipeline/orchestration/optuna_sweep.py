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
import logging
import os
import subprocess
from pathlib import Path

import optuna
import yaml

from graphids.config import (
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    STAGE_MODEL_MAP,
    checkpoint_path,
    resolve,
    stage_dir,
    sweep_result_path,
)
from graphids.pipeline.subprocess_utils import build_cli_cmd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search spaces (loaded from YAML: graphids/config/search_spaces/)
# ---------------------------------------------------------------------------

_SEARCH_SPACES_DIR = Path(__file__).resolve().parents[2] / "config" / "search_spaces"


def _load_search_spaces() -> dict[str, dict[str, tuple]]:
    """Load search space definitions from YAML files."""
    spaces: dict[str, dict[str, tuple]] = {}
    for yaml_path in _SEARCH_SPACES_DIR.glob("*.yaml"):
        raw = yaml.safe_load(yaml_path.read_text())
        model = yaml_path.stem
        parsed: dict[str, tuple] = {}
        for param_name, spec in raw.items():
            stype = spec["type"]
            if stype == "choice":
                parsed[param_name] = ("choice", spec["values"])
            elif stype in ("uniform", "loguniform"):
                parsed[param_name] = (stype, spec["low"], spec["high"])
            else:
                raise ValueError(
                    f"Unknown search space type '{stype}' for {param_name} in {yaml_path}"
                )
        spaces[model] = parsed
    return spaces


_SEARCH_SPACES = _load_search_spaces()


def _suggest_params(trial: optuna.Trial, stage: str) -> list[tuple[str, str]]:
    """Suggest params from search space YAML → Hydra override tuples."""
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
    stage: str,
    dataset: str,
    scale: str,
    max_epochs: int,
    patience: int,
) -> float:
    """Optuna objective: suggest params → subprocess train → read val_loss."""
    overrides = _suggest_params(trial, stage)
    if max_epochs > 0:
        overrides.append(("training.max_epochs", str(max_epochs)))
    if patience > 0:
        overrides.append(("training.patience", str(patience)))

    model = STAGE_MODEL_MAP[stage]
    cmd = build_cli_cmd(stage, model, scale, dataset, overrides=overrides)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        err_tail = result.stderr[-500:] if result.stderr else "unknown"
        log.warning("Trial %d failed: %s", trial.number, err_tail)
        return float("inf")

    # Read val_loss from _manifest.json (single source of truth)
    cfg = resolve(model, scale, dataset=dataset)
    manifest = PROJECT_ROOT / stage_dir(cfg, stage) / "_manifest.json"

    if manifest.exists():
        data = json.loads(manifest.read_text())
        metrics = data.get("metrics", {})
        return metrics.get("val_loss", metrics.get("best_val_loss", float("inf")))

    return float("inf")


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------


def _enqueue_warm_start(
    study: optuna.Study, stage: str, source_dataset: str, scale: str
) -> None:
    """Enqueue prior sweep results as initial trial for warm-starting."""
    path = sweep_result_path(stage, source_dataset, scale)
    if not path.exists():
        log.info("No prior sweep results at %s — starting cold", path)
        return

    payload = yaml.safe_load(path.read_text())
    config = payload["config"]
    log.info("Warm-starting from %s (val_loss=%.6f)", path, payload["val_loss"])
    study.enqueue_trial(config)


# ---------------------------------------------------------------------------
# Sweep DB path
# ---------------------------------------------------------------------------


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

    # Warm-start: enqueue known-good configs from prior dataset
    if warm_start_from:
        _enqueue_warm_start(study, stage, warm_start_from, scale)

    # Skip already-completed trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = num_samples - len(completed)

    if remaining <= 0:
        log.info("Study %s already has %d completed trials, skipping", study_name, len(completed))
    else:
        log.info(
            "Running %d trials for %s (%d already completed)",
            remaining,
            study_name,
            len(completed),
        )
        study.optimize(
            lambda trial: _objective(trial, stage, dataset, scale, max_epochs, patience),
            n_trials=remaining,
        )

    best = study.best_trial
    _export_best_config(best, stage, dataset, scale)
    _log_sweep_to_mlflow(study, stage, dataset, scale)

    log.info(
        "Best config for %s: %s (val_loss=%.6f)", stage, best.params, best.value
    )
    return best.params


# ---------------------------------------------------------------------------
# Export + MLflow logging
# ---------------------------------------------------------------------------


def _export_best_config(
    best: optuna.trial.FrozenTrial, stage: str, dataset: str, scale: str
) -> None:
    """Export best trial config to YAML."""
    out_path = sweep_result_path(stage, dataset, scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "dataset": dataset,
        "scale": scale,
        "val_loss": float(best.value),
        "config": best.params,
    }
    out_path.write_text(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))
    log.info("Best config saved to %s", out_path)

    # Print CLI override flags for easy copy-paste
    cli_parts = [f"-O {key} {value}" for key, value in sorted(best.params.items())]
    log.info("CLI overrides:\n  %s", " \\\n  ".join(cli_parts))


def _log_sweep_to_mlflow(
    study: optuna.Study, stage: str, dataset: str, scale: str
) -> None:
    """Log sweep summary to MLflow as a parent run."""
    try:
        import mlflow

        from graphids.config import MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"kd-gat-sweep-{stage}")

        trials = study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed = [t for t in trials if t.state != optuna.trial.TrialState.COMPLETE]

        with mlflow.start_run(
            run_name=f"sweep_{stage}_{dataset}_{scale}",
            tags={
                "stage": stage,
                "dataset": dataset,
                "scale": scale,
                "num_samples": str(len(trials)),
                "backend": "optuna",
            },
        ):
            mlflow.log_metrics({
                "best_val_loss": float(study.best_value),
                "num_trials": len(completed),
                "num_errors": len(failed),
            })
            mlflow.log_params(study.best_params)

            best_yaml = sweep_result_path(stage, dataset, scale)
            if best_yaml.exists():
                mlflow.log_artifact(str(best_yaml))
    except Exception as e:
        log.warning("MLflow sweep logging failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Best config loading
# ---------------------------------------------------------------------------


def load_best_config(stage: str, dataset: str, scale: str) -> dict:
    """Read best HP config from a completed sweep's YAML output."""
    path = sweep_result_path(stage, dataset, scale)
    if not path.exists():
        raise FileNotFoundError(f"No sweep results found at {path}")
    payload = yaml.safe_load(path.read_text())
    return payload["config"]


# ---------------------------------------------------------------------------
# Full pipeline: sequential 3-stage sweep + train + eval
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    ("autoencoder", "vgae"),
    ("curriculum", "gat"),
    ("fusion", "dqn"),
]


def _verify_step_output(kind: str, stage: str, model: str, dataset: str, scale: str) -> bool:
    """Check if a step's expected output exists."""
    if kind == "sweep":
        return sweep_result_path(stage, dataset, scale).exists()
    elif kind == "train":
        cfg = resolve(model, scale, dataset=dataset)
        return checkpoint_path(cfg, stage).exists()
    elif kind == "evaluate":
        cfg = resolve("vgae", scale, dataset=dataset)
        return (stage_dir(cfg, "evaluation") / "_manifest.json").exists()
    return False


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
    """Sequential sweep → train-best → evaluate for all 3 stages."""
    if dry_run:
        _dry_run(dataset, scale)
        return

    for stage, model in PIPELINE_STAGES:
        # --- Sweep (skip if results exist) ---
        if not sweep_result_path(stage, dataset, scale).exists():
            # Auto-warm-start from set_01 if sweeping a different dataset
            ws = warm_start_from
            if ws is None and dataset != "set_01":
                ref_path = sweep_result_path(stage, "set_01", scale)
                if ref_path.exists():
                    ws = "set_01"
                    log.info("Auto warm-starting %s from set_01 results", stage)

            run_sweep(
                stage,
                dataset,
                scale,
                num_samples=num_samples,
                max_epochs=max_epochs,
                patience=patience,
                warm_start_from=ws,
            )
        else:
            log.info("Sweep for %s already complete, skipping", stage)

        # --- Train best config (skip if checkpoint exists) ---
        cfg = resolve(model, scale, dataset=dataset)
        if not checkpoint_path(cfg, stage).exists():
            best_config = load_best_config(stage, dataset, scale)
            overrides = list(best_config.items())
            env = {**os.environ, "KD_GAT_SWEEP_ID": f"tune_{stage}_{dataset}_{scale}"}
            cmd = build_cli_cmd(stage, model, scale, dataset, overrides=overrides)
            log.info("Training best %s: %s", stage, " ".join(cmd))
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Training step '{stage}' failed with exit code {result.returncode}"
                )
        else:
            log.info("Checkpoint for %s already exists, skipping training", stage)

    # --- Final evaluation ---
    log.info("Running final evaluation")
    cmd = build_cli_cmd("evaluation", "vgae", scale, dataset)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}")

    log.info("Sweep pipeline complete for %s/%s", dataset, scale)

    if multi_seed:
        _run_multi_seed_final(dataset, scale)


# ---------------------------------------------------------------------------
# Multi-seed final training
# ---------------------------------------------------------------------------


def _run_multi_seed_final(dataset: str, scale: str) -> None:
    """Re-train best config with all DEFAULT_SEEDS for statistical significance."""
    log.info("=== Multi-seed training for %s/%s with seeds %s ===", dataset, scale, DEFAULT_SEEDS)

    for stage, model in PIPELINE_STAGES:
        config = load_best_config(stage, dataset, scale)
        for seed in DEFAULT_SEEDS:
            cmd = build_cli_cmd(stage, model, scale, dataset, seed=seed, overrides=list(config.items()))
            log.info("Multi-seed %s (seed=%d): %s", stage, seed, " ".join(cmd))
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Multi-seed training '{stage}' seed={seed} failed "
                    f"with exit code {result.returncode}"
                )

    # Final multi-seed evaluation
    for seed in DEFAULT_SEEDS:
        cmd = build_cli_cmd("evaluation", "vgae", scale, dataset, seed=seed)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Multi-seed evaluation seed={seed} failed")

    log.info("=== Multi-seed training complete for %s/%s ===", dataset, scale)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(dataset: str, scale: str) -> None:
    """Print pipeline state and verify configs without executing."""
    from graphids.config import data_dir
    from graphids.pipeline.validate import validate_datasets

    log.info("=== Sweep Pipeline Dry Run: dataset=%s, scale=%s ===", dataset, scale)

    ds_errors = validate_datasets([dataset], scale)
    if ds_errors:
        for err in ds_errors:
            log.error("Validation FAILED: %s", err)
        return

    cfg = resolve("vgae", scale, dataset=dataset)
    ddir = data_dir(cfg)
    log.info("Config resolution: OK")
    log.info("Data directory: %s (exists=%s)", ddir, ddir.exists())

    log.info("")
    log.info("Pipeline Steps:")
    step_num = 0
    for stage, model in PIPELINE_STAGES:
        # Sweep step
        step_num += 1
        sweep_exists = sweep_result_path(stage, dataset, scale).exists()
        log.info(
            "  [%d] sweep_%-12s  output=%s",
            step_num,
            stage,
            "exists" if sweep_exists else "missing",
        )

        # Train step
        step_num += 1
        step_cfg = resolve(model, scale, dataset=dataset)
        ckpt_exists = checkpoint_path(step_cfg, stage).exists()
        log.info(
            "  [%d] train_%-12s  output=%s",
            step_num,
            stage,
            "exists" if ckpt_exists else "missing",
        )

    # Eval step
    step_num += 1
    eval_cfg = resolve("vgae", scale, dataset=dataset)
    eval_exists = (stage_dir(eval_cfg, "evaluation") / "_manifest.json").exists()
    log.info(
        "  [%d] evaluate          output=%s",
        step_num,
        "exists" if eval_exists else "missing",
    )

    log.info("")
    log.info("=== Dry run complete ===")
