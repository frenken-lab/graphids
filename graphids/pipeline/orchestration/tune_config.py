"""Ray Tune HPO configuration for KD-GAT stages.

Subprocess-based trainable: each trial spawns `cli.py` for CUDA isolation.

Usage:
    from graphids.pipeline.orchestration.tune_config import run_tune
    run_tune("autoencoder", dataset="hcrl_sa", num_samples=20)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search spaces per model (loaded from YAML: graphids/config/search_spaces/)
# ---------------------------------------------------------------------------

from pathlib import Path

import yaml

from graphids.config.constants import STAGE_MODEL_MAP as _STAGE_MODEL
from graphids.config.constants import SWEEP_RESULTS_DIR

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


def _build_search_space(stage: str) -> dict[str, Any]:
    """Build Ray Tune search space from declarative spec."""
    from ray import tune

    _BUILDERS = {"loguniform": tune.loguniform, "choice": tune.choice, "uniform": tune.uniform}
    return {k: _BUILDERS[t](*args) for k, (t, *args) in _SEARCH_SPACES[_STAGE_MODEL[stage]].items()}


def _build_optuna_space(stage: str) -> dict:
    """Build Optuna distributions dict (required for evaluated_rewards warm-start)."""
    import optuna

    _BUILDERS = {
        "loguniform": lambda lo, hi: optuna.distributions.FloatDistribution(lo, hi, log=True),
        "uniform": lambda lo, hi: optuna.distributions.FloatDistribution(lo, hi),
        "choice": lambda vals: optuna.distributions.CategoricalDistribution(vals),
    }
    return {k: _BUILDERS[t](*args) for k, (t, *args) in _SEARCH_SPACES[_STAGE_MODEL[stage]].items()}


def _load_warm_start_configs(
    stage: str, source_dataset: str, source_scale: str
) -> tuple[list[dict], list[float]]:
    """Load prior sweep results for warm-starting OptunaSearch.

    Returns (points_to_evaluate, evaluated_rewards) from a completed sweep's
    best config YAML.
    """
    from pathlib import Path

    import yaml

    path = Path(SWEEP_RESULTS_DIR) / f"{stage}_{source_dataset}_{source_scale}_best.yaml"
    if not path.exists():
        log.info("No prior sweep results at %s — starting cold", path)
        return [], []

    payload = yaml.safe_load(path.read_text())
    config = payload["config"]
    val_loss = payload["val_loss"]

    log.info("Warm-starting from %s (val_loss=%.6f)", path, val_loss)
    return [config], [val_loss]


# ---------------------------------------------------------------------------
# Trainable function (subprocess-based, like the pipeline)
# ---------------------------------------------------------------------------


def _trainable(
    config: dict, stage: str, dataset: str, scale: str, max_epochs: int = 0, patience: int = 0
) -> None:
    """Ray Tune trainable that runs a pipeline stage as subprocess.

    Reports val_loss from the stage's metrics.json.
    """
    import json
    from pathlib import Path

    from ray import tune as ray_tune

    # Project root — subprocess must run from here for relative paths
    project_root = Path(__file__).resolve().parents[3]

    model = _STAGE_MODEL[stage]

    # Build CLI overrides from tune config
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
    for key, value in config.items():
        cmd.extend(["-O", key, str(value)])

    # Inject epoch/patience overrides for shorter tune trials
    if max_epochs > 0:
        cmd.extend(["-O", "training.max_epochs", str(max_epochs)])
    if patience > 0:
        cmd.extend(["-O", "training.patience", str(patience)])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    if result.returncode != 0:
        err_tail = result.stderr[-1000:] if result.stderr else "unknown"
        log.warning("Trial failed (exit %d): %s", result.returncode, err_tail)
        # Print to stdout so it appears in SLURM logs
        print(f"[TRIAL FAILED] exit={result.returncode}\n{err_tail}", flush=True)
        ray_tune.report({"val_loss": float("inf")})
        return

    # Read metrics from the stage output (use absolute path — Ray worker cwd differs)
    from graphids.config import stage_dir
    from graphids.config.resolver import resolve

    overrides = {"dataset": dataset}
    cfg = resolve(model, scale, **overrides)
    mpath = project_root / stage_dir(cfg, stage) / "metrics.json"

    if mpath.exists():
        metrics = json.loads(mpath.read_text())
        val_loss = metrics.get("val_loss", metrics.get("best_val_loss", float("inf")))
        ray_tune.report({"val_loss": val_loss})
    else:
        ray_tune.report({"val_loss": float("inf")})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tune(
    stage: str,
    dataset: str = "hcrl_sa",
    scale: str = "large",
    num_samples: int = 20,
    max_concurrent: int = 1,
    metric: str = "val_loss",
    mode: str = "min",
    grace_period: int = 0,
    local: bool = False,
    max_epochs: int = 0,
    patience: int = 0,
    warm_start_from: str | None = None,
) -> Any:
    """Run Ray Tune HPO for a pipeline stage.

    Parameters
    ----------
    stage : str
        Pipeline stage (autoencoder, curriculum, normal, fusion).
    dataset, scale : str
        Dataset name and model scale.
    num_samples, max_concurrent : int
        Number of HPO trials and max concurrent trials.
    metric, mode : str
        Metric to optimize and direction ("min" or "max").
    grace_period : int
        ASHA grace period (epochs before early stopping a trial).
    local : bool
        Use Ray local mode.
    max_epochs, patience : int
        Override training.max_epochs/patience per trial (0 = use config default).
    warm_start_from : str | None
        Dataset name to warm-start from (loads prior sweep results for OptunaSearch).
    """
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    from .ray_slurm import ray_init_kwargs

    if stage not in _STAGE_MODEL:
        raise ValueError(f"No search space defined for stage '{stage}'")

    # Load config for tune defaults (grace_period, reduction_factor)
    from graphids.config.resolver import resolve as _resolve

    cfg = _resolve(_STAGE_MODEL[stage], scale, dataset=dataset)
    if grace_period <= 0:
        grace_period = cfg.tune.grace_period

    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=max_epochs if max_epochs > 0 else 200,
        grace_period=grace_period,
        reduction_factor=cfg.tune.reduction_factor,
    )

    # Warm-start: seed OptunaSearch with prior results if available
    points, rewards = [], []
    if warm_start_from:
        points, rewards = _load_warm_start_configs(stage, warm_start_from, scale)

    if points and rewards:
        optuna_space = _build_optuna_space(stage)
        search_alg = OptunaSearch(
            space=optuna_space,
            metric=metric,
            mode=mode,
            points_to_evaluate=points,
            evaluated_rewards=rewards,
        )
        search_space: dict[str, Any] = {}  # space is in the searcher
    else:
        search_alg = OptunaSearch(metric=metric, mode=mode)
        search_space = _build_search_space(stage)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                _trainable,
                stage=stage,
                dataset=dataset,
                scale=scale,
                max_epochs=max_epochs,
                patience=patience,
            ),
            resources={"gpu": 1},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name=f"tune_{stage}_{dataset}_{scale}",
            storage_path=str(Path(__file__).resolve().parents[3] / "ray_results"),
        ),
    )

    results = tuner.fit()

    # Save searcher state (fitted Optuna TPE model) for future warm-starts
    from pathlib import Path

    searcher_path = Path(SWEEP_RESULTS_DIR) / f"{stage}_{dataset}_{scale}_searcher.pkl"
    searcher_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        search_alg.save(str(searcher_path))
        log.info("Searcher state saved to %s", searcher_path)
    except Exception as e:
        log.warning("Failed to save searcher state: %s", e)

    best = results.get_best_result(metric=metric, mode=mode)
    log.info(
        "Best config for %s: %s (val_loss=%.6f)",
        stage,
        best.config,
        best.metrics.get(metric, float("inf")),
    )

    export_best_config(best, stage, dataset, scale)

    # Log sweep summary to MLflow as a parent run
    try:
        import mlflow

        from graphids.config import MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"kd-gat-sweep-{stage}")
        with mlflow.start_run(
            run_name=f"sweep_{stage}_{dataset}_{scale}",
            tags={
                "stage": stage,
                "dataset": dataset,
                "scale": scale,
                "num_samples": str(num_samples),
                "trainable_mode": trainable_label,
            },
        ):
            mlflow.log_metrics(
                {
                    "best_val_loss": float(best.metrics.get(metric, float("inf"))),
                    "num_trials": len(results),
                    "num_errors": sum(1 for r in results if r.error),
                }
            )
            mlflow.log_params(best.config)
            # Log best config YAML as artifact
            best_yaml = Path(SWEEP_RESULTS_DIR) / f"{stage}_{dataset}_{scale}_best.yaml"
            if best_yaml.exists():
                mlflow.log_artifact(str(best_yaml))
    except Exception as e:
        log.warning("MLflow sweep logging failed (non-fatal): %s", e)

    # Shutdown Ray so the caller (e.g. sweep_pipeline) can reinitialize cleanly
    ray.shutdown()

    return results


def export_best_config(best_result, stage: str, dataset: str, scale: str) -> None:
    """Export best trial config to YAML and print CLI override flags."""
    from pathlib import Path

    import yaml

    out_dir = Path(SWEEP_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = best_result.config
    val_loss = best_result.metrics.get("val_loss", float("inf"))

    # Write YAML
    out_path = out_dir / f"{stage}_{dataset}_{scale}_best.yaml"
    payload = {
        "stage": stage,
        "dataset": dataset,
        "scale": scale,
        "val_loss": float(val_loss),
        "config": config,
    }
    out_path.write_text(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))
    log.info("Best config saved to %s", out_path)

    # Print CLI override flags for easy copy-paste into Phase B
    cli_parts = []
    for key, value in sorted(config.items()):
        cli_parts.append(f"-O {key} {value}")
    log.info("CLI overrides for Phase B:\n  %s", " \\\n  ".join(cli_parts))
