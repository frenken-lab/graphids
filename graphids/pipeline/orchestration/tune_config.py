"""Ray Tune HPO configuration for KD-GAT stages.

Replaces scripts/generate_sweep.py parallel-command approach with
Ray Tune + OptunaSearch + ASHAScheduler for efficient hyperparameter search.

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
# Search spaces per model (declarative: type + args → Ray Tune sampler)
# ---------------------------------------------------------------------------

_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "vgae": {
        "training.lr": ("loguniform", 1e-4, 1e-2),
        "training.weight_decay": ("loguniform", 1e-6, 1e-3),
        "vgae.latent_dim": ("choice", [16, 32, 48, 64]),
        "vgae.dropout": ("uniform", 0.05, 0.4),
        "vgae.heads": ("choice", [1, 2, 4, 8]),
        "vgae.embedding_dim": ("choice", [8, 16, 32]),
        "vgae.proj_dim": ("choice", [32, 48, 64]),
    },
    "gat": {
        "training.lr": ("loguniform", 1e-4, 1e-2),
        "training.weight_decay": ("loguniform", 1e-6, 1e-3),
        "gat.hidden": ("choice", [32, 48, 64, 96]),
        "gat.layers": ("choice", [2, 3, 4]),
        "gat.heads": ("choice", [4, 8]),
        "gat.dropout": ("uniform", 0.1, 0.4),
        "gat.embedding_dim": ("choice", [8, 16, 32]),
        "gat.fc_layers": ("choice", [2, 3, 4]),
        "gat.proj_dim": ("choice", [32, 48, 64]),
    },
    "dqn": {
        "fusion.lr": ("loguniform", 1e-4, 1e-2),
        "dqn.hidden": ("choice", [256, 512, 576, 768]),
        "dqn.layers": ("choice", [2, 3, 4]),
        "dqn.gamma": ("uniform", 0.95, 0.999),
        "dqn.epsilon": ("uniform", 0.05, 0.2),
        "dqn.epsilon_decay": ("uniform", 0.99, 0.999),
        "fusion.episodes": ("choice", [300, 500, 750]),
    },
}

_STAGE_MODEL = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}


def _build_search_space(stage: str) -> dict[str, Any]:
    """Build Ray Tune search space from declarative spec."""
    from ray import tune

    _BUILDERS = {"loguniform": tune.loguniform, "choice": tune.choice, "uniform": tune.uniform}
    return {k: _BUILDERS[t](*args) for k, (t, *args) in _SEARCH_SPACES[_STAGE_MODEL[stage]].items()}


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
# Dry-run validation (login node safe, no GPU required)
# ---------------------------------------------------------------------------


def dry_run_tune(
    stage: str,
    dataset: str = "hcrl_sa",
    scale: str = "large",
    num_samples: int = 20,
    max_concurrent: int = 1,
    max_epochs: int = 0,
    patience: int = 0,
) -> bool:
    """Validate the full tune setup without actually running trials.

    Checks: imports, config resolution, search space construction, Tuner
    instantiation, subprocess command construction, and data path resolution.
    Safe to run on login nodes (no GPU required).

    Returns True if all checks pass, raises on failure.
    """
    from pathlib import Path

    checks_passed = 0
    total_checks = 0

    def _check(name: str, fn):
        nonlocal checks_passed, total_checks
        total_checks += 1
        try:
            result = fn()
            log.info("  [PASS] %s", name)
            checks_passed += 1
            return result
        except Exception as e:
            log.error("  [FAIL] %s: %s", name, e)
            raise

    log.info(
        "=== Dry-run validation for tune: stage=%s, dataset=%s, scale=%s ===", stage, dataset, scale
    )

    # 1. Stage validation
    _check("Stage in search space", lambda: _STAGE_MODEL[stage])
    model = _STAGE_MODEL[stage]

    # 2. Config resolution
    from graphids.config.resolver import resolve

    cfg = _check("Config resolution", lambda: resolve(model, scale, dataset=dataset))

    # 3. Data directory exists
    from graphids.config import data_dir

    def _check_data():
        d = data_dir(cfg)
        if not d.exists():
            raise FileNotFoundError(f"Data directory not found: {d}")
        return d

    _check("Data directory exists", _check_data)

    # 4. Search space builds
    _check("Search space construction", lambda: _build_search_space(stage))

    # 5. Ray imports
    def _check_ray_imports():
        from ray import tune  # noqa: F811
        from ray.tune.schedulers import ASHAScheduler  # noqa: F401
        from ray.tune.search.optuna import OptunaSearch  # noqa: F401

        return tune

    tune = _check("Ray Tune imports", _check_ray_imports)

    # 6. Tuner construction (no fit)
    def _check_tuner():
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        search_space = _build_search_space(stage)
        scheduler = ASHAScheduler(
            metric="val_loss", mode="min", grace_period=10, reduction_factor=3
        )
        search_alg = OptunaSearch(metric="val_loss", mode="min")

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
            ),
        )
        return tuner

    _check("Tuner construction", _check_tuner)

    # 7. Subprocess command construction
    def _check_subprocess_cmd():
        project_root = Path(__file__).resolve().parents[3]
        sample_config = {"training.lr": 0.001, f"{model}.dropout": 0.2}
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
        for key, value in sample_config.items():
            cmd.extend(["-O", key, str(value)])
        if max_epochs > 0:
            cmd.extend(["-O", "training.max_epochs", str(max_epochs)])
        if patience > 0:
            cmd.extend(["-O", "training.patience", str(patience)])

        # Verify project root exists and has the expected structure
        if not (project_root / "graphids").exists():
            raise FileNotFoundError(f"Project root missing graphids/: {project_root}")
        return {"cmd": cmd, "cwd": str(project_root)}

    cmd_info = _check("Subprocess command", _check_subprocess_cmd)

    log.info("=== Dry-run complete: %d/%d checks passed ===", checks_passed, total_checks)
    log.info(
        "Subprocess command preview:\n  %s\n  cwd=%s",
        " \\\n    ".join(cmd_info["cmd"]),
        cmd_info["cwd"],
    )
    return True


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
    grace_period: int = 10,
    local: bool = False,
    max_epochs: int = 0,
    patience: int = 0,
) -> Any:
    """Run Ray Tune HPO for a pipeline stage.

    Parameters
    ----------
    stage : str
        Pipeline stage (autoencoder, curriculum, normal, fusion).
    dataset : str
        Dataset name.
    scale : str
        Model scale (large, small).
    num_samples : int
        Number of HPO trials.
    max_concurrent : int
        Max concurrent trials (limited by GPU count).
    metric : str
        Metric to optimize.
    mode : str
        "min" or "max".
    grace_period : int
        ASHA grace period (epochs before early stopping a trial).
    local : bool
        Use Ray local mode.
    max_epochs : int
        Override training.max_epochs per trial (0 = use config default).
    patience : int
        Override training.patience per trial (0 = use config default).

    Returns
    -------
    ray.tune.ResultGrid
        Tune results with best config accessible via result.get_best_result().
    """
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    from .ray_slurm import ray_init_kwargs

    if stage not in _STAGE_MODEL:
        raise ValueError(f"No search space defined for stage '{stage}'")

    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    search_space = _build_search_space(stage)

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        grace_period=grace_period,
        reduction_factor=3,
    )

    search_alg = OptunaSearch(metric=metric, mode=mode)

    # Note: Ray 2.54+ RunConfig only accepts ray.train.UserCallback instances,
    # not ray.tune callbacks (TBXLoggerCallback, WandbLoggerCallback).
    # W&B logging is handled inside the trainable via the CLI's W&B init.

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
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric=metric, mode=mode)
    log.info(
        "Best config for %s: %s (val_loss=%.6f)",
        stage,
        best.config,
        best.metrics.get(metric, float("inf")),
    )

    export_best_config(best, stage, dataset, scale)

    # Auto-ingest sweep results to lakehouse + HF Dataset
    try:
        from pathlib import Path

        from graphids.pipeline.sweep_export import ingest_and_push

        experiment_dir = Path(results.experiment_path)
        ingest_and_push(experiment_dir, stage, dataset, scale)
    except Exception as e:
        log.warning("Sweep export failed (non-fatal): %s", e)

    # Shutdown Ray so the caller (e.g. sweep_pipeline) can reinitialize cleanly
    ray.shutdown()

    return results


def export_best_config(best_result, stage: str, dataset: str, scale: str) -> None:
    """Export best trial config to YAML and print CLI override flags."""
    from pathlib import Path

    import yaml

    out_dir = Path("data/sweep_results")
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
