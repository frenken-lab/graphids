"""Ray Tune HPO configuration for KD-GAT stages.

Replaces scripts/generate_sweep.py parallel-command approach with
Ray Tune + OptunaSearch + ASHAScheduler for efficient hyperparameter search.

Two trainable modes:
  - subprocess (default): Each trial spawns `cli.py` — CUDA-isolated but ASHA-inert
    (single val_loss report at end).
  - inprocess: Each trial trains in the same process with per-epoch reporting —
    enables ASHA multi-fidelity pruning (~2.5x speedup). Data loaded once and cached.

Usage:
    from graphids.pipeline.orchestration.tune_config import run_tune
    run_tune("autoencoder", dataset="hcrl_sa", num_samples=20)
    run_tune("curriculum", dataset="set_01", num_samples=20, inprocess=True)
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
# In-process trainable (per-epoch reporting for ASHA multi-fidelity)
# ---------------------------------------------------------------------------

# Module-level cache: data is read-only and shared across trials
_DATA_CACHE: dict[tuple, Any] = {}
_CURRICULUM_CACHE: dict[tuple, Any] = {}


def _dot_to_nested(config: dict) -> dict:
    """Convert dot-path keys (e.g. 'gat.hidden') to nested dicts for resolve()."""
    result: dict = {}
    for key, value in config.items():
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return result


def _load_data_once(stage: str, dataset: str, scale: str):
    """Load and cache data across trials. Data is read-only — safe to share."""
    key = (dataset, scale)
    if key not in _DATA_CACHE:
        from graphids.config.resolver import resolve

        model = _STAGE_MODEL[stage]
        cfg = resolve(model, scale, dataset=dataset)
        from ..stages.data_loading import load_data

        train_data, val_data, num_ids, in_ch = load_data(cfg)
        _DATA_CACHE[key] = (train_data, val_data, num_ids, in_ch)
        log.info(
            "Data cached for %s/%s: %d train, %d val graphs",
            dataset,
            scale,
            len(train_data),
            len(val_data),
        )
    return _DATA_CACHE[key]


def _load_curriculum_extras(dataset: str, scale: str, train_data, cfg):
    """Load and cache curriculum-specific data (difficulty scores, splits)."""
    key = (dataset, scale)
    if key not in _CURRICULUM_CACHE:
        import torch

        from ..stages.data_loading import graph_label
        from ..stages.trainer_factory import load_model
        from ..stages.utils import cleanup

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        num_ids = train_data[0].x[:, 0].long().max().item() + 1 if train_data else 100
        in_ch = train_data[0].x.shape[1] if train_data else 11

        # Load VGAE for difficulty scoring
        vgae = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)
        normals = [g for g in train_data if graph_label(g) == 0]
        attacks = [g for g in train_data if graph_label(g) == 1]

        from ..stages.training import _score_difficulty

        scores = _score_difficulty(vgae, normals, device)
        del vgae
        cleanup()

        _CURRICULUM_CACHE[key] = (normals, attacks, scores)
        log.info(
            "Curriculum cache for %s/%s: %d normals, %d attacks, %d scores",
            dataset,
            scale,
            len(normals),
            len(attacks),
            len(scores),
        )
    return _CURRICULUM_CACHE[key]


def _trainable_inprocess(
    config: dict, stage: str, dataset: str, scale: str, max_epochs: int = 0, patience: int = 0
) -> None:
    """In-process trainable with per-epoch val_loss reporting for ASHA.

    Key differences from _trainable (subprocess):
    - Data loaded once and cached across trials
    - Model built fresh each trial (no state bleeding)
    - val_loss reported after every validation epoch (ASHA can prune early)
    - No subprocess overhead (~40-60s saved per trial)
    """
    import gc

    import pytorch_lightning as pl
    import torch
    from ray import tune as ray_tune

    train_data, val_data, num_ids, in_ch = _load_data_once(stage, dataset, scale)

    # Build config with HP overrides from Ray
    from graphids.config.resolver import resolve

    model_type = _STAGE_MODEL[stage]
    nested = _dot_to_nested(config)
    nested["dataset"] = dataset
    if max_epochs > 0:
        nested.setdefault("training", {})["max_epochs"] = max_epochs
    if patience > 0:
        nested.setdefault("training", {})["patience"] = patience
    cfg = resolve(model_type, scale, **nested)

    pl.seed_everything(cfg.seed)

    # TuneReportCallback: reports val_loss each epoch so ASHA can prune
    class _TuneReportCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                ray_tune.report({"val_loss": float(val_loss), "epoch": trainer.current_epoch})

    try:
        if stage == "autoencoder":
            from ..stages.modules import VGAEModule
            from ..stages.utils import (
                compute_optimal_batch_size,
                make_dataloader,
                make_trainer,
            )

            module = VGAEModule(cfg, num_ids, in_ch)
            if cfg.training.optimize_batch_size:
                from ..stages.batch_sizing import compute_optimal_batch_size

                bs = compute_optimal_batch_size(module.model, train_data, cfg)
            else:
                from ..stages.batch_sizing import effective_batch_size

                bs = effective_batch_size(cfg)
            max_nodes = None
            if cfg.training.dynamic_batching:
                from ..stages.data_loading import compute_node_budget

                max_nodes = compute_node_budget(bs, cfg)
            train_dl = make_dataloader(train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes)
            val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)
            trainer = make_trainer(cfg, "autoencoder", extra_callbacks=[_TuneReportCallback()])
            trainer.fit(module, train_dl, val_dl)

        elif stage in ("curriculum", "normal"):
            from ..stages.modules import CurriculumDataModule, GATModule
            from ..stages.utils import make_trainer

            module = GATModule(cfg, num_ids, in_ch)

            if stage == "curriculum":
                normals, attacks, scores = _load_curriculum_extras(dataset, scale, train_data, cfg)
                dm = CurriculumDataModule(normals, attacks, scores, val_data, cfg)
                trainer = make_trainer(cfg, "curriculum", extra_callbacks=[_TuneReportCallback()])
                trainer.fit(module, datamodule=dm)
            else:
                from ..stages.batch_sizing import (
                    compute_optimal_batch_size,
                    effective_batch_size,
                )
                from ..stages.data_loading import compute_node_budget, make_dataloader

                if cfg.training.optimize_batch_size:
                    bs = compute_optimal_batch_size(module.model, train_data, cfg)
                else:
                    bs = effective_batch_size(cfg)
                max_nodes = None
                if cfg.training.dynamic_batching:
                    max_nodes = compute_node_budget(bs, cfg)
                train_dl = make_dataloader(
                    train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes
                )
                val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)
                trainer = make_trainer(cfg, "normal", extra_callbacks=[_TuneReportCallback()])
                trainer.fit(module, train_dl, val_dl)

        elif stage == "fusion":
            # Fusion uses episodes, not epochs — fallback to subprocess trainable
            # (no Lightning trainer, no per-epoch reporting)
            log.warning("Fusion stage does not support inprocess mode; falling back to subprocess")
            _trainable(config, stage, dataset, scale, max_epochs, patience)
            return

        else:
            raise ValueError(f"Unknown stage for inprocess trainable: {stage}")

    except Exception as e:
        log.warning("In-process trial failed: %s", e)
        print(f"[TRIAL FAILED] {e}", flush=True)
        ray_tune.report({"val_loss": float("inf")})
        return
    finally:
        # Cleanup between trials — prevent GPU memory leaks
        try:
            del module  # noqa: F821
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        log.info("Trial cleanup: peak GPU %.0f MB", peak_mb)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


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
            metric="val_loss",
            mode="min",
            grace_period=cfg.tune.grace_period,
            reduction_factor=cfg.tune.reduction_factor,
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
                storage_path=str(Path(__file__).resolve().parents[3] / "ray_results"),
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
    grace_period: int = 0,
    local: bool = False,
    max_epochs: int = 0,
    patience: int = 0,
    inprocess: bool = False,
    warm_start_from: str | None = None,
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
    inprocess : bool
        Use in-process trainable with per-epoch ASHA reporting (default: False).
        Enables multi-fidelity pruning (~2.5x speedup). Fusion always uses subprocess.
    warm_start_from : str | None
        Dataset name to warm-start from (e.g. "set_01"). Loads prior sweep results
        as points_to_evaluate + evaluated_rewards for OptunaSearch.

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

    # Load config for tune defaults (grace_period, reduction_factor)
    from graphids.config.resolver import resolve as _resolve

    cfg = _resolve(_STAGE_MODEL[stage], scale, dataset=dataset)
    if grace_period <= 0:
        grace_period = cfg.tune.grace_period

    # Fusion can't use inprocess (no Lightning trainer / no epochs)
    use_inprocess = inprocess and stage != "fusion"
    if inprocess and stage == "fusion":
        log.info("Fusion stage: falling back to subprocess trainable (no epoch-based training)")

    trainable_fn = _trainable_inprocess if use_inprocess else _trainable
    trainable_label = "inprocess" if use_inprocess else "subprocess"
    log.info(
        "Trainable mode: %s (ASHA %s)", trainable_label, "active" if use_inprocess else "inert"
    )

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
                trainable_fn,
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
