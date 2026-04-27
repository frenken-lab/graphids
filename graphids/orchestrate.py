"""Render-validated config → instantiated trainer/model/datamodule → fit/test.

Three layers in one file:

* :class:`ResolvedConfig` / :class:`InstantiatedRun` — boundary types.
* :func:`build_run` — class_path resolver for trainer + model + datamodule
  (with optional ``reset_gpu=True`` GPU-state reset preamble for the
  production fit/test path).
* :func:`train` / :func:`evaluate` — atomic stage primitives.

No planner, no cross-stage driver: each stage is a separate
:func:`graphids.slurm.submit.submit` invocation. Multi-stage workflows
live in plan jsonnets rendered to JSONL by :mod:`graphids.slurm.run`.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from graphids._fs import touch_marker
from graphids._otel import get_logger
from graphids._reflect import filter_kwargs, import_class
from graphids.config.constants import PHASE_MARKERS
from graphids.config.schemas import ValidatedConfig, validate_config
from graphids.core.losses.build import inject_loss_fn
from graphids.core.trainer import Trainer, TrainerConfig, seed_everything

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_TRAINER_CONFIG_KEYS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(TrainerConfig))
_DEFAULT_ID_ENCODER = "graphids.core.models.id_encoding.LookupIdEncoder"


# ---------------------------------------------------------------------------
# Boundary types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Rendered, validated config ready for instantiation.

    ``run_dir`` is ``None`` only for smoke invocations of the Typer CLI
    with no ``default_root_dir`` set — markers and file exporters are
    skipped in that case.
    """

    rendered: dict[str, Any]
    validated: ValidatedConfig
    stage_name: str
    run_dir: Path | None
    ckpt_file: Path | None

    @classmethod
    def from_rendered(cls, rendered: dict[str, Any], *, stage_name: str) -> ResolvedConfig:
        """Validate a pre-rendered dict and pull ``run_dir`` from jsonnet."""
        from graphids.config.constants import CKPT_SUBPATH

        validated = validate_config(rendered)
        default_root = (rendered.get("trainer") or {}).get("default_root_dir") or ""
        run_dir = Path(default_root) if default_root else None
        ckpt_file = run_dir / CKPT_SUBPATH if run_dir else None
        return cls(
            rendered=rendered,
            validated=validated,
            stage_name=stage_name,
            run_dir=run_dir,
            ckpt_file=ckpt_file,
        )


@dataclass
class InstantiatedRun:
    """A wired (trainer, model, datamodule) triple built from a rendered config."""

    trainer: Trainer
    model: nn.Module
    datamodule: Any


# ---------------------------------------------------------------------------
# Instantiation — class_path resolver
# ---------------------------------------------------------------------------


def _resolve_nested(value: Any) -> Any:
    """Recursively instantiate any ``{class_path, init_args}`` dicts inside ``value``."""
    if isinstance(value, dict):
        if "class_path" in value:
            return _build_block(value)
        return {k: _resolve_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_nested(v) for v in value]
    return value


def _build_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict (recursing into nested blocks)."""
    klass = import_class(block["class_path"])
    init_args = block.get("init_args") or {}
    resolved = {k: _resolve_nested(v) for k, v in init_args.items()}
    return klass(**resolved)


def _build_model(merged: dict[str, Any]) -> nn.Module:
    class_path = merged["model"]["class_path"]
    init_args = inject_loss_fn(
        merged["model"].get("init_args") or {},
        class_path=class_path,
    )
    klass = import_class(class_path)
    return klass(**filter_kwargs(klass, init_args))


def _build_callbacks(merged: dict[str, Any]) -> list:
    """Instantiate trainer.callbacks; append VRAMDriftCallback when CUDA is available."""
    from graphids.config.settings import get_settings
    from graphids.core.callbacks import VRAMDriftCallback

    entries = (merged.get("trainer") or {}).get("callbacks") or []
    callbacks = [_build_block(entry) for entry in entries]
    if torch.cuda.is_available():
        callbacks.append(VRAMDriftCallback(threshold=get_settings().vram_drift_threshold))
    return callbacks


def _build_loggers(merged: dict[str, Any]) -> list | bool | None:
    logger_cfg = (merged.get("trainer") or {}).get("logger")
    if isinstance(logger_cfg, (list, dict)):
        entries = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
        return [_build_block(e) for e in entries]
    return logger_cfg


def _build_trainer(merged: dict[str, Any]) -> Trainer:
    trainer_dict = merged.get("trainer") or {}
    cfg_kwargs = {k: v for k, v in trainer_dict.items() if k in _TRAINER_CONFIG_KEYS}
    return Trainer(
        config=TrainerConfig(**cfg_kwargs),
        callbacks=_build_callbacks(merged),
        logger=_build_loggers(merged),
    )


def build_run(
    rendered: dict[str, Any],
    *,
    validated: ValidatedConfig | None = None,
    seed_all: bool = True,
    reset_gpu: bool = False,
) -> InstantiatedRun:
    """Instantiate trainer + model + datamodule from a rendered config dict.

    ``reset_gpu=True`` runs ``gc.collect`` + ``torch.cuda.empty_cache`` +
    ``torch.cuda.reset_peak_memory_stats`` + ``torch.compiler.reset``
    before instantiation — the production fit/test path needs this to
    free state from any prior in-process build (compiled-model leftovers
    between fit and test, etc.). Tests pass ``reset_gpu=False`` so unit
    tests don't touch CUDA.
    """
    if reset_gpu:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.compiler.reset()
    merged = copy.deepcopy(rendered)
    if validated is None:
        validate_config(merged)
    if seed_all:
        seed_everything(merged["seed_everything"])
    return InstantiatedRun(
        model=_build_model(merged),
        datamodule=_build_block(merged["data"]),
        trainer=_build_trainer(merged),
    )


# ---------------------------------------------------------------------------
# Stage primitives — build / train / evaluate
# ---------------------------------------------------------------------------


def _stack_predict_results(results: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate per-batch ``predict_step`` dicts into a single tensor dict."""
    if not results:
        return {}
    keys = {k for r in results for k in r}
    stacked: dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [r[k].detach().cpu() for r in results if k in r and torch.is_tensor(r[k])]
        if tensors:
            stacked[k] = torch.cat(tensors)
    return stacked


def _save_split_predictions(artifacts: InstantiatedRun, split: str, out_dir: Path) -> None:
    """Run ``predict_step`` over train/val loader and save tensors to disk."""
    dm = artifacts.datamodule
    loader_fn = getattr(dm, f"{split}_dataloader", None)
    if loader_fn is None:
        return
    try:
        loader = loader_fn()
    except Exception as exc:
        log.warning("save_predictions_no_loader", split=split, error=str(exc))
        return
    if loader is None:
        return
    try:
        results = artifacts.trainer.predict_on(artifacts.model, loader)
        stacked = _stack_predict_results(results)
    except Exception as exc:
        log.warning("save_predictions_failed", split=split, error=str(exc))
        return
    if not stacked:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, out_dir / f"{split}.pt")
    log.info(
        "save_predictions",
        split=split,
        path=str(out_dir / f"{split}.pt"),
        n=int(next(iter(stacked.values())).shape[0]),
    )


def _save_test_predictions(model: Any, out_dir: Path) -> None:
    """Persist ``model._test_predictions`` (one tensor dict per test set)."""
    preds = getattr(model, "_test_predictions", None)
    if not preds:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensors in preds.items():
        if tensors:
            torch.save(tensors, out_dir / f"{name}.pt")
    log.info("save_test_predictions", sets=list(preds.keys()), dir=str(out_dir))


def _check_ckpt_compat(ckpt_path: str | Path, resolved: ResolvedConfig) -> None:
    """Raise on ckpt/config mismatch (Module class or id encoder).

    Strict load catches shape-level state_dict mismatches; this turns the
    two most common silent-mismatch cases into actionable messages first.
    """
    from graphids._fs import atomic_load

    ckpt = atomic_load(str(ckpt_path), map_location="cpu", weights_only=True)
    cfg_model = resolved.rendered.get("model") or {}
    cfg_init = cfg_model.get("init_args") or {}
    pairs = [
        ("model.class_path", ckpt.get("class_path"), cfg_model.get("class_path")),
        (
            "id_encoder_class_path",
            (ckpt.get("hyper_parameters") or {}).get("id_encoder_class_path", _DEFAULT_ID_ENCODER),
            cfg_init.get("id_encoder_class_path", _DEFAULT_ID_ENCODER),
        ),
    ]
    for label, ckpt_val, cfg_val in pairs:
        if ckpt_val and cfg_val and ckpt_val != cfg_val:
            raise ValueError(
                f"{label} mismatch: ckpt={ckpt_val!r} cfg={cfg_val!r} ckpt_path={ckpt_path}"
            )


def train(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
    *,
    resume_from: str | None = None,
) -> Path | None:
    """Fit the model and return the canonical checkpoint path.

    Opens the MLflow run before fit (callback closes it at on_fit_end /
    on_exception); end_training_run in finally is a teardown safety net.
    """
    from graphids._mlflow import end_training_run, start_training_run

    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    log.info("stage_train", stage=stage_name, run_dir=str(run_dir) if run_dir else "")
    if resume_from is not None:
        _check_ckpt_compat(resume_from, resolved)
    if run_dir is not None:
        start_training_run(run_dir, resolved.validated.model_dump())
    try:
        artifacts.trainer.fit(
            artifacts.model,
            datamodule=artifacts.datamodule,
            ckpt_path=resume_from,
        )
    finally:
        end_training_run()
    if run_dir is not None:
        touch_marker(run_dir / PHASE_MARKERS["train"])
        pred_dir = run_dir / "predictions"
        _save_split_predictions(artifacts, "train", pred_dir)
        _save_split_predictions(artifacts, "val", pred_dir)
    log.info(
        "stage_train_complete",
        stage=stage_name,
        ckpt=str(ckpt_file) if ckpt_file else "",
    )
    return ckpt_file


def evaluate(artifacts: InstantiatedRun, resolved: ResolvedConfig) -> dict[str, Any]:
    """Run the test phase; write marker, prediction sidecars, MLflow test row."""
    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    log.info("stage_test", stage=stage_name)
    if ckpt_file is not None:
        _check_ckpt_compat(ckpt_file, resolved)
    metrics = artifacts.trainer.test(
        artifacts.model,
        datamodule=artifacts.datamodule,
        ckpt_path=str(ckpt_file) if ckpt_file is not None else None,
    )
    if run_dir is not None:
        from graphids._mlflow import log_test_run

        touch_marker(run_dir / PHASE_MARKERS["test"])
        _save_test_predictions(artifacts.model, run_dir / "predictions" / "test")
        log_test_run(
            run_dir,
            resolved_config=resolved.validated.model_dump(),
            metrics=metrics or {},
        )
    log.info("stage_complete", stage=stage_name)
    return metrics or {}
