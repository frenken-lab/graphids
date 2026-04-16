"""Single-stage primitives.

``build`` / ``train`` / ``evaluate`` are the atomic verbs between a
``ResolvedConfig`` and a running ``Trainer``. Each takes the resolved
config directly so callers don't have to unpack ``rendered`` /
``validated`` / ``run_dir`` / ``ckpt_file`` into positional arguments at
every call site.

``wire_file_exporters`` is called once by the caller per stage, not by
these primitives.

When ``resolved.run_dir`` is ``None`` (CLI smoke with no
``default_root_dir``), the primitives skip all filesystem side effects:
no markers, no file exporters, no ckpt_path hand-off to ``.test()``.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from graphids._fs import touch_marker
from graphids._otel import get_logger
from graphids.catalog import Catalog
from graphids.config.constants import LAKE_ROOT, PHASE_MARKERS, PROJECT_ROOT
from graphids.orchestrate.config import InstantiatedRun, ResolvedConfig
from graphids.orchestrate.instantiate import build_run

log = get_logger(__name__)


def _stack_predict_results(results: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate per-batch ``predict_step`` dicts into a single tensor dict."""
    if not results:
        return {}
    keys: set[str] = set()
    for r in results:
        keys.update(r.keys())
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
        if not tensors:
            continue
        torch.save(tensors, out_dir / f"{name}.pt")
    log.info("save_test_predictions", sets=list(preds.keys()), dir=str(out_dir))


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _write_summary(run_dir: Path, metrics: dict[str, Any], ckpt_file: Path | None) -> None:
    """Write ``summary.json`` — final metrics + git SHA + best-ckpt relpath."""
    payload = {
        "metrics": metrics,
        "git_sha": _git_sha(),
        "status": "ok",
        "best_ckpt": str(ckpt_file.relative_to(run_dir)) if ckpt_file else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))


def build(resolved: ResolvedConfig) -> InstantiatedRun:
    """Instantiate trainer + model + datamodule from a resolved config.

    GPU state is reset first so a prior stage's VRAM / compiled kernels
    don't leak into this one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.compiler.reset()
    return build_run(resolved.rendered, validated=resolved.validated)


def train(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
    *,
    resume_from: str | None = None,
) -> Path | None:
    """Fit the model and return the canonical checkpoint path.

    Touches the train phase marker on success. Caller is expected to
    have wired OTel file exporters for this stage's run_dir already.
    """
    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    log.info("stage_train", stage=stage_name, run_dir=str(run_dir) if run_dir else "")
    artifacts.trainer.fit(
        artifacts.model,
        datamodule=artifacts.datamodule,
        ckpt_path=resume_from,
    )
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


def evaluate(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
) -> dict[str, Any]:
    """Run the test phase and return metrics.

    On success, writes the test-phase marker, the per-test-set prediction
    sidecars, ``summary.json``, and the catalog row (runs + metrics in
    one transaction on the lake-wide DuckDB file). Timestamps are stamped
    here — ``traces.jsonl`` remains authoritative for spans, but the
    catalog row stands alone.
    """
    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    log.info("stage_test", stage=stage_name)
    started_at_ns = time.time_ns()
    metrics = artifacts.trainer.test(
        artifacts.model,
        datamodule=artifacts.datamodule,
        ckpt_path=str(ckpt_file) if ckpt_file is not None else None,
    )
    ended_at_ns = time.time_ns()
    if run_dir is not None:
        touch_marker(run_dir / PHASE_MARKERS["test"])
        _save_test_predictions(artifacts.model, run_dir / "predictions" / "test")
        _write_summary(run_dir, metrics or {}, ckpt_file)
        run_id = Catalog(LAKE_ROOT).record_run(
            run_dir,
            metrics=metrics or {},
            git_sha=_git_sha(),
            status="ok",
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
        )
        if run_id is None:
            log.info("catalog_skip_non_ablation", run_dir=str(run_dir))
        else:
            log.info("catalog_recorded_run", run_id=run_id)
    log.info("stage_complete", stage=stage_name)
    return metrics or {}
