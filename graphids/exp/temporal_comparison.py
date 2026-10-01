"""One-GPU, cache-only comparison of full recordings and experimental sections.

Run with ``python -m graphids.exp.temporal_comparison CONFIG --output DIR``.
Only training and bounded validation are loaded; no test cache or raw rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.core.data.datamodule.coverage import (
    TemporalCoverageAudit,
    validate_full_lane_settings,
)
from graphids.core.data.datamodule.temporal import TemporalDataModule
from graphids.core.data.state import DatasetState
from graphids.exp.config import ExperimentConfig
from graphids.exp.ray_backend import _scalar_metrics, build_component, resolve_spec


class CachedComparisonSource:
    def __init__(self, root):
        self.root = root
        self.cache_key = f"temporal-comparison:{root}"

    def build(self):
        def load(name):
            return torch.load(self.root / f"data_{name}.pt", mmap=True, map_location="cpu", weights_only=False)
        return DatasetState(train=load("train"), val=load("val"), test={})


def weight_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_comparison(config_path, output):
    experiment = ExperimentConfig.from_yaml(config_path)
    run = experiment.build_run(name=experiment.experiment_name, stage="fit", config=experiment.config)
    cfg = run.payload.model_dump(mode="json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed_everything")
    pl.seed_everything(int(seed if seed is not None else 42), workers=True)
    data_cfg = resolve_spec(cfg["data"])
    source_spec = data_cfg.source
    from graphids.core.data.datasets.can_bus import CANBusTemporalSource
    source = CANBusTemporalSource(name=source_spec.name, val_fraction=source_spec.val_fraction,
                                 representation_cfg=source_spec.representation_cfg,
                                 train_source_mode=source_spec.train_source_mode,
                                 val_warmup_events=data_cfg.val_warmup_events,
                                 test_warmup_events=data_cfg.test_warmup_events)
    cache = source.cache_root_path() / "processed"
    options = data_cfg.model_dump(exclude={"type", "source", "val_warmup_events", "test_warmup_events"})
    dm = TemporalDataModule(CachedComparisonSource(cache), **options)
    dm.setup()
    # Freeze the cache acceptance criteria before any benchmark.
    streams = torch.unique(dm.train_data.stream_id)
    counts = torch.bincount(dm.train_data.y.long(), minlength=2)
    if dm.train_data.num_events != 12_090_403 or streams.numel() != 18 or int(counts[1]) != 57_108:
        raise ValueError(f"unexpected set_01 cache: {dm.train_data.num_events} events, {streams.numel()} streams, {counts.tolist()} classes")
    loss_fn = build_component(cfg["loss_fn"]) if cfg.get("loss_fn") else None
    model_spec = resolve_spec(cfg["model"])
    model = model_spec.build(loss_fn=loss_fn)
    model.prepare_from_datamodule(dm)
    initial_digest = weight_digest(model)
    kwargs = dict(cfg["trainer"])
    full = dm.coverage_policy == "full"
    if full:
        validate_full_lane_settings(kwargs)
    audit = TemporalCoverageAudit(output, require_complete=full)
    callbacks = []
    for spec in cfg.get("callbacks", {}).values():
        # Output stays in this allocation's unique directory.
        spec = json.loads(json.dumps(spec).replace(str(run.outputs.run_dir), str(output)))
        callbacks.append(build_component(spec))
    callbacks.append(audit)
    kwargs.update(default_root_dir=str(output), logger=False, callbacks=callbacks)
    trainer = pl.Trainer(**kwargs)
    started = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    fit_seconds = time.perf_counter() - started
    if full and (audit.last_report is None or not audit.last_report["complete"]):
        raise RuntimeError("production comparison did not finish a complete training epoch")
    checkpoint = output / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint)
    train_metrics = _scalar_metrics(trainer.callback_metrics)
    # Restore and evaluate the saved checkpoint using the same continuous prefixes.
    validation_started = time.perf_counter()
    trainer.validate(model, datamodule=dm, ckpt_path=str(checkpoint), verbose=False)
    metrics = {**train_metrics, **_scalar_metrics(trainer.callback_metrics)}
    report = {
        "config": str(config_path), "checkpoint": str(checkpoint), "initial_weight_sha256": initial_digest,
        "fit_wall_seconds": fit_seconds, "validation_wall_seconds": time.perf_counter() - validation_started,
        "coverage": audit.last_report, "metrics": metrics,
        "validation_auroc_note": "available" if metrics.get("val_auroc_available") else "unavailable: validation scoring prefixes contain one class",
        "training_class_counts": counts.tolist(), "training_streams": streams.tolist(),
        "cache": str(cache), "cache_train_sha256": (cache / "data_train.pt.sha256").read_text().strip(),
        "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("comparison_result " + json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"comparison interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    run_comparison(args.config, args.output)


if __name__ == "__main__":
    main()
