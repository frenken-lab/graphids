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
from graphids.core.data.state import DatasetState
from graphids.exp.config import ExperimentConfig
from graphids.exp.runtime import (
    _scalar_metrics,
    build_model,
    resolve_spec,
    seed_run,
    trainer_components,
)


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


def run_comparison(config_path, output, *, section_limit=None, profile=False):
    experiment = ExperimentConfig.from_yaml(config_path)
    output = Path(output).resolve()
    run = experiment.build_run(name=experiment.experiment_name, stage="fit", run_dir=output)
    run.write_config()
    cfg = run.payload.model_dump(mode="json")
    seed_run(cfg)
    data_cfg = resolve_spec(cfg["data"])
    source_spec = data_cfg.source
    if source_spec.val_split != "chronological":
        raise ValueError("this historical throughput comparison requires the original chronological cache; use the experiment runner for recording holdouts")
    cache = data_cfg.build_source().cache_root_path() / "processed"
    dm = data_cfg.build(dataset=CachedComparisonSource(cache))
    dm.setup()
    if section_limit is not None:
        if data_cfg.coverage_policy != "diagnostic" or data_cfg.stream_execution != "segmented":
            raise ValueError("section limits require diagnostic segmented execution")
        if section_limit < 1:
            raise ValueError("section_limit must be positive")
        dm.train_dataloader()
        dataset = dm._train_lane_dataset
        dataset.sections = dataset.sections[:section_limit]
        dataset._plans = tuple(dataset._filter_plans(dataset._iter_plans()))
    # Freeze the cache acceptance criteria before any benchmark.
    streams = torch.unique(dm.train_data.stream_id)
    counts = torch.bincount(dm.train_data.y.long(), minlength=2)
    if dm.train_data.num_events != 12_090_403 or streams.numel() != 18 or int(counts[1]) != 57_108:
        raise ValueError(f"unexpected set_01 cache: {dm.train_data.num_events} events, {streams.numel()} streams, {counts.tolist()} classes")
    model = build_model(cfg, dm)
    initial_digest = weight_digest(model)
    kwargs = trainer_components(cfg)
    full = dm.coverage_policy == "full"
    if full:
        validate_full_lane_settings(kwargs)
    audit = TemporalCoverageAudit(output, require_complete=full)
    callbacks = kwargs["callbacks"]
    callbacks.append(audit)
    if profile:
        from graphids.exp.temporal_profile import TemporalTimeline
        callbacks.append(TemporalTimeline(output / "profile"))
    kwargs.update(default_root_dir=str(output), logger=False, callbacks=callbacks)
    trainer = pl.Trainer(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    fit_seconds = time.perf_counter() - started
    if full and (audit.last_report is None or not audit.last_report["complete"]):
        raise RuntimeError("production comparison did not finish a complete training epoch")
    cuda_peak = {
        "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    } if torch.cuda.is_available() else None
    dataset = dm._train_lane_dataset
    section_plan = [vars(section) for section in dataset.sections]
    checkpoint = output / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint)
    train_metrics = _scalar_metrics(trainer.callback_metrics)
    graph_report = model.cuda_graph_report()
    optimizer_steps = int(getattr(model, "_optimizer_steps", 0))
    # Restore and evaluate the saved checkpoint using the same continuous prefixes.
    validation_started = time.perf_counter()
    trainer.validate(model, datamodule=dm, ckpt_path=str(checkpoint), verbose=False)
    metrics = {**train_metrics, **_scalar_metrics(trainer.callback_metrics)}
    report = {
        "config": str(config_path), "checkpoint": str(checkpoint), "initial_weight_sha256": initial_digest,
        "section_limit": section_limit,
        "section_queue_sha256": hashlib.sha256(json.dumps(section_plan, sort_keys=True).encode()).hexdigest(),
        "planned_scored_events": sum(section.core_end - section.core_start for section in dataset.sections),
        "planned_warmup_events": sum(section.core_start - section.warmup_start for section in dataset.sections),
        "planned_successor_targets": sum(
            section.core_end - section.core_start - int(section.core_end == dataset._streams[section.stream_idx].length)
            for section in dataset.sections
        ),
        "cuda_peak_memory": cuda_peak,
        "cuda_graph": graph_report, "optimizer_steps": optimizer_steps,
        "optimizer_updates_per_second": optimizer_steps / fit_seconds,
        "end_to_end_scored_events_per_second": audit.last_report["unique"] / fit_seconds,
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
    parser.add_argument("--section-limit", type=int)
    parser.add_argument("--profile", action="store_true", help="capture eight CPU/CUDA batches after warmup")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"comparison interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    run_comparison(args.config, args.output, section_limit=args.section_limit, profile=args.profile)


if __name__ == "__main__":
    main()
