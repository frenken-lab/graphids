"""Production lane coverage checks and compact CPU bitmaps."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import SequentialSampler


def validate_full_lane_settings(kwargs):
    limit = kwargs.get("limit_train_batches", 1.0)
    if type(limit) is not float or limit != 1.0:
        raise ValueError("full lane coverage requires unrestricted limit_train_batches: 1.0")
    if kwargs.get("max_steps", -1) != -1:
        raise ValueError("full lane coverage rejects finite max_steps")
    if kwargs.get("fast_dev_run", False) or kwargs.get("overfit_batches", 0):
        raise ValueError("full lane coverage rejects fast_dev_run and overfit_batches")
    devices = kwargs.get("devices", 1)
    if devices not in (1, "1", "auto") and devices != [0]:
        raise ValueError("full lane coverage initially supports one process/GPU")
    if kwargs.get("num_nodes", 1) != 1:
        raise ValueError("full lane coverage rejects distributed index sharding")


class CoverageBitmap:
    """One bit per source event; count duplicates even within a single batch."""

    def __init__(self, size):
        self.size = size
        self.bits = np.zeros((size + 7) // 8, dtype=np.uint8)
        self.observed = self.duplicate = self.unexpected = 0

    def add(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.observed += indices.size
        good = indices[(indices >= 0) & (indices < self.size)]
        self.unexpected += indices.size - good.size
        unique = np.unique(good)
        already = (self.bits[unique // 8] >> (unique % 8)) & 1
        self.duplicate += good.size - unique.size + int(already.sum())
        np.bitwise_or.at(self.bits, unique // 8, (1 << (unique % 8)).astype(np.uint8))

    def mask(self):
        return np.unpackbits(self.bits, bitorder="little")[:self.size].astype(bool)


class TemporalCoverageAudit(pl.Callback):
    """Verify core ownership before fit and actual scored source indices per epoch."""

    def __init__(self, report_dir=None, require_complete=True):
        self.report_dir = report_dir
        self.require_complete = require_complete
        self.events = None
        self.last_report = None

    @staticmethod
    def verify_plan(dataset):
        intervals = [[] for _ in dataset._streams]
        for section in dataset.sections:
            intervals[section.stream_idx].append((section.core_start, section.core_end))
        for stream, cores in zip(dataset._streams, intervals, strict=True):
            cursor = 0
            for start, end in sorted(cores):
                if start != cursor or end <= start:
                    raise ValueError("coverage plan has missing or duplicate core events")
                cursor = end
            if cursor != stream.length:
                raise ValueError("coverage plan is incomplete")
        cursors = [section.warmup_start for section in dataset.sections]
        for plan in dataset._plans:
            for item in plan.items:
                if item.offset != cursors[item.section_idx] or item.end <= item.offset:
                    raise ValueError("coverage batch plan skips or repeats section history")
                cursors[item.section_idx] = item.end
        if any(cursor != section.core_end for cursor, section in zip(cursors, dataset.sections, strict=True)):
            raise ValueError("coverage batch plan is incomplete")
        scored = getattr(dataset.data, "is_scored", None)
        if scored is not None and not bool(scored.all()):
            raise ValueError("full training coverage requires every source event to be scored")
        if dataset.drop_partial_batches:
            raise ValueError("full coverage cannot drop partial lane batches")

    def on_train_start(self, trainer, pl_module):
        dm = trainer.datamodule
        self.dataset = dm._train_lane_dataset
        if self.require_complete:
            validate_full_lane_settings({
                "limit_train_batches": trainer.limit_train_batches,
                "max_steps": trainer.max_steps,
                "fast_dev_run": trainer.fast_dev_run,
                "overfit_batches": trainer.overfit_batches,
            })
            if trainer.world_size != 1:
                raise ValueError("full lane coverage rejects distributed index sharding")
            loader = trainer.train_dataloader
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last or not getattr(loader, "in_order", True):
                raise ValueError("full lane coverage requires ordered, complete sequential sampling")
            self.verify_plan(self.dataset)
        self.report_path = Path(self.report_dir or trainer.default_root_dir) / "coverage"

    def on_train_epoch_start(self, trainer, pl_module):
        n = int(self.dataset.data.dst.numel())
        self.events = CoverageBitmap(n)
        self.targets = CoverageBitmap(n)
        self.warmup = self.bad_warmup = self.padding = self.capacity = self.active = self.batches = 0
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        valid = batch.valid_mask.detach().cpu().bool()
        scored = batch.is_scored.detach().cpu().bool()
        source = batch.source_index.detach().cpu()
        warmup = batch.warmup_mask.detach().cpu().bool()
        self.events.add(source[scored].numpy())
        self.bad_warmup += int((scored & (warmup | ~valid)).sum())
        self.warmup += int((valid & warmup).sum())
        self.targets.add(source[scored & (batch.successor_index.detach().cpu() >= 0)].numpy())
        self.padding += int((~valid).sum())
        self.capacity += valid.numel()
        self.active += int(valid.any(dim=1).sum())
        self.batches += 1
        if self.batches % 1000 == 0:
            elapsed = time.perf_counter() - self.started
            progress = {
                "epoch": trainer.current_epoch, "batches": self.batches,
                "scored_events": self.events.observed, "warmup_events": self.warmup,
                "wall_seconds": elapsed,
                "scored_events_per_second": self.events.observed / max(elapsed, 1e-9),
            }
            if callable(getattr(pl_module, "cuda_graph_report", None)):
                graph = pl_module.cuda_graph_report()
                progress["cuda_graph"] = {
                    key: graph.get(key) for key in
                    ("replays", "mixed_replays", "scored_replay_fraction", "fallback_batches", "disabled", "mixed_disabled")
                }
            print("temporal_coverage_progress " + json.dumps(progress), flush=True)

    def report(self, trainer, *, interrupted=False):
        if self.events is None:
            return None
        seen = self.events.mask()
        targets = self.targets.mask()
        data = self.dataset.data
        n = seen.size
        per_stream = {}
        eligible = np.ones(n, dtype=bool)
        for stream in self.dataset._streams:
            indices = stream.source_indices(0, stream.length).numpy()
            eligible[indices[-1]] = False
            per_stream[str(stream.stream_id)] = {"expected": stream.length, "observed": int(seen[indices].sum())}
        classes = data.y.detach().cpu().numpy()
        per_class = {str(c): {"expected": int((classes == c).sum()), "observed": int(seen[classes == c].sum())} for c in np.unique(classes)}
        elapsed = time.perf_counter() - self.started
        report = {
            "epoch": trainer.current_epoch, "interrupted": interrupted,
            "expected": n, "observed": self.events.observed, "unique": int(seen.sum()),
            "missing": int((~seen).sum()), "duplicate": self.events.duplicate,
            "unexpected": self.events.unexpected, "incorrectly_scored_warmup_or_padding": self.bad_warmup,
            "per_stream": per_stream, "per_class": per_class,
            "warmup_events": self.warmup, "warmup_overhead": self.warmup / max(1, int(seen.sum())),
            "eligible_successor_targets": int(eligible.sum()), "observed_successor_targets": int(targets.sum()),
            "missing_successor_targets": int((eligible & ~targets).sum()),
            "unexpected_successor_targets": int((~eligible & targets).sum()) + self.targets.unexpected,
            "duplicate_successor_targets": self.targets.duplicate,
            "mean_active_lanes": self.active / max(1, self.batches),
            "lane_occupancy": (self.capacity - self.padding) / max(1, self.capacity),
            "padding_events": self.padding, "padding_fraction": self.padding / max(1, self.capacity),
            "wall_seconds": elapsed, "scored_events_per_second": int(seen.sum()) / max(elapsed, 1e-9),
            "diagnostic_gap_events": self.dataset.diagnostic_gaps,
        }
        report["complete"] = not any(report[k] for k in (
            "missing", "duplicate", "unexpected", "incorrectly_scored_warmup_or_padding",
            "missing_successor_targets", "unexpected_successor_targets", "duplicate_successor_targets",
        )) and not interrupted
        self.report_path.mkdir(parents=True, exist_ok=True)
        path = self.report_path / f"epoch-{trainer.current_epoch}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
        self.last_report = report
        print("temporal_coverage " + json.dumps(report), flush=True)
        return report

    def on_train_epoch_end(self, trainer, pl_module):
        report = self.report(trainer)
        if self.require_complete and not report["complete"]:
            raise RuntimeError("incomplete production training epoch; see coverage report")

    def on_exception(self, trainer, pl_module, exception):
        try:
            self.report(trainer, interrupted=True)
        except Exception as exc:
            warnings.warn(f"Coverage report failed while preserving original exception: {exc}", stacklevel=2)
