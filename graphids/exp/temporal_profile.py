"""Bounded CPU/CUDA timeline capture; profiling never changes training geometry."""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import lightning.pytorch as pl
import torch


def union_us(intervals):
    """Elapsed time covered by intervals, without double-counting concurrency."""
    total = 0.0
    end = float('-inf')
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    complete = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) > 0]
    batches = [event for event in complete if event.get("name") == "temporal/train_batch"
               and event.get("cat") == "user_annotation"]
    if not batches:
        raise ValueError("trace contains no training batches")
    start = min(event["ts"] for event in batches)
    end = max(event["ts"] + event["dur"] for event in batches)
    window = end - start
    selected = [event for event in complete if start <= event["ts"] < end]
    groups = defaultdict(list)
    totals = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for event in selected:
        category = event.get("cat", "")
        duration = min(event["ts"] + event["dur"], end) - event["ts"]
        groups[category].append((event["ts"], event["ts"] + duration))
        if category in {"kernel", "cuda_runtime", "gpu_memcpy", "gpu_memset"}:
            row = totals[(category, event["name"])]
            row["calls"] += 1
            row["total_us"] += duration
    device_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    busy = union_us([interval for category in device_categories for interval in groups[category]])
    return {
        "captured_batches": len(batches), "window_ms": window / 1000,
        "batch_union_ms": union_us([(e["ts"], e["ts"] + e["dur"]) for e in batches]) / 1000,
        "device_busy_ms": busy / 1000, "device_busy_fraction": busy / window,
        "category_union_ms": {key: union_us(value) / 1000 for key, value in groups.items()},
        "top_device_and_runtime": [dict(category=category, name=name, **row)
                                   for (category, name), row in sorted(totals.items(), key=lambda item: item[1]["total_us"], reverse=True)[:40]],
        "note": "Instrumented eight-batch window; overlapping device events count once. CPU operator totals can overlap and are not an additive wall-time decomposition.",
    }


class TemporalTimeline(pl.Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.profiler = None
        self.batch_region = None

    def on_train_start(self, trainer, pl_module):
        self.output.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities, schedule=torch.profiler.schedule(wait=16, warmup=4, active=8, repeat=1),
            on_trace_ready=self._save, record_shapes=True, profile_memory=False, with_stack=False,
        )
        self.profiler.start()

    def _save(self, profiler):
        trace = self.output / "trace.json"
        profiler.export_chrome_trace(str(trace))
        (self.output / "operators.txt").write_text(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
        (self.output / "summary.json").write_text(json.dumps(summarize_trace(trace), indent=2) + "\n")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_region = torch.profiler.record_function("temporal/train_batch")
        self.batch_region.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_region.__exit__(None, None, None)
        self.batch_region = None
        self.profiler.step()

    def _stop(self):
        if self.batch_region is not None:
            self.batch_region.__exit__(None, None, None)
            self.batch_region = None
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

    def on_train_end(self, trainer, pl_module):
        self._stop()

    def on_exception(self, trainer, pl_module, exception):
        try:
            self._stop()
        except Exception as exc:
            warnings.warn(f"Profiler cleanup failed while preserving original exception: {exc}", stacklevel=2)
