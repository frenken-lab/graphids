#!/usr/bin/env python3
"""Unified bounded correctness, public-launch and single-A100 performance gate.

Run inside Slurm. No target data and no validation accuracy tuning.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path


def persist(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def baseline_readout():
    """The archived graph readout, keeping identical continuous scan semantics."""
    import numpy as np
    import torch

    with tarfile.open("archives/snapshot-2026-10-16.tar.gz") as archive:
        source = (
            archive.extractfile("graphids/core/models/event_graph_mamba.py")
            .read()
            .decode()
        )
    start, stop = source.index("    def _readout"), source.index("    def forward")
    import textwrap

    namespace = dict(np=np, torch=torch, math=math)
    exec(textwrap.dedent(source[start:stop]), namespace)
    return namespace["_readout"]


def benchmark(config, output, steps):
    import numpy as np
    import torch

    from graphids.core.data.events.data import EventStreamWindows
    from graphids.core.models.event_graph_mamba import EventGraphMambaState
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.runtime import build_component

    if not torch.cuda.is_available():
        raise RuntimeError("official CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = ExperimentConfig.from_yaml(config)
    # Construct only training query caches. No validation files are opened.
    from graphids.core.data.events.data import EventRecording, materialize_events
    from graphids.core.data.recordings import CANRecordingManifest

    manifest = CANRecordingManifest.load(
        cfg.config["data"]["manifest_path"], verify_files=False
    )
    records = [
        r
        for r in manifest.records
        if r.role == "train_query" and r.dataset in cfg.config["data"]["datasets"]
    ]
    if len({r.vehicle for r in records}) != 2 or {
        r.verified_normal for r in records
    } != {False, True}:
        raise ValueError(
            "benchmark requires both reviewed vehicles, normal and attack recordings"
        )
    # Interleave vehicles/normality before the remaining independent recordings.
    # Round robin provenance classes so short diagnostics include both vehicles.
    ordered = []
    for normal in (True, False):
        for vehicle in sorted({r.vehicle for r in records}):
            ordered.append(
                next(
                    r
                    for r in records
                    if r.verified_normal == normal and r.vehicle == vehicle
                )
            )
    records = ordered + [r for r in records if r not in ordered]
    events = [
        EventRecording(materialize_events(r, Path(cfg.config["data"]["cache_root"])))
        for r in records
    ]
    offsets = {}
    for recording in events:
        positives = np.flatnonzero(recording.arrays["labels"] == 1)
        offsets[recording.recording.key] = (
            (int(positives[0]) // 8192) * 8192 if len(positives) else 0
        )

    class Segment:
        def __init__(self, source):
            self.source, self.recording = source, source.recording
            self.offset = offsets[source.recording.key]
            self.count = min(8192, source.count - self.offset)

        def batch(self, start, stop):
            from dataclasses import replace

            return replace(
                self.source.batch(self.offset + start, self.offset + stop),
                recording_end=stop == self.count,
            )

    source_epoch_events = sum(r.count for r in events)
    cohort = [
        dict(
            recording=r.recording.key,
            vehicle=r.recording.vehicle,
            verified_normal=r.recording.verified_normal,
            available_events=r.count,
            timed_start_row=offsets[r.recording.key],
            timed_events=min(8192, r.count - offsets[r.recording.key]),
            timed_positive_frames=int(
                (
                    r.arrays["labels"][
                        offsets[r.recording.key] : offsets[r.recording.key] + 8192
                    ]
                    == 1
                ).sum()
            ),
        )
        for r in events
    ]
    original_readout = baseline_readout()
    results = dict(
        device=torch.cuda.get_device_name(),
        seed=42,
        precision="FP32; TF32 disabled",
        manifest_digest=manifest.digest,
        source_epoch_events=source_epoch_events,
        cohort=cohort,
        configurations=[],
        baseline_definition="archived graph readout with continuous differentiable scan, 512 events x 1 stream",
        note="bounded training segments with continuous preceding history; no validation tuning or full learning run",
    )
    persist(output / "benchmark.json", results)

    def check_configuration(model, bounded, chunk, streams, prepared):
        """Outputs, caches and gradients versus independent serial execution."""
        model.eval()
        before = [r.batch(0, chunk).to("cuda") for r in bounded[:streams]]
        current = [r.batch(chunk, 2 * chunk).to("cuda") for r in bounded[:streams]]
        with torch.no_grad():
            _, incoming = model.scan(
                before,
                [prepared.get(b.recording, EventGraphMambaState()) for b in before],
                detach=True,
            )
        model.zero_grad(set_to_none=True)
        outputs, following = model.scan(current, incoming, detach=True)
        nc = sum(int((b.scored & (b.labels >= 0)).sum()) for b in current)
        npayload = sum(
            int(b.lengths.sum())
            for b in current
            if b.verified_normal and b.payload_length_verified
        )

        def objective(output, batch):
            c, _, p, _ = model.loss_sums(output, batch)
            return c / max(1, nc) + model.hparams.payload_weight * p / max(1, npayload)

        sum(objective(o, b) for o, b in zip(outputs, current, strict=True)).backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        for i, (prefix, batch) in enumerate(zip(before, current, strict=True)):
            with torch.no_grad():
                _, initial = model.scan(
                    [prefix],
                    [prepared.get(prefix.recording, EventGraphMambaState())],
                    detach=True,
                )
            serial, state = model.scan([batch], initial, detach=True)
            for name in outputs[i]:
                torch.testing.assert_close(
                    outputs[i][name], serial[0][name], rtol=5e-4, atol=5e-5
                )
            torch.testing.assert_close(
                following[i].nodes, state[0].nodes, rtol=5e-4, atol=5e-5
            )
            torch.testing.assert_close(
                following[i].previous_bus, state[0].previous_bus, rtol=5e-4, atol=5e-5
            )
            for actual, expected in zip(
                following[i].caches, state[0].caches, strict=True
            ):
                for x, y in zip(actual, expected, strict=True):
                    torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)
            objective(serial[0], batch).backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                gradients[name], parameter.grad, rtol=1e-2, atol=1e-4, msg=name
            )
        model.zero_grad(set_to_none=True)
        model.train()

    history_cache = None
    initial_digest = None

    def trial(chunk, streams, baseline=False, profile=False):
        nonlocal history_cache, initial_digest
        gc.collect()
        torch.cuda.empty_cache()
        (
            random.seed(42),
            np.random.seed(42),
            torch.manual_seed(42),
            torch.cuda.manual_seed_all(42),
        )
        model = build_component(cfg.config["model"]).cuda().train()
        if baseline:
            model._readout = types.MethodType(original_readout, model)
        optimizer = model.configure_optimizers()
        # All configurations traverse the same bounded cohort, with continuous
        # prefixes and separate histories. Limit each stream to an equal prefix.
        bounded = [Segment(r) for r in events]
        import hashlib
        from dataclasses import replace

        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        weight_digest = digest.hexdigest()
        if initial_digest is None:
            initial_digest = weight_digest
        if weight_digest != initial_digest:
            raise RuntimeError(
                "benchmark configurations must start with identical weights"
            )
        model.eval()
        history_started = time.perf_counter()
        if history_cache is None:
            history_cache = {}
            with torch.no_grad():
                for source in events:
                    state = EventGraphMambaState()
                    offset = offsets[source.recording.key]
                    for lo in range(0, offset, 2048):
                        _, state = model.stream(
                            source.batch(lo, min(lo + 2048, offset)).to("cuda"), state
                        )
                    if offset:
                        history_cache[source.recording.key] = state
        # These histories depend only on identical initial temporal weights and
        # physical frames. Reset each trial with cloned caches, never shared updates.
        prepared = {
            key: replace(
                state,
                caches=[(c.clone(), m.clone()) for c, m in state.caches],
                nodes=state.nodes.clone(),
                previous_bus=state.previous_bus.clone(),
            )
            for key, state in history_cache.items()
        }
        torch.cuda.synchronize()
        history_seconds = time.perf_counter() - history_started
        dataset = EventStreamWindows(bounded, chunk, streams, 8192)
        if steps > len(dataset) or steps < 1:
            raise ValueError(f"steps must be between 1 and {len(dataset)}")
        check_configuration(model, bounded, chunk, streams, prepared)
        states = dict(prepared)
        iterator = iter(dataset)
        timings = dict(
            cpu_preparation_s=0.0,
            host_to_device_s=0.0,
            forward_s=0.0,
            backward_s=0.0,
            optimizer_s=0.0,
        )
        counted = 0
        optimizer_steps = 0
        # First optimizer window warms kernels and allocator; reset all state,
        # weights and RNG afterwards so measured trials share initialization.
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        trace = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
            if profile
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(steps + 1):
            if step == 1:
                model.load_state_dict(initial)
                optimizer = model.configure_optimizers()
                states, iterator = dict(prepared), iter(dataset)
                (
                    random.seed(42),
                    np.random.seed(42),
                    torch.manual_seed(42),
                    torch.cuda.manual_seed_all(42),
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if trace:
                    trace.__enter__()
                started = time.perf_counter()
            begin = time.perf_counter()
            window = next(iterator)
            if step:
                timings["cpu_preparation_s"] += time.perf_counter() - begin
            begin = time.perf_counter()
            window = window.pin_memory().to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            if step:
                timings["host_to_device_s"] += time.perf_counter() - begin
            optimizer.zero_grad(set_to_none=True)
            for batches in window.groups:
                begin = time.perf_counter()
                outputs, following = model.scan(
                    batches,
                    [states.get(b.recording, EventGraphMambaState()) for b in batches],
                    detach=True,
                )
                sums = [
                    model.loss_sums(o, b) for o, b in zip(outputs, batches, strict=True)
                ]
                loss = sum(
                    c / max(1, window.classification_targets)
                    + model.hparams.payload_weight * p / max(1, window.payload_targets)
                    for c, _, p, _ in sums
                )
                torch.cuda.synchronize()
                if step:
                    timings["forward_s"] += time.perf_counter() - begin
                begin = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                if step:
                    timings["backward_s"] += time.perf_counter() - begin
                for batch, state in zip(batches, following, strict=True):
                    if batch.recording_end:
                        states.pop(batch.recording, None)
                    else:
                        states[batch.recording] = state
                if trace and step:
                    trace.step()
            begin = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            if step:
                timings["optimizer_s"] += time.perf_counter() - begin
                counted += sum(len(b.ids) for g in window.groups for b in g)
                optimizer_steps += 1
            if torch.cuda.max_memory_reserved() > 32 * 1024**3:
                raise RuntimeError("32 GiB reserved memory ceiling exceeded")
        elapsed = time.perf_counter() - started
        result = dict(
            chunk_events=chunk,
            streams=streams,
            baseline=baseline,
            correctness="passed: batched/serial outputs, nonzero caches, gradients",
            scored_events=counted,
            optimizer_steps=optimizer_steps,
            effective_events=8192,
            continuous_history_preparation_s=history_seconds,
            initial_weight_sha256=weight_digest,
            elapsed_s=elapsed,
            events_per_second=counted / elapsed,
            optimizer_step_s=elapsed / optimizer_steps,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            breakdown=timings,
            projected_epoch_s=source_epoch_events / (counted / elapsed),
        )
        if trace:
            trace.__exit__(None, None, None)
            trace.export_chrome_trace(str(output / "profile.json"))
            (output / "profile-operators.txt").write_text(
                trace.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            )
            result["profile"] = "profile.json"
            result["gpu_activity"] = summarize_trace(output / "profile.json")
        return result

    combinations = [(512, 1, True)] + [
        (c, s, False)
        for c in (256, 512, 1024, 2048)
        for s in (1, 2, 4, 8)
        if c * s <= 8192
    ]
    for chunk, streams, baseline in combinations:
        try:
            result = trial(chunk, streams, baseline)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            result = dict(
                chunk_events=chunk, streams=streams, baseline=baseline, error=str(exc)
            )
        results["configurations"].append(result)
        persist(output / "benchmark.json", results)
        print(json.dumps(result), flush=True)
    passed = [
        r for r in results["configurations"] if not r["baseline"] and "error" not in r
    ]
    if not passed:
        raise RuntimeError(
            "all performance configurations failed; inspect benchmark.json"
        )
    best = max(passed, key=lambda r: r["events_per_second"])
    results["selected"] = best
    results["profiled"] = trial(best["chunk_events"], best["streams"], profile=True)
    results["baseline_speedup"] = (
        best["events_per_second"] / results["configurations"][0]["events_per_second"]
    )
    persist(output / "benchmark.json", results)


def summarize_trace(path):
    """Measure device activity and gaps directly from the CUDA trace."""
    trace = json.loads(path.read_text())
    kernels = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel" and e.get("ph") == "X"
    )
    if not kernels:
        raise RuntimeError("profiler captured no CUDA kernels")
    first, end = kernels[0]
    busy, gaps = 0.0, []
    for lo, hi in kernels[1:]:
        if lo > end:
            busy += end - first
            gaps.append(lo - end)
            first, end = lo, hi
        else:
            end = max(end, hi)
    busy += end - first
    span = kernels[-1][1] - kernels[0][0]
    gaps.sort()
    return dict(
        kernel_count=len(kernels),
        kernel_span_ms=span / 1000,
        kernel_busy_ms=busy / 1000,
        active_fraction=busy / span,
        idle_gap_total_ms=sum(gaps) / 1000,
        idle_gap_median_us=gaps[len(gaps) // 2] if gaps else 0.0,
        idle_gap_max_us=max(gaps, default=0.0),
    )


def public_smoke(output):
    """Exercise the public Ray launch with a complete tiny synthetic epoch."""
    from dataclasses import asdict

    import yaml

    from graphids.core.data.recordings import CANRecordingManifest
    from tests.event_fixtures import write_record

    train = write_record(
        output,
        key="smoke-train",
        normal=True,
        rows=[(0.0, "A", "00", 0), (0.01, "B", "01", 0), (0.02, "A", "02", 0)],
    )
    val = write_record(
        output,
        key="smoke-val",
        role="val_query",
        rows=[(0.0, "1", "00", 0), (0.01, "2", "FF", 1), (0.02, "1", "02", 0)],
    )
    manifest = output / "smoke-manifest.json"
    manifest.write_text(
        json.dumps(asdict(CANRecordingManifest((train, val), provenance_reviewed=True)))
    )
    config = dict(
        experiment_name="event-public-smoke",
        dataset="fixture",
        seed=42,
        representation_cfg=dict(kind="can_event", schema_version=1),
        resources=dict(accelerator="gpu", cpus_per_worker=2, gpus_per_worker=1),
        config=dict(
            model=dict(
                type="event_graph_mamba",
                width=8,
                layers=2,
                d_state=3,
                expand=1,
                backend="official",
                dropout=0.0,
            ),
            data=dict(
                type="can_event_dm",
                manifest_path=str(manifest),
                datasets=["set_01"],
                cache_root=str(output / "smoke-cache"),
                chunk_events=2,
            ),
            trainer=dict(
                accelerator="gpu",
                devices=1,
                max_epochs=1,
                num_sanity_val_steps=0,
                enable_progress_bar=False,
                enable_checkpointing=False,
            ),
        ),
    )
    path = output / "smoke.yml"
    path.write_text(yaml.safe_dump(config))
    env = dict(
        os.environ,
        GRAPHIDS_RUN_ROOT=str(output / "public-runs"),
        GRAPHIDS_MLFLOW_MODE="offline",
        RAY_TMPDIR=str(Path(os.environ.get("TMPDIR", "/tmp")) / "event-ray"),
    )
    with (output / "public-smoke.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "graphids", "exp", "launch", str(path)],
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", default="configs/experiments/graph_mamba_events_source.yml"
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run this gate in a Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running", mode=args.mode, slurm_job=os.environ["SLURM_JOB_ID"]
    )
    persist(args.output / "gate.json", result)
    try:
        command = [sys.executable, "-m", "pytest", "-q"]
        if args.mode == "gpu":
            command += [
                "tests/test_event_graph_mamba.py",
                "tests/test_event_continuous.py",
            ]
        with (args.output / "correctness.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.mode == "gpu":
            benchmark(args.config, args.output, args.steps)
            public_smoke(args.output)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="incomplete", error=repr(exc))
        raise
    finally:
        persist(args.output / "gate.json", result)


if __name__ == "__main__":
    main()
