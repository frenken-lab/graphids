#!/usr/bin/env python3
"""Unified bounded correctness, public-launch and single-A100 performance gate.

Run inside Slurm. No target data and no validation accuracy tuning.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def persist(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def benchmark(
    config,
    output,
    *,
    max_microbatch_events,
    effective_events,
    chunks,
    stream_counts,
    repeats=3,
    workers,
    cuda_prefetch,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    profiles=True,
    events_per_recording=65536,
):
    import numpy as np
    import torch

    from graphids.core.data.events.data import EventStreamWindows, EventTrainingLoader
    from graphids.core.models.event_graph_mamba import (
        _GRAPH_WORKSPACE_BYTES,
        EventGraphMambaState,
    )
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.runtime import build_component
    from graphids.primitives_models import EventGraphMambaCfg
    from tests.event_reference import serial_loss_sums, serial_scan

    if not torch.cuda.is_available():
        raise RuntimeError("official CUDA required")
    # Enforce the existing ceiling during correctness checks as well as timed
    # steps, before a larger stream group can consume all physical VRAM.
    memory_limit = 32 * 1024**3
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, memory_limit / torch.cuda.get_device_properties(0).total_memory)
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not 1 <= max_microbatch_events <= 262144 or effective_events < 1:
        raise ValueError(
            "use a positive microbatch cap <=262144 and a positive effective window"
        )
    if repeats < 1:
        raise ValueError("repeats must be positive")
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
    # Include a complete, identical bounded cohort in every repetition, even
    # for the largest chunk. No recording is replicated to fill stream slots.
    if events_per_recording < max(chunks):
        raise ValueError("timed recording segment must cover the largest chunk")
    segment_events = events_per_recording
    model_spec = EventGraphMambaCfg.model_validate(cfg.config["model"]).model_dump()
    # Raise routing capacity explicitly for the requested chunks, without
    # changing model dimensions or the independent 32 GiB allocator ceiling.
    required_tokens = max(chunks) * max(len(r.first_ids) for r in events)
    model_spec["max_graph_tokens"] = max(
        model_spec["max_graph_tokens"],
        1 << (required_tokens - 1).bit_length(),
    )
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
            self.count = min(segment_events, source.count - self.offset)

        def batch(self, start, stop):
            from dataclasses import replace

            return replace(
                self.source.batch(
                    self.offset + start,
                    self.offset + stop,
                ),
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
            timed_events=min(segment_events, r.count - offsets[r.recording.key]),
            timed_positive_frames=int(
                (
                    r.arrays["labels"][
                        offsets[r.recording.key] : offsets[r.recording.key]
                        + segment_events
                    ]
                    == 1
                ).sum()
            ),
        )
        for r in events
    ]
    import hashlib

    diff = subprocess.check_output(["git", "diff", "--binary"])
    (output / "working-tree.diff").write_bytes(diff)
    (output / "configuration.yml").write_bytes(Path(config).read_bytes())
    source_files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    ).splitlines()
    source_hashes = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in source_files
        if name.startswith(("graphids/", "tests/", "scripts/", "configs/"))
        and Path(name).is_file()
    }
    persist(output / "source-sha256.json", source_hashes)
    persist(output / "resolved-configuration.json", cfg.model_dump(mode="json"))
    results = dict(
        workers=workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if workers else None,
        persistent_workers=persistent_workers if workers else False,
        source_manifest_sha256=hashlib.sha256(
            (output / "source-sha256.json").read_bytes()
        ).hexdigest(),
        cuda_prefetch=cuda_prefetch,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        working_tree_diff_sha256=hashlib.sha256(diff).hexdigest(),
        configuration_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(),
        seed=42,
        precision="FP32; TF32 disabled",
        manifest_digest=manifest.digest,
        source_epoch_events=source_epoch_events,
        cohort=cohort,
        configurations=[],
        max_microbatch_events=max_microbatch_events,
        effective_events=effective_events,
        events_per_recording=segment_events,
        repeats=repeats,
        model_config=model_spec,
        memory_limit_bytes=memory_limit,
        graph_workspace_bytes=_GRAPH_WORKSPACE_BYTES,
        note="bounded training segments with continuous preceding history; no validation tuning or full learning run",
    )
    persist(output / "benchmark.json", results)

    def check_configuration(model, bounded, chunk, streams, prepared):
        """Outputs, caches and gradients versus independent serial execution."""
        model.eval()

        # Parity diagnostics may read beyond the timed segment, but only from
        # these same reviewed source recordings. This permits a one-chunk timed
        # segment while still checking a full chunk with nonzero incoming state.
        def diagnostic_batch(segment, start, stop):
            return segment.source.batch(
                segment.offset + start,
                segment.offset + stop,
            ).to("cuda")

        available = min(r.source.count - r.offset for r in bounded[:streams])
        prefix_length = min(chunk, available // 2)
        current_length = min(chunk, available - prefix_length)
        before = (
            [diagnostic_batch(r, 0, prefix_length) for r in bounded[:streams]]
            if prefix_length
            else [None] * streams
        )
        current = [
            diagnostic_batch(r, prefix_length, prefix_length + current_length)
            for r in bounded[:streams]
        ]
        incoming = [prepared.get(b.recording, EventGraphMambaState()) for b in current]
        if prefix_length:
            with torch.no_grad():
                _, incoming = model.scan(before, incoming, detach=True)
        model.zero_grad(set_to_none=True)
        outputs, following = model.scan(current, incoming, detach=True)
        nc = sum(int((b.scored & (b.labels >= 0)).sum()) for b in current)
        npayload = sum(
            int(b.lengths.sum())
            for b in current
            if b.verified_normal and b.payload_length_verified
        )

        def objective(output, batch):
            c, _, p, _ = serial_loss_sums(output, batch)
            return c / max(1, nc) + model.hparams.payload_weight * p / max(1, npayload)

        c, _, p, _ = model.batched_loss_sums(outputs, current)
        (
            c / max(1, nc) + model.hparams.payload_weight * p / max(1, npayload)
        ).backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        outputs = [
            {name: value.detach().cpu() for name, value in output.items()}
            for output in outputs
        ]
        model.zero_grad(set_to_none=True)
        for i, (prefix, batch) in enumerate(zip(before, current, strict=True)):
            initial = [prepared.get(batch.recording, EventGraphMambaState())]
            if prefix is not None:
                with torch.no_grad():
                    _, initial = serial_scan(model, [prefix], initial, detach=True)
            serial, state = serial_scan(
                model,
                [batch],
                initial,
                detach=True,
            )
            for name in outputs[i]:
                torch.testing.assert_close(
                    outputs[i][name],
                    serial[0][name].detach().cpu(),
                    rtol=5e-4,
                    atol=5e-5,
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

    def trial(chunk, streams, profile=False):
        nonlocal history_cache, initial_digest
        gc.collect()
        torch.cuda.empty_cache()
        (
            random.seed(42),
            np.random.seed(42),
            torch.manual_seed(42),
            torch.cuda.manual_seed_all(42),
        )
        model = build_component(model_spec).cuda().train()
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
        dataset = EventStreamWindows(bounded, chunk, streams, effective_events)
        measured_steps = len(dataset)
        expected_coverage = {r.recording.key: r.offset + r.count for r in bounded}
        check_configuration(model, bounded, chunk, streams, prepared)
        loader = EventTrainingLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            cuda_prefetch=cuda_prefetch,
            generator=torch.Generator().manual_seed(42),
            **(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": persistent_workers,
                }
                if workers
                else {}
            ),
        )
        states = dict(prepared)
        iterator = iter(loader)
        timings = dict(
            cpu_preparation_s=0.0,
            host_to_device_s=0.0,
            forward_s=0.0,
            backward_s=0.0,
            optimizer_s=0.0,
        )
        phase_events = []

        def phase_start():
            if profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event

        def phase_end(name, begin, measured):
            if profile and measured:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                phase_events.append((name, begin, end))

        counted = 0
        coverage = {}
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
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
            )
            if profile
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(measured_steps + 1):
            if step == 1:
                model.load_state_dict(initial)
                optimizer = model.configure_optimizers()
                states, iterator = dict(prepared), iter(loader)
                coverage = {}
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
            begin = phase_start()
            window = window.to("cuda", non_blocking=True)
            phase_end("host_to_device_s", begin, step)
            optimizer.zero_grad(set_to_none=True)
            for batches in window.groups:
                begin = phase_start()
                outputs, following = model.scan(
                    batches,
                    [states.get(b.recording, EventGraphMambaState()) for b in batches],
                    detach=True,
                )
                c, _, p, _ = model.batched_loss_sums(outputs, batches)
                loss = c / max(
                    1, window.classification_targets
                ) + model.hparams.payload_weight * p / max(1, window.payload_targets)
                phase_end("forward_s", begin, step)
                begin = phase_start()
                loss.backward()
                phase_end("backward_s", begin, step)
                for batch, state in zip(batches, following, strict=True):
                    if step:
                        start = (
                            prepared[batch.recording].next_row
                            if batch.recording in prepared
                            else 0
                        )
                        if batch.row_start != coverage.get(batch.recording, start):
                            raise ValueError("duplicate/missing benchmark scored rows")
                        coverage[batch.recording] = state.next_row
                    if batch.recording_end:
                        states.pop(batch.recording, None)
                    else:
                        states[batch.recording] = state
                if trace and step:
                    trace.step()
            begin = phase_start()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            phase_end("optimizer_s", begin, step)
            if step:
                counted += sum(len(b.ids) for g in window.groups for b in g)
                optimizer_steps += 1
            if torch.cuda.max_memory_reserved() > 32 * 1024**3:
                raise RuntimeError("32 GiB reserved memory ceiling exceeded")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for name, begin, end in phase_events:
            timings[name] += begin.elapsed_time(end) / 1000
        if coverage != expected_coverage or counted != sum(r.count for r in bounded):
            raise ValueError("incomplete bounded benchmark cohort coverage")
        result = dict(
            chunk_events=chunk,
            streams=streams,
            correctness="passed: batched/serial outputs, nonzero caches, gradients",
            scored_events=counted,
            coverage=coverage,
            optimizer_steps=optimizer_steps,
            effective_events=effective_events,
            continuous_history_preparation_s=history_seconds,
            initial_weight_sha256=weight_digest,
            elapsed_s=elapsed,
            events_per_second=counted / elapsed,
            optimizer_step_s=elapsed / optimizer_steps,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            breakdown=timings
            if profile
            else {"cpu_preparation_enqueue_s": timings["cpu_preparation_s"]},
            profiled=profile,
            trace_scope="first two stream groups; CUDA-event phases cover full cohort"
            if profile
            else None,
            projected_epoch_s=source_epoch_events / (counted / elapsed),
        )
        if trace:
            trace.__exit__(None, None, None)
            trace.export_chrome_trace(str(output / f"profile-{chunk}-{streams}.json"))
            averages = trace.key_averages()
            (output / f"profile-{chunk}-{streams}-operators.txt").write_text(
                averages.table(sort_by="self_device_time_total", row_limit=40)
            )
            result["incoming_correction_trace"] = [
                dict(
                    calls=e.count,
                    cuda_ms=e.device_time_total / 1000,
                    cpu_ms=e.cpu_time_total / 1000,
                )
                for e in averages
                if e.key == "mamba.incoming_state_correction" and e.cpu_time_total > 0
            ]
            result["profile"] = f"profile-{chunk}-{streams}.json"
            result["gpu_activity"] = summarize_trace(
                output / f"profile-{chunk}-{streams}.json"
            )
        return result

    combinations = list(dict.fromkeys((c, s) for c in chunks for s in stream_counts))
    if not combinations or any(
        c < 1
        or s < 1
        or s > min(16, len(events))
        or c * s > min(max_microbatch_events, effective_events)
        for c, s in combinations
    ):
        raise ValueError(
            "every requested chunk/stream configuration must fit the cohort, microbatch cap and optimizer window"
        )
    for chunk, streams in combinations:
        try:
            samples = [trial(chunk, streams) for _ in range(repeats)]
            samples.sort(key=lambda r: r["events_per_second"])
            result = dict(samples[len(samples) // 2])
            result["samples"] = samples
            result["peak_allocated_bytes"] = max(
                s["peak_allocated_bytes"] for s in samples
            )
            result["peak_reserved_bytes"] = max(
                s["peak_reserved_bytes"] for s in samples
            )
            result["selection_statistic"] = (
                "median events/s across independently reset repetitions"
            )
        except torch.OutOfMemoryError as exc:
            result = dict(
                chunk_events=chunk,
                streams=streams,
                error=repr(exc),
            )
        results["configurations"].append(result)
        persist(output / "benchmark.json", results)
        print(
            json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True
        )
    passed = [r for r in results["configurations"] if "error" not in r]
    if not passed:
        raise RuntimeError(
            "all performance configurations failed; inspect benchmark.json"
        )
    best = max(passed, key=lambda r: r["events_per_second"])
    results["selected"] = best
    results["profiled"] = []
    for candidate in passed if profiles else []:
        results["profiled"].append(
            trial(candidate["chunk_events"], candidate["streams"], profile=True)
        )
    comparison = results["configurations"][0]
    results["speedup_over_first_requested"] = (
        best["events_per_second"] / comparison["events_per_second"]
        if "error" not in comparison
        else None
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
    span = max(hi for _, hi in kernels) - kernels[0][0]
    transfers = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X"
    ]
    gaps.sort()
    return dict(
        memcpy_count=len(transfers),
        memcpy_total_ms=sum(e["dur"] for e in transfers) / 1000,
        kernel_count=len(kernels),
        kernel_span_ms=span / 1000,
        kernel_busy_ms=busy / 1000,
        active_fraction=busy / span,
        idle_gap_total_ms=sum(gaps) / 1000,
        idle_gap_median_us=gaps[len(gaps) // 2] if gaps else 0.0,
        idle_gap_max_us=max(gaps, default=0.0),
    )


def public_smoke(output, *, selected_config=None):
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
    if selected_config is not None:
        from graphids.exp.config import ExperimentConfig

        selected = ExperimentConfig.from_yaml(selected_config)
        config["config"]["model"] = dict(selected.config["model"])
        for key in (
            "chunk_events",
            "stream_batch_size",
            "effective_batch_events",
            "max_microbatch_events",
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "cuda_prefetch",
            "pin_memory",
        ):
            if key not in selected.config["data"]:
                continue
            config["config"]["data"][key] = selected.config["data"][key]
        config["config"]["trainer"]["precision"] = "32-true"
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


def benchmark_options(args):
    """Use the same typed defaults and explicit overrides as public training."""
    from graphids.exp.config import ExperimentConfig
    from graphids.primitives_data import CANEventDMCfg

    cfg = ExperimentConfig.from_yaml(args.config)
    data = CANEventDMCfg.model_validate(cfg.config["data"])

    def chosen(name, default):
        value = getattr(args, name)
        return default if value is None else value

    return dict(
        max_microbatch_events=chosen(
            "max_microbatch_events", data.max_microbatch_events
        ),
        effective_events=chosen("effective_events", data.effective_batch_events),
        chunks=chosen("chunks", [data.chunk_events]),
        stream_counts=chosen("streams", [data.stream_batch_size]),
        workers=chosen("workers", data.num_workers),
        cuda_prefetch=chosen("cuda_prefetch", data.cuda_prefetch),
        pin_memory=data.pin_memory,
        prefetch_factor=data.prefetch_factor,
        persistent_workers=data.persistent_workers,
        repeats=args.repeats,
        profiles=not args.no_profiles,
        events_per_recording=args.events_per_recording,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", default="configs/experiments/graph_mamba_events_source.yml"
    )
    parser.add_argument("--max-microbatch-events", type=int)
    parser.add_argument("--effective-events", type=int)
    parser.add_argument("--chunks", type=int, nargs="+")
    parser.add_argument("--streams", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--cuda-prefetch", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--no-profiles", action="store_true")
    parser.add_argument("--events-per-recording", type=int, default=65536)
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
        with (args.output / "correctness.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.mode in ("gpu", "benchmark"):
            benchmark(args.config, args.output, **benchmark_options(args))
            if args.mode == "gpu":
                public_smoke(args.output, selected_config=args.config)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="incomplete", error=repr(exc))
        raise
    finally:
        persist(args.output / "gate.json", result)


if __name__ == "__main__":
    main()
