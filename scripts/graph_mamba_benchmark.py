"""Real-checkpoint equivalence and warm performance gate; run in Slurm on CUDA."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from itertools import zip_longest
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from scripts.graph_mamba_real_pilot import PilotTiming


def compare_scores(actual_path, expected_path, thresholds):
    """Stream the original JSONL; exact coverage and tolerance-bounded scores."""
    report = dict(
        bins=0,
        score_violations=0,
        alert_disagreements=0,
        unexplained_alert_disagreements=0,
        max_abs_error={},
    )
    exact = (
        "recording",
        "dataset",
        "vehicle",
        "role",
        "context",
        "bin",
        "close",
        "exposure",
        "label",
        "normal",
        "events",
        "live_nodes",
        "silent_anchors",
        "unavailable_payload_fields",
        "anomaly_component_mask",
        "anomaly_eligible",
    )
    with actual_path.open() as actual, expected_path.open() as expected:
        for left, right in zip_longest(actual, expected):
            if left is None or right is None:
                raise AssertionError("score coverage length differs")
            a, e = json.loads(left), json.loads(right)
            for name in exact:
                if a[name] != e[name]:
                    raise AssertionError(
                        f"{name} differs at bin {report['bins']}: {a[name]} != {e[name]}"
                    )
            for name in ("classification", "anomaly", "anomaly_components"):
                av, ev = np.asarray(a[name]), np.asarray(e[name])
                error = np.abs(av - ev)
                report["max_abs_error"][name] = max(
                    report["max_abs_error"].get(name, 0.0), float(error.max())
                )
                report["score_violations"] += int(
                    np.count_nonzero(error > 2e-5 + 2e-4 * np.abs(ev))
                )
            if a["alert"] != e["alert"]:
                report["alert_disagreements"] += 1
                near = abs(e["classification"] - thresholds[0]) <= 2e-5 + 2e-4 * abs(
                    thresholds[0]
                )
                near |= e["anomaly_eligible"] and abs(
                    e["anomaly"] - thresholds[1]
                ) <= 2e-5 + 2e-4 * abs(thresholds[1])
                report["unexplained_alert_disagreements"] += int(not near)
            report["bins"] += 1
    report["passed"] = (
        report["score_violations"] == report["unexplained_alert_disagreements"] == 0
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--legacy-calibration-replay", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-full-evaluation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPHIDS_RUN_DIR"] = str(args.output.resolve())
    torch.set_num_threads(2)
    pl.seed_everything(42, workers=True)
    dm = CANSnapshotDataModule(
        manifest_path=str(args.manifest),
        datasets=["set_01"],
        cache_root=str(args.cache_root),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dm.setup()
    net = VehicleGraphMambaJoint.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        legacy_calibration_replay=args.legacy_calibration_replay,
    ).cuda()
    net.prepare_from_datamodule(dm)
    sample = dm.collate([dm.val_data[0]]).to("cuda")
    net.eval()

    def measure(fn):
        torch.cuda.synchronize()
        start, cpu = time.perf_counter(), time.process_time()
        fn()
        torch.cuda.synchronize()
        return {
            "wall_seconds": time.perf_counter() - start,
            "cpu_seconds": time.process_time() - cpu,
        }

    report = dict(
        job_id=os.getenv("SLURM_JOB_ID"),
        calibration_track="legacy-pooled-replay"
        if args.legacy_calibration_replay
        else "source-normal",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        worktree_status=subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        device=torch.cuda.get_device_name(),
        timings={},
    )
    for mode in ("chunked", "recurrent"):
        net.hparams.stream_execution = mode
        with torch.inference_mode():
            net.stream(sample)
            trials = [measure(lambda: net.stream(sample)) for _ in range(5)]
        report["timings"][mode] = {
            "trials": trials,
            "median_seconds": statistics.median(row["wall_seconds"] for row in trials),
        }
    net.hparams.stream_execution = "chunked"
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            net.stream(sample)
            torch.cuda.synchronize()
    report["profiler_calls_256_bins"] = {
        entry.key: entry.count
        for entry in profile.key_averages()
        if entry.key
        in (
            "cudaStreamSynchronize",
            "aten::item",
            "aten::clone",
            "cudaMemcpyAsync",
            "cudaLaunchKernel",
        )
    }
    (args.output / "operators.txt").write_text(
        profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
    )
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sample = dm.collate([dm.train_data[dm.train_sampler.schedule()[0]]]).to("cuda")

    def train():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = net.losses(net(sample), sample)
        loss.backward()
        optimizer.step()

    train()
    trials = [measure(train) for _ in range(5)]
    report["timings"]["training"] = {
        "trials": trials,
        "median_seconds": statistics.median(row["wall_seconds"] for row in trials),
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2))
    if not args.skip_full_evaluation:
        # Reload the unmodified original checkpoint after optimizer benchmarking.
        net = VehicleGraphMambaJoint.load_from_checkpoint(
            args.checkpoint,
            map_location="cpu",
            legacy_calibration_replay=args.legacy_calibration_replay,
        )
        net.prepare_from_datamodule(dm)
        timer = PilotTiming()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            use_distributed_sampler=False,
            callbacks=[timer],
            default_root_dir=str(args.output),
        )
        start = time.perf_counter()
        trainer.test(net, datamodule=dm)
        report["test_seconds"] = time.perf_counter() - start
        report["phase_timings"] = timer.report
        report["model_phase_timings"] = net.phase_timings
        report["coverage"] = json.loads(
            (args.output / "artifacts/coverage.json").read_text()
        )
        if args.baseline_scores:
            report["comparison"] = compare_scores(
                args.output / "artifacts/bin-scores.jsonl",
                args.baseline_scores,
                net.alert_thresholds.detach().cpu().tolist(),
            )
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if report.get("comparison", {}).get("passed") is False:
        raise AssertionError(
            "real-checkpoint equivalence gate failed; see benchmark.json"
        )


if __name__ == "__main__":
    main()
