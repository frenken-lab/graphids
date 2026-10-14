"""Run a reviewed real-data pilot in one GPU Slurm allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint


class PilotTiming(pl.Callback):
    """Separate phase time and per-batch optimizer/inference work."""

    def __init__(self):
        self.report = {}
        self.starts = {}

    def begin(self, name):
        torch.cuda.synchronize()
        self.starts[name] = (time.monotonic(), time.process_time())

    def end(self, name):
        torch.cuda.synchronize()
        wall, cpu = self.starts.pop(name)
        entry = self.report.setdefault(
            name, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0}
        )
        entry["wall_seconds"] += time.monotonic() - wall
        entry["cpu_seconds"] += time.process_time() - cpu
        entry["calls"] += 1

    def on_train_epoch_start(self, trainer, module):
        self.begin("training_epoch")

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.begin("optimizer_batches")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.end("optimizer_batches")

    def on_validation_epoch_start(self, trainer, module):
        # Lightning validation may begin before on_train_epoch_end.
        if "training_epoch" in self.starts:
            self.end("training_epoch")
        self.begin("validation_inference")

    def on_validation_epoch_end(self, trainer, module):
        self.end("validation_inference")
        self.begin("validation_finalize")

    def on_validation_end(self, trainer, module):
        self.end("validation_finalize")

    def on_test_epoch_start(self, trainer, module):
        self.begin("target_inference")

    def on_test_epoch_end(self, trainer, module):
        self.end("target_inference")
        self.begin("target_finalize")

    def on_test_end(self, trainer, module):
        self.end("target_finalize")


def packing_fingerprint(dm, driver_hash):
    """Invalidate packing evidence on data, geometry or implementation changes."""
    geometry = (
        "datasets",
        "seed",
        "sequence_bins",
        "warmup_bins",
        "context_withholding",
        "neighbor_cap",
        "retention_seconds",
        "anchor_capacity",
        "transient_capacity",
        "max_node_bin_tokens",
        "max_graph_messages",
        "max_snapshot_bytes",
    )
    source_root = Path(__file__).resolve().parents[1] / "graphids/core/data/snapshots"
    return hashlib.sha256(
        json.dumps(
            {
                "manifest": dm.manifest.digest,
                "representation": asdict(dm.representation_cfg),
                "driver": driver_hash,
                "config": {key: getattr(dm, key) for key in geometry},
                "implementation": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(source_root.glob("*.py"))
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--stream-execution", choices=("chunked", "recurrent"), default="chunked"
    )
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPHIDS_RUN_DIR"] = str(root)
    torch.set_num_threads(2)
    pl.seed_everything(42, workers=True)
    driver = Path(__file__).read_bytes()
    (root / "pilot-driver.py").write_bytes(driver)
    report = {
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "backend": "official",
        "seed": 42,
        "driver_sha256": hashlib.sha256(driver).hexdigest(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "train_batch_limit": args.train_batches,
        "scope": f"{args.train_batches}-window fit diagnostic; whole-recording validation and paired target evaluation",
        "quality_acceptance": False,
    }

    def status(stage):
        report["stage"] = stage
        (root / "pilot.json").write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    started = time.monotonic()
    status("source_preflight")
    dm = CANSnapshotDataModule(
        manifest_path=str(args.manifest.resolve()),
        datasets=["set_01"],
        num_workers=args.num_workers,
        pin_memory=not args.preflight_only,
        cache_root=str(args.cache_root or root / "cache"),
    )
    dm.setup()
    census = []
    for rec in dm.records.values():
        if rec.role.startswith("target_"):
            continue
        snapshot = dm.snapshot_for(rec.key, dm.context_for(rec.key))
        census.append(
            {
                "key": rec.key,
                "role": rec.role,
                "bins": len(snapshot.labels),
                "ids": len(snapshot.ids),
                "max_live_nodes": int(snapshot.live.sum(1).max()),
                "snapshot_array_bytes": sum(
                    value.nbytes
                    for value in vars(snapshot).values()
                    if hasattr(value, "nbytes")
                ),
                "empty_bins": int((snapshot.event_count == 0).sum()),
            }
        )
    (root / "source-census.json").write_text(json.dumps(census, indent=2))
    report.update(
        manifest_digest=dm.manifest.digest,
        anchor_capacity=dm.anchor_capacity,
        transient_capacity=dm.transient_capacity,
        available_train_windows=len(dm.train_data),
    )
    # Inspect every source window's packing and workload before fitting.
    fingerprint = packing_fingerprint(dm, report["driver_sha256"])
    preflight_file = root / "packing-preflight.json"
    cached = json.loads(preflight_file.read_text()) if preflight_file.exists() else {}
    if cached.get("fingerprint") == fingerprint:
        max_tokens, max_messages = cached["max_tokens"], cached["max_messages"]
    else:
        max_tokens = max_messages = 0
        for i in range(len(dm.train_data)):
            batch = dm.train_data[i]
            max_tokens = max(max_tokens, batch.live.numel())
            max_messages = max(
                max_messages, int(batch.edge_mask.sum()) * batch.x.shape[1]
            )
        preflight_file.write_text(
            json.dumps(
                dict(
                    fingerprint=fingerprint,
                    max_tokens=max_tokens,
                    max_messages=max_messages,
                )
            )
        )
    report["preflight_cache_hit"] = cached.get("fingerprint") == fingerprint
    report["preflight_seconds"] = time.monotonic() - started
    report.update(
        max_source_node_bin_tokens=max_tokens, max_source_graph_messages=max_messages
    )
    if args.preflight_only:
        status("source_preflight_complete")
        return
    report["device"] = torch.cuda.get_device_name()
    net = VehicleGraphMambaJoint(
        backend="official", stream_execution=args.stream_execution
    )
    net.prepare_from_datamodule(dm)
    timer = PilotTiming()
    trainer = pl.Trainer(
        callbacks=[timer],
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=args.train_batches,
        logger=False,
        default_root_dir=str(root),
        enable_checkpointing=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
    status("fit_and_whole_source_validation")
    began = time.monotonic()
    trainer.fit(net, datamodule=dm)
    checkpoint = root / "joint.ckpt"
    trainer.save_checkpoint(checkpoint)
    report["phase_timings"] = timer.report
    report["model_phase_timings"] = dict(net.phase_timings)
    report.update(
        fit_seconds=time.monotonic() - began,
        optimizer_steps=trainer.global_step,
        checkpoint=str(checkpoint),
        calibrated=bool(net.calibrated),
    )
    restored = VehicleGraphMambaJoint.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored.prepare_from_datamodule(dm)
    evaluation = pl.Trainer(
        callbacks=[timer],
        accelerator="gpu",
        devices=1,
        logger=False,
        default_root_dir=str(root),
        enable_checkpointing=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
    )
    status("whole_target_evaluation_both_modes")
    began = time.monotonic()
    evaluation.test(restored, datamodule=dm)
    report.update(
        test_seconds=time.monotonic() - began,
        coverage=json.loads((root / "artifacts/coverage.json").read_text()),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        total_seconds=time.monotonic() - started,
        phase_timings=timer.report,
    )
    report["model_phase_timings"].update(restored.phase_timings)
    status("complete")


if __name__ == "__main__":
    main()
