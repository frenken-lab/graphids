"""Run the functional cutover gate inside an OSC compute allocation.

Fixtures are synthetic CAN recordings with explicit provenance, not evidence
of real-dataset transfer quality. CPU diagnostics use --backend reference;
the production cutover gate requires --backend official --accelerator gpu.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.data.snapshots.manifest import Recording, file_hash
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint


def fixture_manifest(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    serial = 0
    for dataset, vehicle in (
        ("set_01", "source_a"),
        ("hcrl_sa", "source_b"),
        ("set_02", "target"),
    ):
        roles = (
            (
                "target_reference",
                "target_calibration",
                "target_normal_test",
                "target_attack_test",
            )
            if vehicle == "target"
            else (
                "train_reference",
                "train_query",
                "train_query",
                "val_reference",
                "val_query",
                "val_query",
            )
        )
        for role_index, role in enumerate(roles):
            serial += 1
            key = f"{vehicle}_{role}_{role_index}"
            path = root / f"{key}.csv"
            normal = not (
                role.endswith("attack_test")
                or (role.endswith("query") and role_index % 2 == 0)
            )
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
                for i in range(320):
                    attack = int(not normal and 160 <= i < 240)
                    writer.writerow(
                        [
                            round(i * 0.01, 6),
                            f"{vehicle}_id_{i % 3}",
                            f"{(i + serial + attack * 80) % 256:02x}",
                            attack,
                        ]
                    )
            records.append(
                asdict(
                    Recording(
                        key,
                        dataset,
                        vehicle,
                        key,
                        str(path),
                        file_hash(path),
                        role,
                        verified_normal=normal,
                        payload_length_verified=True,
                        labels_verified=True,
                    )
                )
            )
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {"version": 1, "provenance_reviewed": True, "records": records}, indent=2
        )
    )
    return path


def run_gate(root, *, backend, accelerator, devices, max_epochs):
    torch.set_num_threads(2)
    pl.seed_everything(42, workers=True)
    manifest = fixture_manifest(root / "fixtures")
    # Target vehicle AND dataset are held out from all training gradients.
    dm = CANSnapshotDataModule(
        manifest_path=str(manifest),
        datasets=["set_01", "hcrl_sa"],
        sequence_bins=32,
        warmup_bins=16,
        eval_chunk_bins=37,
        cache_root=str(root / "cache"),
    )
    dm.setup()
    net = VehicleGraphMambaJoint(width=32, layers=2, dropout=0.0, backend=backend)
    net.prepare_from_datamodule(dm)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        logger=False,
        default_root_dir=str(root),
        enable_checkpointing=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
    began = time.monotonic()
    trainer.fit(net, datamodule=dm)
    checkpoint = root / "joint.ckpt"
    trainer.save_checkpoint(checkpoint)
    fit_seconds = time.monotonic() - began
    restored = VehicleGraphMambaJoint.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored.prepare_from_datamodule(dm)
    test_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(root),
        use_distributed_sampler=False,
    )
    test_trainer.test(restored, datamodule=dm)
    restored = restored.eval().to("cuda" if accelerator == "gpu" else "cpu")
    sample = dm.val_data[0].to(restored.device)
    with torch.no_grad():
        sequence = restored(sample)
        streamed, _ = restored.stream(sample)
    for name in sequence:
        torch.testing.assert_close(sequence[name], streamed[name], rtol=2e-4, atol=2e-5)
    if backend == "official":
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as profile:
            restored.train()
            loss, _ = restored.losses(restored(sample), sample)
            loss.backward()
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in profile.events()
            if "selective_scan" in event.name.lower()
            or "causal_conv" in event.name.lower()
        ]
        if not kernels:
            raise AssertionError(
                "profiler did not observe optimized Mamba/convolution operations"
            )
    else:
        kernels = ["explicit reference diagnostic"]
    scores = [
        json.loads(line)
        for line in (root / "artifacts/bin-scores.jsonl").read_text().splitlines()
    ]
    assert scores and {r["context"] == "zero-shot" for r in scores} == {False, True}
    keys = [(r["recording"], r["context"], r["bin"]) for r in scores]
    assert len(keys) == len(set(keys))
    assert all(
        torch.isfinite(torch.tensor([r["classification"], r["anomaly"]])).all()
        for r in scores
    )
    report = {
        "backend": backend,
        "device": torch.cuda.get_device_name() if accelerator == "gpu" else "cpu",
        "fit_seconds": fit_seconds,
        "optimizer_steps": trainer.global_step,
        "scored_test_bins": len(scores),
        "datasets": ["set_01", "hcrl_sa"],
        "synthetic_fixtures": True,
        "kernels": sorted(set(kernels)),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if accelerator == "gpu"
        else None,
    }
    (root / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("official", "reference"), default="official"
    )
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    os.environ["GRAPHIDS_RUN_DIR"] = str(args.output)
    run_gate(
        args.output,
        backend=args.backend,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
    )
