"""Same-node two-GPU native evaluation or public Ray fit gate, inside Slurm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from graphids.core.models.score_columns import ScoreTable
from scripts.graph_mamba_gate import fixture_manifest


def native(root, *, device="cuda"):
    local_rank = int(os.environ["LOCAL_RANK"])
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl" if device == "cuda" else "gloo")
    target = local_rank if device == "cuda" else "cpu"
    backend = "official" if device == "cuda" else "reference"
    rank = torch.distributed.get_rank()
    try:
        if rank == 0:
            fixture_manifest(root / "fixtures")
        torch.distributed.barrier()
        dm = CANSnapshotDataModule(
            manifest_path=str(root / "fixtures/manifest.json"),
            datasets=["set_01", "hcrl_sa"],
            cache_root=str(root / "cache"),
            eval_chunk_bins=37,
        )
        dm.setup()
        torch.manual_seed(42)
        net = (
            VehicleGraphMambaJoint(width=32, dropout=0, backend=backend)
            .to(target)
            .eval()
        )
        net.prepare_from_datamodule(dm)
        logged = {}
        net.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))
        net._trainer = SimpleNamespace(
            datamodule=dm, is_global_zero=rank == 0, default_root_dir=str(root)
        )
        reference = None
        if rank == 0:
            reference = (
                VehicleGraphMambaJoint(width=32, dropout=0, backend=backend)
                .to(target)
                .eval()
            )
            reference.load_state_dict(net.state_dict())
            reference.prepare_from_datamodule(dm)
            reference.log = lambda *args, **kwargs: None
        net.on_validation_epoch_start()
        for batch in dm.val_dataloader():
            net._evaluate(batch.to(target), 0, "val")
        net.on_validation_epoch_end()
        all_metrics = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_metrics, logged)
        assert all(metrics == all_metrics[0] for metrics in all_metrics), (
            "ranks must see identical global checkpoint metrics"
        )
        if rank == 0:
            for i in range(len(dm.val_data)):
                reference._evaluate(dm.collate([dm.val_data[i]]).to(target), 0, "val")
            table = ScoreTable(reference._validation_scores)
            table.calibrate(reference)
            for mode in ("zero-shot", "conditioned"):
                left, right = (
                    net.calibration_bundles[mode],
                    reference.calibration_bundles[mode],
                )
                for field in ("center", "scale", "thresholds"):
                    torch.testing.assert_close(
                        torch.tensor(getattr(left, field)),
                        torch.tensor(getattr(right, field)),
                        rtol=2e-4,
                        atol=2e-5,
                    )
                assert left.available == right.available
            for name in (
                "score_center",
                "score_scale",
                "alert_thresholds",
                "calibrated",
            ):
                torch.testing.assert_close(
                    getattr(net, name), getattr(reference, name), rtol=2e-4, atol=2e-5
                )
        for name in ("score_center", "score_scale", "alert_thresholds", "calibrated"):
            buffer = getattr(net, name)
            expected = buffer.clone()
            torch.distributed.broadcast(expected, src=0)
            torch.testing.assert_close(buffer, expected)
        net.on_test_epoch_start()
        for loader_idx, loader in enumerate(dm.test_dataloader()):
            for batch in loader:
                net._evaluate(batch.to(target), loader_idx, "test")
        net.on_test_epoch_end()
        torch.distributed.barrier()
        if rank == 0:
            report = dict(
                passed=True,
                world_size=torch.distributed.get_world_size(),
                device=device,
                backend=backend,
                calibration_matches_single_rank=True,
                global_validation_metrics_match_all_ranks=True,
                coverage=json.loads((root / "artifacts/coverage.json").read_text()),
            )
            (root / "parallel-gate.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
    finally:
        torch.distributed.destroy_process_group()


def ray_gate(root):
    import ray

    from graphids.exp.config import ExperimentConfig
    from graphids.exp.ray_backend import launch_run

    manifest = fixture_manifest(root / "fixtures")
    os.environ["GRAPHIDS_MLFLOW_MODE"] = "disabled"
    cfg = ExperimentConfig(
        experiment_name="graph_mamba_parallel_gate",
        dataset="pooled_can",
        resources={"accelerator": "gpu", "gpus_per_worker": 1, "cpus_per_worker": 4},
        config={
            "model": {
                "type": "vehicle_graph_mamba_joint",
                "backend": "official",
                "width": 32,
                "dropout": 0,
            },
            "data": {
                "type": "can_snapshot_dm",
                "manifest_path": str(manifest),
                "datasets": ["set_01", "hcrl_sa"],
                "cache_root": str(root / "cache"),
                "sequence_bins": 32,
                "warmup_bins": 16,
                "eval_chunk_bins": 37,
            },
            "trainer": {
                "devices": 2,
                "max_epochs": 1,
                "limit_train_batches": 2,
                "num_sanity_val_steps": 0,
                "use_distributed_sampler": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        },
    )
    run = cfg.build_run(name="fit", stage="fit", run_dir=root / "run")
    ray.init(num_cpus=8, num_gpus=2, include_dashboard=False)
    try:
        summary = launch_run(run)
        report = dict(passed=True, summary=str(summary), world_size=2)
        (root / "parallel-gate.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "ray"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPHIDS_RUN_DIR"] = str(args.output.resolve())
    torch.set_num_threads(2)
    if args.mode == "native":
        native(args.output.resolve(), device=args.device)
    else:
        ray_gate(args.output.resolve())


if __name__ == "__main__":
    main()
