"""Two-GPU gate: torchrun --standalone --nproc-per-node=2 this script."""

import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from scripts.graph_mamba_gate import fixture_manifest


def run(root, rank_override=None):
    torch.set_num_threads(2)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if rank_override is None:
        torch.distributed.init_process_group("nccl")
    else:
        rendezvous = root / "rendezvous.json"
        root.mkdir(parents=True, exist_ok=True)
        if rank_override == 0:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            temp = rendezvous.with_suffix(".tmp")
            temp.write_text(json.dumps({"host": socket.getfqdn(), "port": port}))
            temp.replace(rendezvous)
        began = time.monotonic()
        while not rendezvous.exists():
            if time.monotonic() - began > 300:
                raise RuntimeError(
                    "second data-parallel allocation could not rendezvous within five minutes"
                )
            time.sleep(0.2)
        endpoint = json.loads(rendezvous.read_text())
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"tcp://{endpoint['host']}:{endpoint['port']}",
            rank=rank_override,
            world_size=2,
            timeout=timedelta(seconds=300),
        )
    rank = torch.distributed.get_rank()
    root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        fixture_manifest(root / "fixtures")
    torch.distributed.barrier()
    dm = CANSnapshotDataModule(
        manifest_path=str(root / "fixtures/manifest.json"),
        datasets=["set_01", "hcrl_sa"],
        sequence_bins=32,
        warmup_bins=16,
        cache_root=str(root / "cache"),
    )
    dm.setup()
    torch.manual_seed(42)
    net = VehicleGraphMambaJoint(width=32, dropout=0, backend="official").cuda(
        local_rank
    )
    model = DistributedDataParallel(net, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    scored = []
    steps = 0
    losses = []
    for batch in dm.train_dataloader():
        batch = batch.to(local_rank)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = net.losses(output, batch)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
        )
        optimizer.step()
        steps += 1
        losses.append(float(loss))
        for bi, key in enumerate(batch.recording_keys):
            scored.extend(
                (key, batch.context_digests[bi], int(i))
                for i in batch.bin_indices[bi, batch.scored[bi]]
            )
    gathered = [None, None]
    torch.distributed.all_gather_object(
        gathered, {"rank": rank, "scored": scored, "steps": steps, "losses": losses}
    )
    flat = [key for result in gathered for key in result["scored"]]
    assert len(flat) == len(set(flat))
    assert gathered[0]["steps"] == gathered[1]["steps"]
    parameters = torch.cat([p.detach().flatten() for p in net.parameters()])
    other = parameters.clone()
    torch.distributed.broadcast(other, src=0)
    torch.testing.assert_close(parameters, other)
    if rank == 0:
        report = {
            "world_size": 2,
            "device": torch.cuda.get_device_name(local_rank),
            "steps_per_rank": steps,
            "scored_bins": len(flat),
            "duplicate_bins": 0,
            "finite_gradients": True,
            "synchronized_parameters": True,
            "synthetic_fixtures": True,
        }
        (root / "ddp-gate.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rank",
        type=int,
        choices=(0, 1),
        help="Explicit rank for two separate single-GPU allocations",
    )
    args = parser.parse_args()
    run(args.output, args.rank)
