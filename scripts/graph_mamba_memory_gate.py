"""Preflight the configured maximum tensor geometry on one GPU."""

import argparse
import json
import time
from pathlib import Path

import torch

from graphids.core.data.snapshots.batch import CANSnapshotBatch
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint


def capacity_batch(bins, nodes, neighbors):
    b, t, n, k = 1, bins, nodes, neighbors
    x = torch.rand(b, t, n, 23, device="cuda")
    x[..., 12:20] = 1
    live = torch.ones(b, t, n, dtype=torch.bool, device="cuda")
    times = torch.arange(t, device="cuda")[None]
    batch = CANSnapshotBatch(
        x,
        live,
        torch.zeros_like(live, dtype=torch.long),
        torch.zeros(b, n, 4, device="cuda"),
        torch.ones(b, n, dtype=torch.bool, device="cuda"),
        torch.ones(b, dtype=torch.bool, device="cuda"),
        (
            torch.arange(n, device="cuda")[:, None]
            + torch.arange(1, k + 1, device="cuda")[None]
        )
        % n,
        torch.ones(b, n, k, 2, device="cuda"),
        torch.ones(b, n, k, dtype=torch.bool, device="cuda"),
        torch.zeros(b, t, dtype=torch.long, device="cuda"),
        torch.ones(b, t, dtype=torch.bool, device="cuda"),
        torch.ones(b, t, dtype=torch.bool, device="cuda"),
        torch.ones(b, t, n, device="cuda"),
        torch.zeros(b, t, n, device="cuda"),
        torch.ones(b, t, n, dtype=torch.bool, device="cuda"),
        torch.zeros(b, t, device="cuda"),
        torch.full((b, t), 0.01, device="cuda"),
        times * 0.01,
        torch.ones(b, t, dtype=torch.long, device="cuda"),
        ("capacity_preflight",),
        ("synthetic",),
        ("capacity",),
        (tuple(str(i) for i in range(n)),),
        times,
    )
    batch.neighbors = batch.neighbors.unsqueeze(0)
    return batch


def run(root, bins, nodes, neighbors, width):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    batch = capacity_batch(bins, nodes, neighbors)
    b, t, n, k = 1, bins, nodes, neighbors
    net = VehicleGraphMambaJoint(width=width, backend="official").cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = net.losses(net(batch), batch)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = net.losses(net(batch), batch)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    if reserved > total * 0.85:
        raise AssertionError(
            "worst-geometry preflight exceeds 85% device memory budget"
        )
    net.eval()
    with torch.inference_mode():
        net.stream(batch)
        torch.cuda.synchronize()
    stream_reserved = torch.cuda.max_memory_reserved()
    if stream_reserved > total * 0.85:
        raise AssertionError("chunk inference exceeds 85% device memory budget")
    report = {
        "device": torch.cuda.get_device_name(),
        "bins": t,
        "nodes": n,
        "neighbors": k,
        "width": width,
        "node_bin_tokens": b * t * n,
        "graph_messages": b * t * n * k,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "training_and_stream_reserved_bytes": stream_reserved,
        "device_bytes": total,
        "step_seconds": time.monotonic() - began,
        "loss": float(loss),
        "synthetic_geometry": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory-gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    run(args.output, args.bins, args.nodes, args.neighbors, args.width)
