"""Spike: benchmark fast_collate vs Batch.from_data_list across batch sizes.

Tests both cold (no _data_list cache) and warm (cached) paths to find
the crossover point where fast_collate wins.

Run on a compute node:
    python tests/spikes/spike_fast_collate.py
"""

import gc
import time

import torch
from torch_geometric.data import Batch

from graphids.config import cache_dir, data_dir
from graphids.core.preprocessing.datamodule import _make_fast_collate_fn
from graphids.core.preprocessing.datasets.can_bus import CANBusDataset

N_REPEATS = 10
BATCH_SIZES = [50, 100, 200, 300, 500, 750, 1000]


def _bench_old_cold(ds, indices):
    """Old path, cold _data_list cache — what spawn workers hit every first access."""
    # Nuke the cache to simulate a fresh spawn worker
    ds._data_list = None
    gc.collect()
    t0 = time.perf_counter()
    data_list = [ds.get(i) for i in indices]
    Batch.from_data_list(data_list)
    return time.perf_counter() - t0


def _bench_old_warm(ds, indices):
    """Old path, warm cache — what persistent_workers see after epoch 1."""
    # Ensure cache is populated
    if ds._data_list is None or ds._data_list[indices[0]] is None:
        for i in indices:
            ds.get(i)
    t0 = time.perf_counter()
    data_list = [ds.get(i) for i in indices]
    Batch.from_data_list(data_list)
    return time.perf_counter() - t0


def _bench_new(fast_collate, indices):
    """New path — always the same, no cache dependency."""
    t0 = time.perf_counter()
    fast_collate(indices)
    return time.perf_counter() - t0


def main():
    lake = "/fs/ess/PAS1266/kd-gat"
    ds = CANBusDataset(
        root=cache_dir(lake, "set_01"),
        raw_dir=data_dir(lake, "set_01"),
        split="train", window_size=100, stride=100,
    )
    physical = list(ds.indices())
    fast_collate = _make_fast_collate_fn(ds)

    print(f"Dataset: set_01, {len(ds)} graphs, {ds._data.x.shape[0]} total nodes")
    print(f"{'batch':>6} | {'old_cold':>10} {'old_warm':>10} {'new':>10} | {'cold/new':>10} {'warm/new':>10}")
    print("-" * 75)

    # Correctness check on first batch size
    test_idx = physical[:BATCH_SIZES[0]]
    for i in test_idx:
        ds.get(i)
    ref = Batch.from_data_list([ds.get(i) for i in test_idx])
    new = fast_collate(test_idx)
    for attr in ["x", "edge_index", "edge_attr", "node_id", "y", "attack_type", "batch", "ptr"]:
        assert torch.equal(getattr(ref, attr), getattr(new, attr)), f"MISMATCH on {attr}"
    print("Correctness: PASS\n")

    for bs in BATCH_SIZES:
        if bs > len(physical):
            break
        # Pick indices from middle of dataset (avoid edge effects)
        start = len(physical) // 3
        indices = physical[start : start + bs]

        # -- Cold: nuke cache each time, single run (can't avg — cache warms) --
        t_cold = _bench_old_cold(ds, indices)

        # -- Warm: populate cache, then time repeated access -------------------
        # Populate
        ds._data_list = None
        for i in indices:
            ds.get(i)
        # Time
        times_warm = [_bench_old_warm(ds, indices) for _ in range(N_REPEATS)]
        t_warm = sum(times_warm) / N_REPEATS

        # -- New: no state dependency, average directly ------------------------
        # Warmup
        fast_collate(indices)
        times_new = [_bench_new(fast_collate, indices) for _ in range(N_REPEATS)]
        t_new = sum(times_new) / N_REPEATS

        print(
            f"{bs:>6} | {t_cold*1000:>9.1f}ms {t_warm*1000:>9.1f}ms {t_new*1000:>9.1f}ms"
            f" | {t_cold/t_new:>9.1f}x {t_warm/t_new:>9.1f}x"
        )

    print("\ncold = spawn worker first access (no _data_list cache)")
    print("warm = persistent_workers after epoch 1 (_data_list cached)")
    print("new  = fast_collate (always indexes into pre-collated tensors)")
    print("cold/new > 1 = new wins over cold old path")
    print("warm/new > 1 = new wins over warm old path")


if __name__ == "__main__":
    main()
