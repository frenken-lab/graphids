"""CPU lifecycle plans shared by training and carried inference.

Only indices cross to CUDA; discovering segments never reads one CUDA scalar
per node. Plans are ephemeral and do not change the snapshot cache schema.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentGroup:
    segments: list[tuple[int, int, int, int, int]]
    indices: torch.Tensor


@dataclass
class ExecutionPlan:
    groups: list[SegmentGroup]
    live: np.ndarray
    generation: np.ndarray
    bins: np.ndarray
    scored: np.ndarray

    @classmethod
    def build(cls, batch):
        live = batch.live.detach().cpu().numpy().copy()
        generation = batch.generation.detach().cpu().numpy().copy()
        b, t, n = live.shape
        groups = {}
        for bi in range(b):
            for ni in range(n):
                valid, gen = live[bi, :, ni], generation[bi, :, ni]
                starts = valid & np.r_[True, ~valid[:-1] | (gen[1:] != gen[:-1])]
                stops = valid & np.r_[~valid[1:] | (gen[1:] != gen[:-1]), True]
                for start, stop in zip(
                    np.flatnonzero(starts), np.flatnonzero(stops) + 1, strict=True
                ):
                    groups.setdefault(int(stop - start), []).append(
                        (bi, ni, int(start), int(stop), int(gen[start]))
                    )
        packed = []
        for length, segments in groups.items():
            offsets = torch.tensor(
                [(bi * t + start) * n + ni for bi, ni, start, _, _ in segments]
            )
            indices = offsets[:, None] + torch.arange(length)[None] * n
            packed.append(SegmentGroup(segments, indices))
        return cls(
            packed,
            live,
            generation,
            batch.bin_indices.detach().cpu().numpy().copy(),
            batch.scored.detach().cpu().numpy().copy(),
        )


def execution_plan(batch):
    # CPU callers can edit masks in place, so don't retain stale metadata there.
    if batch.execution is not None and batch.live.device.type != "cpu":
        return batch.execution
    return ExecutionPlan.build(batch)
