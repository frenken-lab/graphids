import json

import pytest
import torch

from graphids.core.data.snapshots.datamodule import (
    CANSnapshotDataModule,
    EpisodeSampler,
    RecordingEvalSampler,
)
from scripts.graph_mamba_gate import fixture_manifest


def test_pooled_sources_holdout_context_and_cache(tmp_path):
    path = fixture_manifest(tmp_path / "fixtures")
    dm = CANSnapshotDataModule(
        manifest_path=str(path),
        datasets=["set_01", "hcrl_sa"],
        cache_root=str(tmp_path / "cache"),
        sequence_bins=32,
        warmup_bins=16,
        eval_chunk_bins=37,
    )
    dm.setup()
    assert {dm.records[e[0]].dataset for e in dm.train_data.entries} == {
        "set_01",
        "hcrl_sa",
    }
    assert all(dm.records[e[0]].vehicle != "target" for e in dm.train_data.entries)
    assert len(dm.test_data) == 6
    modes = {e[-1] for e in dm.train_data.entries}
    assert modes == {"zero-shot", "conditioned"}
    sample = dm.train_data[0]
    assert sample.scored.sum() == 32
    assert sample.x.shape[-1] == 23
    assert not dm.train_data[-1].scored.any()
    schedule = EpisodeSampler(dm.train_data, seed=42).schedule()
    assert len(schedule) == len(set(schedule))
    counts = {
        dataset: sum(
            dm.records[dm.train_data.entries[i][0]].dataset == dataset for i in schedule
        )
        for dataset in dm.datasets
    }
    assert len(set(counts.values())) == 1
    again = CANSnapshotDataModule(
        manifest_path=str(path),
        datasets=["set_01", "hcrl_sa"],
        cache_root=str(tmp_path / "cache"),
        sequence_bins=32,
        warmup_bins=16,
    )
    again.setup()
    torch.testing.assert_close(sample.x, again.train_data[0].x)


def test_requested_incompatible_dataset_is_not_silently_dropped(tmp_path):
    path = fixture_manifest(tmp_path / "fixtures")
    with pytest.raises(ValueError, match="missing from reviewed manifest"):
        CANSnapshotDataModule(
            manifest_path=str(path),
            datasets=["hcrl_ch"],
            cache_root=str(tmp_path / "cache"),
        ).setup()


def test_distributed_schedules_do_not_repeat_scored_bins(tmp_path, monkeypatch):
    path = fixture_manifest(tmp_path / "fixtures")
    dm = CANSnapshotDataModule(
        manifest_path=str(path),
        datasets=["set_01", "hcrl_sa"],
        cache_root=str(tmp_path / "cache"),
        sequence_bins=32,
        eval_chunk_bins=37,
    )
    dm.setup()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    samples = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        samples.append(list(dm.train_sampler))
    assert len(samples[0]) == len(samples[1])
    real = [x for part in samples for x in part if x >= 0]
    assert len(real) == len(set(real))
    evaluator = RecordingEvalSampler(dm.val_data)
    rank_groups = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        rank_groups.append(list(evaluator))
    assert len(rank_groups[0]) == len(rank_groups[1])
    seen = [x for part in rank_groups for x in part if x >= 0]
    assert sorted(seen) == list(range(len(dm.val_data)))
    for rank, indices in enumerate(rank_groups):
        owned = {
            (dm.val_entries[i][0], dm.val_entries[i][-1]) for i in indices if i >= 0
        }
        other = {
            (dm.val_entries[i][0], dm.val_entries[i][-1])
            for i in rank_groups[1 - rank]
            if i >= 0
        }
        assert owned.isdisjoint(other)


def test_manifest_content_changes_invalidate_cache(tmp_path):
    path = fixture_manifest(tmp_path / "fixtures")
    manifest = json.loads(path.read_text())
    recording = manifest["records"][0]
    with open(recording["path"], "a") as handle:
        handle.write("4.0,A,00,0\n")
    with pytest.raises(ValueError, match="hash changed"):
        CANSnapshotDataModule(
            manifest_path=str(path),
            datasets=["set_01", "hcrl_sa"],
            cache_root=str(tmp_path / "cache"),
        ).setup()
