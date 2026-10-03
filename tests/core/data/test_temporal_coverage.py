from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import TemporalData

from graphids.core.data.datamodule.coverage import (
    CoverageBitmap,
    TemporalCoverageAudit,
    validate_full_lane_settings,
)
from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader
from graphids.core.data.datamodule.temporal import TemporalDataModule
from graphids.core.data.state import clear_cache
from graphids.core.models.temporal._hybrid.targets import lane_targets


def data_for(lengths):
    n = sum(lengths)
    idx = torch.arange(n)
    sid = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    reset = torch.zeros(n, dtype=torch.bool)
    if n:
        reset[torch.tensor(lengths).cumsum(0) - 1] = True
    return TemporalData(src=idx % 4, dst=(idx + 1) % 4, t=idx.float(),
                        msg=idx.float().view(-1, 1).expand(-1, 17).clone(), y=idx % 2,
                        stream_id=sid, event_id=idx, is_scored=torch.ones(n, dtype=torch.bool),
                        reset_after=reset)


@pytest.mark.parametrize("workers,lengths,lanes,interleaved", [
    (0, [1, 7, 2, 10], 2, False),
    (2, [1, 7, 2, 10], 2, True),
    (0, [1, 2], 8, True),
    (2, [17], 4, False),
])
def test_complete_ordered_coverage(workers, lengths, lanes, interleaved):
    data = data_for(lengths)
    if interleaved:
        # Interleave recordings while retaining within-recording chronology.
        order = sorted(range(sum(lengths)), key=lambda i: (int(data.t[i]) - sum(lengths[:int(data.stream_id[i])]), int(data.stream_id[i])))
        data = data[torch.tensor(order)]
    dataset = TemporalStreamLaneLoader(data, stream_lanes=lanes, chunk_size=3)
    TemporalCoverageAudit.verify_plan(dataset)
    seen, per_stream = [], {}
    for batch in DataLoader(dataset, batch_size=None, num_workers=workers):
        assert not batch.is_scored[~batch.valid_mask].any()
        seen.extend(batch.source_index[batch.is_scored].tolist())
        for sid, event in zip(batch.stream_id[batch.valid_mask].tolist(), batch.event_id[batch.valid_mask].tolist(), strict=True):
            per_stream.setdefault(sid, []).append(event)
    assert sorted(seen) == list(range(sum(lengths)))
    for events in per_stream.values():
        assert events == sorted(events)


@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_sections_replay_and_successor_ownership(chunk):
    data = data_for([1, 19, 6])
    dataset = TemporalStreamLaneLoader(data, stream_lanes=4, chunk_size=chunk,
                                      stream_execution="segmented", segment_events=5, segment_warmup_events=3)
    TemporalCoverageAudit.verify_plan(dataset)
    batches = list(dataset)
    again = TemporalStreamLaneLoader(data, stream_lanes=4, chunk_size=chunk,
                                    stream_execution="segmented", segment_events=5, segment_warmup_events=3)
    assert dataset.sections == again.sections
    scored = torch.cat([b.source_index[b.is_scored] for b in batches])
    assert sorted(scored.tolist()) == list(range(26))
    warmup = sum(int(b.warmup_mask.sum()) for b in batches)
    assert warmup == sum(s.core_start - s.warmup_start for s in dataset.sections)
    assert any(len(set(b.stream_id[b.valid_mask].tolist())) < int(b.valid_mask.any(1).sum()) for b in batches)
    successor_count = 0
    boundary_target = False
    for b in batches:
        assert not (b.is_scored & b.warmup_mask).any()
        # A lane chunk is entirely replay or core, even with unaligned sizes.
        assert not (b.warmup_mask.any(dim=1) & b.is_scored.any(dim=1)).any()
        targets = lane_targets(b, in_channels=17)
        valid = targets["valid_next"].reshape(b.valid_mask.shape)
        successor_count += int((valid & b.is_scored).sum())
        for idx in b.source_index[b.valid_mask]:
            lane, pos = (b.source_index == idx).nonzero()[0]
            if b.successor_index[lane, pos] >= 0:
                nxt = int(b.successor_index[lane, pos])
                assert data.stream_id[idx] == data.stream_id[nxt]
                assert targets["next_id"].reshape_as(valid)[lane, pos] == data.dst[nxt]
                torch.testing.assert_close(b.successor_msg[lane, pos], data.msg[nxt])
                if b.lane_section_end[lane] and pos == b.valid_mask[lane].sum() - 1:
                    boundary_target = True
    assert successor_count == 26 - 3
    assert boundary_target


@pytest.mark.parametrize("kwargs", [
    {"limit_train_batches": 10}, {"limit_train_batches": 1}, {"limit_train_batches": .5},
    {"max_steps": 1}, {"devices": 2}, {"num_nodes": 2}, {"fast_dev_run": True}, {"overfit_batches": 1},
])
def test_production_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        validate_full_lane_settings(kwargs)
    validate_full_lane_settings({"limit_train_batches": 1.0, "max_steps": -1})


def test_drop_partial_requires_diagnostic():
    with pytest.raises(ValueError, match="full coverage"):
        TemporalDataModule(None, batch_mode="stream_lanes", stream_lanes=2, chunk_size=2,
                           drop_partial_lane_batches=True)
    TemporalDataModule(None, batch_mode="stream_lanes", stream_lanes=2, chunk_size=2,
                       drop_partial_lane_batches=True, coverage_policy="diagnostic")


def test_validation_can_use_narrower_lanes_than_training():
    data = data_for([7, 11, 13])
    dm = TemporalDataModule(None, batch_mode="stream_lanes", stream_lanes=256, chunk_size=32,
                            eval_batch_mode="stream_lanes", eval_stream_lanes=3, eval_chunk_size=5)
    loader = dm._lane_loader(data, training=False)
    assert loader.dataset.stream_lanes == 3
    assert loader.dataset.chunk_size == 5
    assert loader.dataset.stream_lanes * loader.dataset.chunk_size == 15


def test_bitmap_detects_missing_duplicate_unexpected():
    bitmap = CoverageBitmap(11)
    bitmap.add([0, 1, 1, 8, -1, 11])
    bitmap.add([8, 10])
    assert bitmap.duplicate == 2
    assert bitmap.unexpected == 2
    assert bitmap.observed == 8
    assert np.flatnonzero(bitmap.mask()).tolist() == [0, 1, 8, 10]
    assert bitmap.bits.nbytes == 2


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unexpected", "warmup", "target"])
def test_audit_fails_and_persists_corrupt_epoch(tmp_path, corruption):
    dataset = TemporalStreamLaneLoader(data_for([7]), stream_lanes=2, chunk_size=3,
                                      stream_execution="segmented", segment_events=4, segment_warmup_events=2)
    audit = TemporalCoverageAudit(tmp_path)
    trainer = SimpleNamespace(current_epoch=0)
    audit.dataset, audit.report_path = dataset, tmp_path
    audit.on_train_epoch_start(trainer, None)
    batches = list(dataset)
    if corruption == "missing":
        batches[0].is_scored.zero_()
    elif corruption == "duplicate":
        batches.append(batches[0])
    elif corruption == "unexpected":
        b = next(b for b in batches if b.is_scored.any())
        b.source_index[b.is_scored] = 99
    elif corruption == "warmup":
        b = next(b for b in batches if b.warmup_mask.any())
        b.is_scored[b.warmup_mask] = True
    else:
        for b in batches:
            b.successor_index.fill_(-1)
    for idx, b in enumerate(batches):
        audit.on_train_batch_end(trainer, None, None, b, idx)
    with pytest.raises(RuntimeError, match="incomplete production"):
        audit.on_train_epoch_end(trainer, None)
    assert (tmp_path / "epoch-0.json").exists()
    assert not audit.last_report["complete"]


def test_audit_interruption_preserves_exception(tmp_path, monkeypatch):
    audit = TemporalCoverageAudit(tmp_path)
    monkeypatch.setattr(audit, "report", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.warns(UserWarning, match="preserving original"):
        audit.on_exception(None, None, KeyboardInterrupt())


def test_bounded_lane_validation_keeps_all_prefixes():
    clear_cache()
    data = data_for([3, 19, 9])
    data.is_scored[:2] = False
    class Source:
        cache_key = "bounded-lane-test"
        def build(self):
            return SimpleNamespace(train=data, val=data, test={})
    dm = TemporalDataModule(Source(), batch_mode="stream_lanes", stream_lanes=4, chunk_size=3,
                            eval_batch_mode="stream_lanes", val_prefix_events=7)
    dm.setup()
    batches = list(dm.val_dataloader())
    seen = torch.cat([b.event_id[b.valid_mask] for b in batches]).tolist()
    assert sorted(seen) == list(range(3)) + list(range(3, 10)) + list(range(22, 29))
    assert sum(int(b.is_scored.sum()) for b in batches) == 15
    assert sum(int((b.successor_index >= 0).sum()) for b in batches) == 14


def test_diagnostic_gap_resets_affected_lanes():
    dataset = TemporalStreamLaneLoader(data_for([2, 12, 12, 12]), stream_lanes=2,
                                      chunk_size=3, drop_partial_batches=True)
    assert dataset.diagnostic_gaps > 0
    assert dataset[0].lane_reset.tolist() == [True, True]


def test_audit_checks_materialized_plan_history():
    dataset = TemporalStreamLaneLoader(data_for([8, 4]), stream_lanes=2, chunk_size=2)
    dataset._plans = dataset._plans[1:]
    with pytest.raises(ValueError, match="history"):
        TemporalCoverageAudit.verify_plan(dataset)


def test_full_lane_fit_auto_audit_and_validation(tmp_path):
    from graphids.core.models.temporal.hybrid import TemporalHybridModel

    clear_cache()
    class Source:
        cache_key = "full-audit-fit"
        def build(self):
            return SimpleNamespace(train=data_for([7, 1, 5]), val=data_for([3, 5]), test={})
    dm = TemporalDataModule(Source(), batch_mode="stream_lanes", stream_lanes=2, chunk_size=3,
                            eval_batch_mode="stream_lanes", val_prefix_events=4)
    dm.setup()
    model = TemporalHybridModel(scale="small", objective="joint", num_ids=4, in_channels=17,
                                input={"hidden": 8, "embedding_dim": 4, "dropout": 0.0})
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=tmp_path)
    trainer.fit(model, datamodule=dm)
    audit = next(cb for cb in trainer.callbacks if isinstance(cb, TemporalCoverageAudit))
    assert audit.last_report["complete"]
    assert audit.last_report["unique"] == 13
    assert audit.last_report["observed_successor_targets"] == 10
    assert trainer.callback_metrics["val_class_0_events"] == 4
    assert trainer.callback_metrics["val_class_1_events"] == 3
