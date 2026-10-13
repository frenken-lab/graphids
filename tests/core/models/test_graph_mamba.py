import pytest
import torch

from graphids.core.data.snapshots.batch import collate_snapshots, recording_batch
from graphids.core.data.snapshots.context import build_context
from graphids.core.data.snapshots.preprocessing import (
    CANSnapshotRepresentationCfg,
    build_snapshots,
)
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from tests.core.data.test_snapshots import write_record


def make_batch(tmp_path):
    rec = write_record(
        tmp_path,
        role="train_reference",
        normal=True,
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "B", "00", 0),
            (0.02, "A", "01", 0),
            (0.03, "B", "01", 0),
            (0.04, "A", "02", 0),
        ],
    )
    reference = build_snapshots(rec, CANSnapshotRepresentationCfg())
    context = build_context("v", [reference])
    s = build_snapshots(
        rec, CANSnapshotRepresentationCfg(), anchor_ids=context.anchor_ids
    )
    return recording_batch(s, context), s, context


def model(**kwargs):
    return VehicleGraphMambaJoint(
        width=16, layers=2, backend="reference", dropout=0, **kwargs
    )


def test_full_sequence_streaming_and_chunk_carry(tmp_path):
    torch.manual_seed(42)
    batch, s, c = make_batch(tmp_path)
    net = model().eval()
    with torch.no_grad():
        parallel = net(batch)
        streaming, state = net.stream(batch)
        first, state = net.stream(recording_batch(s, c, stop=2))
        second, state = net.stream(recording_batch(s, c, start=2), state)
    for name in parallel:
        torch.testing.assert_close(
            parallel[name], streaming[name], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            streaming[name],
            torch.cat((first[name], second[name]), 1),
            rtol=1e-5,
            atol=1e-6,
        )
    assert not streaming["anomaly_eligible"][0, 0]
    assert streaming["anomaly_eligible"][0, 1:-1].all()
    assert not streaming["anomaly_eligible"][0, -1]  # zero-exposure terminal bin
    assert len(state.caches) == 2


def test_gradients_objective_masks_and_future_causality(tmp_path):
    batch, s, c = make_batch(tmp_path)
    net = model()
    original = net(batch)
    loss, terms = net.losses(original, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )
    changed = recording_batch(s, c)
    changed.x[:, 3:] = 100
    modified = net(changed)
    for key in original:
        torch.testing.assert_close(original[key][:, :3], modified[key][:, :3])
    batch.normal.zero_()
    _, terms = net.losses(net(batch), batch)
    assert all(terms[name] == 0 for name in ("count", "change", "new_id"))


def test_padding_isolation_and_graph_permutation(tmp_path):
    batch, s, c = make_batch(tmp_path)
    net = model().eval()
    shorter = recording_batch(s, c, stop=2)
    combined = collate_snapshots([shorter, batch])
    with torch.no_grad():
        expected = net(shorter)
        actual = net(combined)
    for key in expected:
        torch.testing.assert_close(
            expected[key], actual[key][:1, :2], rtol=1e-5, atol=1e-6
        )
    perm = torch.tensor([1, 0])
    permuted = recording_batch(s, c)
    for name in (
        "x",
        "live",
        "generation",
        "counts",
        "payload_change",
        "payload_observed",
    ):
        setattr(permuted, name, getattr(permuted, name)[:, :, perm])
    for name in ("profiles", "in_reference", "edge_mask", "edge_features"):
        setattr(permuted, name, getattr(permuted, name)[:, perm])
    permuted.neighbors = perm[permuted.neighbors[:, perm]]
    with torch.no_grad():
        out, other = net(batch), net(permuted)
    torch.testing.assert_close(
        out["classification_logits"], other["classification_logits"]
    )
    torch.testing.assert_close(out["count_rate"][:, :, perm], other["count_rate"])


def test_transient_capacity_fails_and_legacy_checkpoint_rejected(tmp_path):
    rec = write_record(tmp_path)
    s = build_snapshots(rec, CANSnapshotRepresentationCfg())
    with pytest.raises(ValueError, match="transient overflow"):
        recording_batch(s, None, transient_capacity=1)
    with pytest.raises(ValueError, match="legacy/incompatible"):
        model().on_load_checkpoint({"state_dict": {}})


def test_transient_eviction_and_recording_reset(tmp_path):
    rec = write_record(
        tmp_path,
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "B", "00", 0),
            (0.05, "A", "01", 0),
            (0.06, "B", "01", 0),
        ],
    )
    s = build_snapshots(rec, CANSnapshotRepresentationCfg(), retention_seconds=0.015)
    batch = recording_batch(s, None)
    net = model().eval()
    with torch.no_grad():
        seq = net(batch)
        step, state = net.stream(batch)
    torch.testing.assert_close(
        seq["classification_logits"],
        step["classification_logits"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert all(generation == 1 for key, generation in state.caches)
    other = recording_batch(s, None)
    other.recording_keys = ("new_capture",)
    with torch.no_grad():
        reset, _ = net.stream(other, state)
    torch.testing.assert_close(
        reset["classification_logits"], step["classification_logits"]
    )
