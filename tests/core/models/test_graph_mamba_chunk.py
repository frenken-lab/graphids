"""Chunking, lifecycle and score parity against the frozen pre-refactor code."""

from dataclasses import replace

import pytest
import torch

from graphids.core.data.snapshots.batch import collate_snapshots, recording_batch
from graphids.core.data.snapshots.preprocessing import (
    CANSnapshotRepresentationCfg,
    build_snapshots,
)
from graphids.core.models.graph_mamba import GraphMambaState, VehicleGraphMambaJoint
from graphids.core.models.score_columns import ScoreBlock, ScoreTable
from tests.core.data.test_snapshots import write_record
from tests.core.models.stream_oracle import oracle_stream
from tests.core.models.test_graph_mamba import make_batch, model


def lifecycle_batch(tmp_path):
    rec = write_record(
        tmp_path,
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "B", "01", 0),
            (0.04, "C", "", 0),
            (0.07, "A", "02", 1),
            (0.08, "B", "03", 0),
            (0.12, "C", "01", 0),
        ],
    )
    snapshot = build_snapshots(
        rec, CANSnapshotRepresentationCfg(), retention_seconds=0.015
    )
    return snapshot


@pytest.mark.parametrize("mode", ["chunked", "recurrent"])
def test_lifecycle_chunks_match_independent_oracle(tmp_path, mode):
    torch.manual_seed(31)
    snapshot = lifecycle_batch(tmp_path)
    net = model(stream_execution=mode).eval()
    batch = recording_batch(snapshot, None)
    expected, _ = oracle_stream(net, batch)
    state = None
    parts = []
    cutoffs = [0, 1, 3, 4, 8, len(snapshot.closes)]
    for start, stop in zip(cutoffs[:-1], cutoffs[1:], strict=True):
        chunk = recording_batch(snapshot, None, start=start, stop=stop)
        # Exercise remapping when a caller changes local node order.
        if chunk.x.shape[2] > 1:
            perm = torch.arange(chunk.x.shape[2] - 1, -1, -1)
            for name in (
                "x",
                "live",
                "generation",
                "counts",
                "payload_change",
                "payload_observed",
            ):
                setattr(chunk, name, getattr(chunk, name)[:, :, perm])
            for name in (
                "profiles",
                "in_reference",
                "neighbors",
                "edge_features",
                "edge_mask",
            ):
                setattr(chunk, name, getattr(chunk, name)[:, perm])
            chunk.node_keys = (tuple(chunk.node_keys[0][i] for i in perm),)
        output, state = net.stream(chunk, state)
        parts.append(output)
        assert len(state.caches) <= chunk.state_capacity
    for name in (
        "classification_logits",
        "new_id_rate",
        "anomaly_components",
        "anomaly_component_mask",
        "anomaly_eligible",
        "anomaly_score",
    ):
        torch.testing.assert_close(
            torch.cat([part[name] for part in parts], 1),
            expected[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_nonzero_state_all_outputs_and_empty_live_chunk(tmp_path):
    batch, snapshot, context = make_batch(tmp_path)
    net = model().eval()
    expected, _ = oracle_stream(net, batch)
    first, state = net.stream(recording_batch(snapshot, context, stop=1))
    second, state = net.stream(recording_batch(snapshot, context, start=1), state)
    for name in expected:
        torch.testing.assert_close(
            torch.cat((first[name], second[name]), 1),
            expected[name],
            rtol=1e-5,
            atol=1e-6,
        )
    empty = recording_batch(snapshot, context, stop=2)
    empty.live.zero_()
    empty.generation.fill_(-1)
    output, state = net.stream(empty)
    assert not state.caches
    assert torch.isfinite(output["classification_logits"]).all()
    assert not output["anomaly_component_mask"][..., :2].any()


def test_state_version_gaps_and_context_reset(tmp_path):
    batch, snapshot, context = make_batch(tmp_path)
    net = model().eval()
    with pytest.raises(ValueError, match="reset"):
        net.stream(batch, GraphMambaState(version=1))
    _, state = net.stream(recording_batch(snapshot, context, stop=1))
    with pytest.raises(ValueError, match="contiguous"):
        net.stream(recording_batch(snapshot, context, start=2), state)
    with pytest.raises(ValueError, match="contiguous"):
        net.stream(batch, state)
    changed = replace(batch, context_digests=("different",))
    actual, _ = net.stream(changed, state)
    expected, _ = net.stream(changed)
    torch.testing.assert_close(
        actual["anomaly_components"], expected["anomaly_components"]
    )


def test_score_columns_calibration_and_row_contract(tmp_path):
    batch, _, _ = make_batch(tmp_path)
    net = model().eval()
    net._recordings = {
        batch.recording_keys[0]: type(
            "Rec",
            (),
            dict(
                key=batch.recording_keys[0], dataset="d", vehicle="v", role="val_query"
            ),
        )()
    }
    output, _ = net.stream(batch)
    block = ScoreBlock.from_batch(net, batch, output)
    table = ScoreTable([block])
    table.check_unique("validation")
    # A complete validation calibration requires both inference modes.
    from dataclasses import replace

    net.data_manifest_digest = "fixture"
    other = replace(
        block,
        identity=dict(
            block.identity,
            context="zero-shot"
            if block.identity["context"] != "zero-shot"
            else "conditioned",
        ),
    )
    ScoreTable([block, other]).calibrate(net)
    table.apply_bundles(net.calibration_bundles)
    assert net.calibrated
    rows = list(table)
    assert len(rows) == batch.scored.sum()
    assert isinstance(rows[0]["bin"], int)
    assert isinstance(rows[0]["alert"], bool)
    assert len(rows[0]["anomaly_components"]) == 3
    assert len(rows[0]["anomaly_component_mask"]) == 3
    with pytest.raises(ValueError, match="duplicate"):
        ScoreTable([block, block]).check_unique("validation")


@pytest.mark.slurm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("mode", ["chunked", "recurrent"])
def test_official_nonzero_carry_and_lifecycle(tmp_path, mode):
    torch.manual_seed(42)
    snapshot = lifecycle_batch(tmp_path)
    net = (
        VehicleGraphMambaJoint(
            width=32, dropout=0, backend="official", stream_execution=mode
        )
        .cuda()
        .eval()
    )
    batch = recording_batch(snapshot, None).to("cuda")
    expected, _ = oracle_stream(net, batch)
    state = None
    parts = []
    for start, stop in ((0, 1), (1, 3), (3, 8), (8, len(snapshot.closes))):
        output, state = net.stream(
            recording_batch(snapshot, None, start=start, stop=stop).to("cuda"), state
        )
        parts.append(output)
    for name in (
        "classification_logits",
        "new_id_rate",
        "anomaly_components",
        "anomaly_component_mask",
        "anomaly_eligible",
        "anomaly_score",
    ):
        torch.testing.assert_close(
            torch.cat([part[name] for part in parts], 1),
            expected[name],
            rtol=2e-4,
            atol=2e-5,
        )


def test_vectorized_training_gradients_match_original(tmp_path):
    from tests.core.models.stream_oracle import oracle_encode

    torch.manual_seed(42)
    batch, _, _ = make_batch(tmp_path)
    batch = collate_snapshots([batch, replace(batch, x=batch.x * 1.1)])
    net = model()
    original = net._heads(oracle_encode(net, batch), batch)
    loss, _ = net.losses(original, batch)
    loss.backward()
    gradients = {name: value.grad.clone() for name, value in net.named_parameters()}
    net.zero_grad(set_to_none=True)
    actual = net(batch)
    loss, _ = net.losses(actual, batch)
    loss.backward()
    for name in actual:
        torch.testing.assert_close(actual[name], original[name])
    for name, value in net.named_parameters():
        torch.testing.assert_close(value.grad, gradients[name], rtol=1e-5, atol=1e-6)


@pytest.mark.slurm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_official_tiled_correction_long_carry_and_final_caches():
    from scripts.graph_mamba_memory_gate import capacity_batch

    torch.manual_seed(17)
    batch = capacity_batch(256, 51, 8)
    net = VehicleGraphMambaJoint(width=128, dropout=0, backend="official").cuda().eval()
    expected, oracle_state = oracle_stream(net, batch)
    first = replace(
        batch,
        **{
            name: getattr(batch, name)[:, :3]
            for name in (
                "x",
                "live",
                "generation",
                "labels",
                "scored",
                "normal",
                "counts",
                "payload_change",
                "payload_observed",
                "new_ids",
                "exposure",
                "closes",
                "event_count",
                "bin_indices",
            )
        },
        execution=None,
    )
    second = replace(
        batch,
        **{
            name: getattr(batch, name)[:, 3:]
            for name in (
                "x",
                "live",
                "generation",
                "labels",
                "scored",
                "normal",
                "counts",
                "payload_change",
                "payload_observed",
                "new_ids",
                "exposure",
                "closes",
                "event_count",
                "bin_indices",
            )
        },
        execution=None,
    )
    a, state = net.stream(first)
    b, state = net.stream(second, state)
    for name in expected:
        torch.testing.assert_close(
            torch.cat((a[name], b[name]), 1), expected[name], rtol=2e-4, atol=2e-5
        )
    for identity, slot in state.caches.items():
        for layer, (conv, ssm) in enumerate(state.layers):
            previous_conv, previous_ssm = oracle_state.caches[identity][layer]
            torch.testing.assert_close(
                conv[slot : slot + 1], previous_conv, rtol=2e-4, atol=2e-5
            )
            torch.testing.assert_close(
                ssm[slot : slot + 1], previous_ssm, rtol=2e-4, atol=2e-5
            )
