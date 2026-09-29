from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import TemporalData


def _temporal_batch(reset_after: list[bool] | None = None) -> TemporalData:
    n = 5
    return TemporalData(
        src=torch.tensor([0, 1, 1, 2, 2], dtype=torch.long),
        dst=torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
        t=torch.arange(n, dtype=torch.float32),
        msg=torch.tensor(
            [
                [0.0, 0.1, 0.0, 1.0],
                [1.0, 0.1, 1.0, 1.0],
                [0.0, 0.2, 0.0, 1.0],
                [1.0, 0.2, 1.0, 1.0],
                [0.0, 0.3, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        attack_type=torch.tensor([0, 2, 0, 2, 0], dtype=torch.long),
        stream_id=torch.zeros(n, dtype=torch.long),
        reset_after=torch.tensor(reset_after if reset_after is not None else [False, False, False, False, True]),
        event_id=torch.arange(n, dtype=torch.long),
        is_scored=torch.tensor([True, False, True, True, True]),
    )


def _lane_semantic_data() -> TemporalData:
    msg = torch.zeros(6, 17, dtype=torch.float32)
    msg[:, :4] = torch.tensor(
        [
            [0.0, 0.1, 0.0, 1.0],
            [1.0, 0.1, 1.0, 1.0],
            [0.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 1.0, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [1.0, 0.3, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    msg[:, 8:12] = msg[:, :4].roll(1, dims=0)
    msg[:, 16] = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    return TemporalData(
        src=torch.tensor([0, 1, 1, 2, 3, 3], dtype=torch.long),
        dst=torch.tensor([1, 1, 2, 3, 3, 0], dtype=torch.long),
        t=torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        msg=msg,
        y=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
        attack_type=torch.tensor([0, 2, 0, 2, 0, 2], dtype=torch.long),
        stream_id=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        reset_after=torch.tensor([False, False, True, False, False, True]),
        event_id=torch.arange(6, dtype=torch.long),
        is_scored=torch.ones(6, dtype=torch.bool),
    )


def _slice_temporal(data: TemporalData, mask: torch.Tensor) -> TemporalData:
    return TemporalData(
        src=data.src[mask],
        dst=data.dst[mask],
        t=data.t[mask],
        msg=data.msg[mask],
        y=data.y[mask],
        attack_type=data.attack_type[mask],
        stream_id=data.stream_id[mask],
        reset_after=data.reset_after[mask],
        event_id=data.event_id[mask],
        is_scored=data.is_scored[mask],
    )


_RAW_FORWARD_KEYS = {
    "features",
    "logits",
    "next_id_logits",
    "iat_loc",
    "iat_log_scale",
    "iat_pred",
    "payload_delta_loc",
    "payload_delta_log_scale",
    "payload_delta_pred",
}


def _hybrid(**overrides):
    from graphids.core.models.temporal import TemporalHybridModel

    params = {
        "hidden": None,
        "layers": None,
    }
    params.update(overrides)
    return TemporalHybridModel(
        scale="small",
        objective=params.get("objective", "supervised"),
        input={"hidden": 8, "embedding_dim": 4, "dropout": 0.0},
        memory=params.get("memory", {"type": "tgn", "enabled": True, "reset_on_stream_end": True}),
        backbone=params.get("backbone", {"type": params.get("backbone_type", "gru"), "layers": 1, "dropout": 0.0}),
        heads=params.get("heads"),
        anomaly=params.get("anomaly"),
        rhythm=params.get("rhythm"),
        motif=params.get("motif"),
        loss_fn=params.get("loss_fn"),
        loss_weights={"classification": 1.0, "next_id": 0.2, "iat": 0.1, "payload_delta": 0.1},
        num_ids=4,
        in_channels=params.get("in_channels", 4),
        num_classes=2,
    )


def _reference_forward_with_state(model, batch, state=None):
    encoded = model.encoder(batch)
    if encoded.numel() == 0:
        empty = encoded.new_empty((0, int(model.hparams.input["hidden"])))
        return model._heads_from_features(empty, batch), model._initial_state_like(encoded)

    current_state = model._initial_state_like(encoded) if state is None else dict(state)
    if model.memory is not None:
        current_state["memory"] = model.memory.ensure_state(current_state.get("memory"), encoded)
        current_state["last_seen"] = model.memory.ensure_last_seen(current_state.get("last_seen"), encoded)
    if model.rhythm is not None:
        current_state["rhythm"] = model.rhythm.ensure_state(current_state.get("rhythm"), encoded)
    if model.motif is not None:
        current_state["motif_ids"] = model.motif.ensure_ids(current_state.get("motif_ids"), encoded)
        current_state["motif_iats"] = model.motif.ensure_iats(current_state.get("motif_iats"), encoded)

    reset_after = getattr(batch, "reset_after", None)
    if reset_after is None:
        reset_after = torch.zeros(encoded.size(0), dtype=torch.bool, device=encoded.device)
    else:
        reset_after = reset_after.to(device=encoded.device, dtype=torch.bool)

    outputs = []
    memory_state = current_state.get("memory")
    last_seen = current_state.get("last_seen")
    rhythm_state = current_state.get("rhythm")
    motif_ids = current_state.get("motif_ids")
    motif_iats = current_state.get("motif_iats")
    backbone_state = current_state.get("backbone")
    reset_memory = bool(model.hparams.memory.get("reset_on_stream_end", True))
    t = getattr(batch, "t", None)

    for idx in range(encoded.size(0)):
        x_t = encoded[idx]
        iat_t = model._event_iat(batch, idx, x_t)
        if model.memory is not None and memory_state is not None:
            x_t = x_t + model.memory.read_context(
                memory_state,
                last_seen,
                batch.src[idx],
                batch.dst[idx],
                None if t is None else t[idx],
            )
        if model.rhythm is not None and rhythm_state is not None:
            x_t = x_t + model.rhythm.read_context(rhythm_state, batch.src[idx], batch.dst[idx])
        if model.motif is not None and motif_ids is not None and motif_iats is not None:
            x_t = x_t + model.motif.read_context(motif_ids, motif_iats)
        z_t, backbone_state = model.backbone.step(x_t, backbone_state)
        outputs.append(z_t)
        if model.memory is not None and memory_state is not None:
            memory_state = model.memory.update_one(memory_state, batch.src[idx], batch.dst[idx], z_t)
            if last_seen is not None:
                last_seen = model.memory.update_last_seen(last_seen, batch.src[idx], batch.dst[idx], None if t is None else t[idx])
        if model.rhythm is not None and rhythm_state is not None:
            rhythm_state = model.rhythm.update_one(rhythm_state, batch.src[idx], batch.dst[idx], iat_t)
        if model.motif is not None and motif_ids is not None and motif_iats is not None:
            motif_ids, motif_iats = model.motif.update_one(motif_ids, motif_iats, batch.dst[idx], iat_t)
        if reset_after[idx]:
            backbone_state = model.backbone.reset_state()
            if model.memory is not None and reset_memory:
                memory_state = model.memory.initial_state(device=encoded.device, dtype=encoded.dtype)
                last_seen = model.memory.initial_last_seen(device=encoded.device, dtype=encoded.dtype)
            if model.rhythm is not None:
                rhythm_state = model.rhythm.initial_state(device=encoded.device, dtype=encoded.dtype)
            if model.motif is not None:
                motif_ids = model.motif.initial_ids(device=encoded.device)
                motif_iats = model.motif.initial_iats(device=encoded.device, dtype=encoded.dtype)

    features = torch.stack(outputs, dim=0)
    next_state = {
        "memory": memory_state,
        "last_seen": last_seen,
        "rhythm": rhythm_state,
        "motif_ids": motif_ids,
        "motif_iats": motif_iats,
        "backbone": backbone_state,
    }
    return model._heads_from_features(features, batch), next_state


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        elif actual_value.is_floating_point():
            assert torch.allclose(actual_value, expected_value)
        else:
            assert torch.equal(actual_value, expected_value)


def _assert_outputs_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if actual_value.is_floating_point():
            assert torch.allclose(actual_value, expected_value)
        else:
            assert torch.equal(actual_value, expected_value)


def _old_memory_update(memory, state, src, dst, message):
    src_idx = int(src.clamp_min(0).clamp_max(memory.num_ids - 1).item())
    dst_idx = int(dst.clamp_min(0).clamp_max(memory.num_ids - 1).item())
    src_new = memory.update_cell(message.unsqueeze(0), state[src_idx].unsqueeze(0)).squeeze(0)
    src_mask = F.one_hot(torch.tensor(src_idx, device=state.device), num_classes=memory.num_ids).to(
        dtype=state.dtype
    ).unsqueeze(-1)
    after_src = (state * (1.0 - src_mask)) + (src_new.unsqueeze(0) * src_mask)

    dst_new = memory.update_cell(message.unsqueeze(0), after_src[dst_idx].unsqueeze(0)).squeeze(0)
    dst_mask = F.one_hot(torch.tensor(dst_idx, device=state.device), num_classes=memory.num_ids).to(
        dtype=state.dtype
    ).unsqueeze(-1)
    return (after_src * (1.0 - dst_mask)) + (dst_new.unsqueeze(0) * dst_mask)


def _old_last_seen_update(memory, last_seen, src, dst, t):
    src_idx = int(src.clamp_min(0).clamp_max(memory.num_ids - 1).item())
    dst_idx = int(dst.clamp_min(0).clamp_max(memory.num_ids - 1).item())
    now = t.to(device=last_seen.device, dtype=last_seen.dtype)
    src_mask = F.one_hot(torch.tensor(src_idx, device=last_seen.device), num_classes=memory.num_ids).bool()
    dst_mask = F.one_hot(torch.tensor(dst_idx, device=last_seen.device), num_classes=memory.num_ids).bool()
    updated = torch.where(src_mask, now, last_seen)
    return torch.where(dst_mask, now, updated)


def _old_rhythm_update(rhythm, state, idx, value):
    pos = int(idx.clamp_min(0).clamp_max(rhythm.num_ids - 1).item())
    old = state[pos]
    val = value.to(device=state.device, dtype=state.dtype).clamp_min(0)
    count = old[0] + 1.0
    delta = val - old[1]
    mean = old[1] + (delta / count)
    m2 = old[2] + delta * (val - mean)
    replacement = torch.stack([count, mean, m2])
    mask = F.one_hot(torch.tensor(pos, device=state.device), num_classes=rhythm.num_ids).to(
        state.dtype
    ).unsqueeze(-1)
    return (state * (1.0 - mask)) + (replacement.unsqueeze(0) * mask)


def test_temporal_hybrid_memory_indexed_updates_match_old_one_hot_semantics():
    from graphids.core.models.temporal.hybrid import TemporalIdMemory

    torch.manual_seed(7)
    memory = TemporalIdMemory(num_ids=4, hidden=5)
    expected = torch.randn(4, 5)
    actual = expected.clone()
    src = torch.tensor([1, 3, 2, 9])
    dst = torch.tensor([1, 1, -4, 2])
    messages = torch.randn(4, 5)

    for idx in range(src.numel()):
        expected = _old_memory_update(memory, expected, src[idx], dst[idx], messages[idx])
        actual = memory.update_one(actual, src[idx], dst[idx], messages[idx])
        assert torch.allclose(actual, expected)


def test_temporal_hybrid_last_seen_indexed_updates_match_old_time_aware_semantics():
    from graphids.core.models.temporal.hybrid import TemporalIdMemory

    memory = TemporalIdMemory(num_ids=4, hidden=3, time_encoding_dim=2)
    expected = memory.initial_last_seen(device=torch.device("cpu"), dtype=torch.float32)
    actual = expected.clone()
    src = torch.tensor([0, 2, 2, 8])
    dst = torch.tensor([1, 2, -3, 3])
    times = torch.tensor([0.25, 1.5, 3.0, 7.5])

    for idx in range(src.numel()):
        expected = _old_last_seen_update(memory, expected, src[idx], dst[idx], times[idx])
        actual = memory.update_last_seen(actual, src[idx], dst[idx], times[idx])
        assert torch.equal(actual, expected)


def test_temporal_hybrid_rhythm_indexed_updates_match_old_one_hot_semantics():
    from graphids.core.models.temporal.hybrid import TemporalRhythmContext

    rhythm = TemporalRhythmContext(num_ids=4, hidden=6)
    expected = rhythm.initial_state(device=torch.device("cpu"), dtype=torch.float32)
    actual = expected.clone()
    src = torch.tensor([1, 1, 6, 2])
    dst = torch.tensor([1, 3, 2, -5])
    iats = torch.tensor([0.4, 1.2, 0.0, 2.5])

    for idx in range(src.numel()):
        expected = _old_rhythm_update(rhythm, expected, src[idx], iats[idx])
        expected = _old_rhythm_update(rhythm, expected, dst[idx], iats[idx])
        actual = rhythm.update_one(actual, src[idx], dst[idx], iats[idx])
        assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "reset_after",
    [
        [False, False, False, False, False],
        [False, False, False, False, True],
        [False, True, False, False, False],
    ],
)
def test_temporal_hybrid_segment_scan_matches_reference_loop(reset_after):
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
    )
    batch = _temporal_batch(reset_after)

    actual_out, actual_state = model._forward_with_state(batch, None)
    expected_out, expected_state = _reference_forward_with_state(model, batch, None)

    _assert_outputs_equal(actual_out, expected_out)
    _assert_state_equal(actual_state, expected_state)


def test_temporal_hybrid_rich_scan_uses_sparse_row_overlay(monkeypatch):
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
    )

    def forbidden_full_state_update(*_args, **_kwargs):
        raise AssertionError("scan should update row overlays, not clone full states per event")

    monkeypatch.setattr(model.memory, "_update_one_indexed", forbidden_full_state_update)
    monkeypatch.setattr(model.rhythm, "_update_one_indexed", forbidden_full_state_update)

    out, state = model._forward_with_state(_temporal_batch([False, False, False, False, False]), None)

    assert tuple(out["features"].shape) == (5, 8)
    assert state["memory"] is not None
    assert state["rhythm"] is not None


def test_temporal_hybrid_vectorized_iat_matches_event_iat_payload_timestamp_and_missing_timestamp():
    payload_model = _hybrid(objective="supervised", heads={"classification": True}, in_channels=17)
    payload_batch = TemporalData(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 3]),
        t=torch.tensor([0.0, 10.0, 11.0]),
        msg=torch.stack([torch.arange(17), torch.arange(17) + 2, torch.arange(17) + 4]).float(),
        y=torch.tensor([0, 1, 0]),
    )
    ref = payload_batch.msg
    expected = torch.stack([payload_model._event_iat(payload_batch, idx, ref[idx]) for idx in range(3)])
    assert torch.equal(payload_model._prepare_scan_batch(payload_batch, ref)["iat"], expected)

    timestamp_model = _hybrid(objective="supervised", heads={"classification": True})
    timestamp_batch = _temporal_batch([False, False, False, False, False])
    ref = timestamp_batch.msg
    expected = torch.stack([timestamp_model._event_iat(timestamp_batch, idx, ref[idx]) for idx in range(5)])
    assert torch.equal(timestamp_model._prepare_scan_batch(timestamp_batch, ref)["iat"], expected)

    missing_timestamp_batch = TemporalData(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 3]),
        msg=torch.ones(3, 4),
        y=torch.tensor([0, 1, 0]),
    )
    ref = missing_timestamp_batch.msg
    expected = torch.stack(
        [timestamp_model._event_iat(missing_timestamp_batch, idx, ref[idx]) for idx in range(3)]
    )
    assert torch.equal(timestamp_model._prepare_scan_batch(missing_timestamp_batch, ref)["iat"], expected)


def test_temporal_hybrid_mid_batch_reset_matches_fresh_suffix_state():
    model = _hybrid(objective="supervised", heads={"classification": True})
    batch = _temporal_batch([False, True, False, False, False])
    suffix = TemporalData(
        src=batch.src[2:],
        dst=batch.dst[2:],
        t=batch.t[2:],
        msg=batch.msg[2:],
        y=batch.y[2:],
        attack_type=batch.attack_type[2:],
        stream_id=batch.stream_id[2:],
        reset_after=batch.reset_after[2:],
        event_id=batch.event_id[2:],
        is_scored=batch.is_scored[2:],
    )

    full_out, full_state = model._forward_with_state(batch, None)
    suffix_out, suffix_state = model._forward_with_state(suffix, None)

    assert torch.allclose(full_out["features"][2:], suffix_out["features"])
    assert torch.allclose(full_state["memory"], suffix_state["memory"])
    assert torch.equal(full_state["last_seen"], suffix_state["last_seen"])
    assert torch.allclose(full_state["backbone"], suffix_state["backbone"])


def test_temporal_hybrid_builds_from_primitive_and_consumes_temporal_data():
    from graphids.exp.ray_backend import build_component

    model = build_component(
        {
            "type": "temporal_hybrid",
            "objective": "joint",
            "input": {"hidden": 8, "embedding_dim": 4, "dropout": 0.0},
            "backbone": {"type": "ssm_lite", "layers": 1, "dropout": 0.0},
        }
    )
    model.num_ids = 4
    model.in_channels = 4
    model.num_classes = 2
    model.hparams["num_ids"] = 4
    model.hparams["in_channels"] = 4
    model.hparams["num_classes"] = 2
    model._build()
    model._built = True

    out = model(_temporal_batch())

    assert tuple(out["logits"].shape) == (5, 2)
    assert tuple(out["anomaly_score"].shape) == (5,)


def test_temporal_hybrid_supervised_backpropagates_ce_loss():
    model = _hybrid(objective="supervised", heads={"classification": True})
    batch = _temporal_batch()

    out = model(batch)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert tuple(out["logits"].shape) == (5, 2)
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_hybrid_anomaly_scores_and_backpropagates_self_supervised_loss():
    model = _hybrid(
        objective="anomaly",
        backbone_type="ssm_lite",
        heads={"classification": False, "next_id": True, "iat": True, "payload_delta": True},
    )
    batch = _temporal_batch()

    out = model(batch)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert "logits" not in out
    assert tuple(out["anomaly_score"].shape) == (5,)
    assert torch.isfinite(out["anomaly_score"]).all()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_hybrid_joint_records_classifier_and_anomaly_metrics(monkeypatch):
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
    )
    model._test_set_names = ["holdout"]
    model._attack_type_names = {0: "benign", 2: "fuzzing"}
    logged = {}
    monkeypatch.setattr(model, "log_dict", lambda values, *args, **kwargs: logged.update(values))

    model.on_test_epoch_start()
    model.test_step(_temporal_batch(), 0, dataloader_idx=0)
    model.on_test_epoch_end()

    assert "test/holdout/accuracy" in logged
    assert "test/holdout/auroc" in logged
    assert "test/auroc" in logged


def test_temporal_hybrid_reset_after_clears_backbone_and_memory_state():
    model = _hybrid(
        objective="joint",
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
    )
    batch = _temporal_batch([False, False, False, False, True])

    _out, state = model._forward_with_state(batch, None)

    assert state["backbone"] is None
    assert state["memory"] is not None
    assert torch.count_nonzero(state["memory"]) == 0
    assert state["last_seen"] is not None
    assert torch.all(state["last_seen"] < 0)


def test_temporal_hybrid_rich_context_and_nll_heads_backpropagate():
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
    )
    batch = _temporal_batch()

    out, state = model._forward_with_state(batch, None)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert "iat_nll" in out
    assert "payload_delta_nll" in out
    assert tuple(out["anomaly_score"].shape) == (5,)
    assert state["rhythm"] is not None
    assert state["motif_ids"] is not None
    assert state["motif_iats"] is not None
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_hybrid_lane_batch_matches_independent_rich_streams():
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        in_channels=17,
    )
    data = _lane_semantic_data()

    expected_by_event = {}
    for stream_id in [0, 1]:
        stream = _slice_temporal(data, data.stream_id == stream_id)
        out, _state = model._forward_with_state(stream, None)
        for pos, event_id in enumerate(stream.event_id.tolist()):
            expected_by_event[event_id] = {
                key: value[pos].detach()
                for key, value in out.items()
                if key in _RAW_FORWARD_KEYS and value.ndim > 0
            }

    state = None
    actual_by_event = {}
    for batch in TemporalStreamLaneLoader(data, stream_lanes=2, chunk_size=2):
        out, state = model._forward_with_state(batch, model._detach_state(state))
        flat_ids = batch.event_id.reshape(-1)
        flat_valid = batch.valid_mask.reshape(-1)
        for pos in flat_valid.nonzero(as_tuple=False).flatten().tolist():
            actual_by_event[int(flat_ids[pos].item())] = {
                key: value[pos].detach()
                for key, value in out.items()
                if key in _RAW_FORWARD_KEYS and value.ndim > 0
            }

    assert actual_by_event.keys() == expected_by_event.keys()
    for event_id, expected in expected_by_event.items():
        actual = actual_by_event[event_id]
        for key, expected_value in expected.items():
            assert key in actual
            if expected_value.is_floating_point():
                assert torch.allclose(actual[key], expected_value, atol=1e-6)
            else:
                assert torch.equal(actual[key], expected_value)


def test_temporal_hybrid_lane_batch_loss_backward_is_finite():
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        in_channels=17,
    )
    batch = next(iter(TemporalStreamLaneLoader(_lane_semantic_data(), stream_lanes=2, chunk_size=2)))

    loss = model.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_hybrid_checkpoint_loads_without_loss_fn_for_anomaly(tmp_path):
    from graphids.core.models.base import safe_load_checkpoint
    from graphids.core.models.temporal import TemporalHybridModel

    model = _hybrid(
        objective="anomaly",
        heads={"classification": False, "next_id": True, "iat": True, "payload_delta": True},
    )
    ckpt = {
        "hyper_parameters": dict(model.hparams),
        "state_dict": model.state_dict(),
    }
    model.on_save_checkpoint(ckpt)
    path = tmp_path / "hybrid.ckpt"
    torch.save(ckpt, path)

    loaded = safe_load_checkpoint("temporal_hybrid", path)

    assert isinstance(loaded, TemporalHybridModel)
    assert loaded.hparams.objective == "anomaly"
    assert loaded(_temporal_batch())["anomaly_score"].shape == (5,)


def test_temporal_hybrid_rejects_invalid_objective_head_combos():
    from graphids.core.losses import CrossEntropyLoss

    with pytest.raises(ValueError, match="objective='supervised' requires heads.classification=true"):
        _hybrid(objective="supervised", heads={"classification": False})

    with pytest.raises(ValueError, match="objective='anomaly' requires heads.classification=false"):
        _hybrid(objective="anomaly", heads={"classification": True})

    with pytest.raises(ValueError, match="objective='anomaly' requires at least one anomaly head"):
        _hybrid(
            objective="anomaly",
            heads={"next_id": False, "iat": False, "payload_delta": False},
        )

    with pytest.raises(ValueError, match="objective='joint' requires at least one anomaly head"):
        _hybrid(
            objective="joint",
            heads={
                "classification": True,
                "next_id": False,
                "iat": False,
                "payload_delta": False,
            },
        )

    with pytest.raises(ValueError, match="does not accept loss_fn"):
        _hybrid(
            objective="anomaly",
            heads={"next_id": True, "iat": True, "payload_delta": True},
            loss_fn=CrossEntropyLoss(),
        )
