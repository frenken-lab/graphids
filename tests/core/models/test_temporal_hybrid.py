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


def _uneven_lane_semantic_data() -> TemporalData:
    data = _lane_semantic_data()
    data.stream_id = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    data.reset_after = torch.tensor([False, False, False, False, False, True])
    return data


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
        profiling=params.get("profiling"),
        scan=params.get("scan"),
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
            assert torch.allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(actual_value, expected_value)


def _assert_outputs_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if actual_value.is_floating_point():
            assert torch.allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(actual_value, expected_value)


def _lane_batch_reference_forward_with_state(model, batch):
    lanes, chunk = batch.valid_mask.shape
    hidden = int(model.hparams.input["hidden"])
    expected_features = batch.msg.new_zeros((lanes, chunk, hidden))
    initial = model._initial_lane_state_like(expected_features)
    lane_states: list[dict[str, torch.Tensor | None]] = []

    for lane_idx in range(lanes):
        valid = batch.valid_mask[lane_idx]
        if not valid.any():
            lane_states.append({key: None if value is None else value[lane_idx].clone() for key, value in initial.items()})
            continue
        temporal = TemporalData(
            src=batch.src[lane_idx, valid],
            dst=batch.dst[lane_idx, valid],
            t=batch.t[lane_idx, valid],
            msg=batch.msg[lane_idx, valid],
            y=batch.y[lane_idx, valid],
            attack_type=batch.attack_type[lane_idx, valid],
            stream_id=batch.stream_id[lane_idx, valid],
            reset_after=batch.reset_after[lane_idx, valid],
            event_id=batch.event_id[lane_idx, valid],
            is_scored=batch.is_scored[lane_idx, valid],
        )
        lane_out, lane_state = _reference_forward_with_state(model, temporal, None)
        expected_features[lane_idx, valid] = lane_out["features"]

        state_with_prev = dict(lane_state)
        state_with_prev["prev_t"] = batch.t.new_tensor(-1.0)
        if not batch.lane_stream_end[lane_idx]:
            valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
            state_with_prev["prev_t"] = batch.t[lane_idx, valid_positions[-1]].clone()
        elif model.backbone.backbone_type != "none":
            state_with_prev["backbone"] = expected_features.new_zeros((model.backbone.layers, hidden))
        lane_states.append(state_with_prev)

    expected_state: dict[str, torch.Tensor | None] = {}
    for key, initial_value in initial.items():
        values = [state.get(key) for state in lane_states]
        if key == "backbone" and values[0] is not None:
            expected_state[key] = torch.stack([value for value in values if value is not None], dim=1)
        elif initial_value is None or values[0] is None:
            expected_state[key] = None
        else:
            expected_state[key] = torch.stack([value for value in values if value is not None], dim=0)
    expected_out = model._heads_from_features(expected_features.reshape(lanes * chunk, hidden), batch)
    return expected_out, expected_state


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


@pytest.mark.parametrize("rhythm_kernel,device", [("torch", "cpu"), ("triton", "cpu"), ("triton", "cuda")])
def test_temporal_hybrid_lane_scan_matches_plain_reference_loop(rhythm_kernel, device):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        pytest.importorskip("triton")
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        in_channels=17,
        scan={"profile_subphases": True, "rhythm_kernel": rhythm_kernel},
    )
    model = model.to(device)
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    batch = next(iter(TemporalStreamLaneLoader(_lane_semantic_data(), stream_lanes=2, chunk_size=2)))
    batch = batch.to(device)
    actual_out, actual_state = model._forward_with_state(batch, None)

    expected_out, expected_state = _lane_batch_reference_forward_with_state(model, batch)

    if device == "cuda":
        assert not getattr(model, "_rhythm_kernel_failed", False)
        actual_out["features"].sum().backward()
        for parameter in model.rhythm.proj.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    _assert_outputs_equal(actual_out, expected_out)
    _assert_state_equal(actual_state, expected_state)


def test_temporal_hybrid_rejects_retired_scan_backend():
    with pytest.raises(ValueError, match="unknown scan config key"):
        _hybrid(scan={"backend": "lane_fastpath"})


def test_temporal_hybrid_lane_scan_writer_handles_partial_motif_roll(monkeypatch):
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        memory={"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 4, "embedding_dim": 4, "time_dim": 4},
        in_channels=17,
        scan={"profile_subphases": True},
    )
    original_writer = model._write_lane_step_state
    writer_lane_counts = []

    def spy_writer(**kwargs):
        writer_lane_counts.append(int(kwargs["gathered"].lane_idx.numel()))
        return original_writer(**kwargs)

    monkeypatch.setattr(model, "_write_lane_step_state", spy_writer)

    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    batch = next(iter(TemporalStreamLaneLoader(_uneven_lane_semantic_data(), stream_lanes=2, chunk_size=3)))
    actual_out, actual_state = model._forward_with_state(batch, None)

    expected_out, expected_state = _lane_batch_reference_forward_with_state(model, batch)

    assert writer_lane_counts == [2, 2, 1]
    _assert_outputs_equal(actual_out, expected_out)
    _assert_state_equal(actual_state, expected_state)
    assert torch.equal(actual_state["motif_ids"][0], torch.tensor([0, 1, 1, 2]))


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "memory": {"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
            "rhythm": {"enabled": True},
            "motif": {"enabled": False},
        },
        {
            "memory": {"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
            "rhythm": {"enabled": False},
            "motif": {"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        },
        {
            "memory": {"enabled": False},
            "rhythm": {"enabled": True},
            "motif": {"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        },
        {
            "memory": {"type": "tgn", "enabled": True, "reset_on_stream_end": True, "time_encoding_dim": 4},
            "rhythm": {"enabled": False},
            "motif": {"enabled": False},
        },
        {
            "memory": {"enabled": False},
            "rhythm": {"enabled": False},
            "motif": {"enabled": False},
        },
        {
            "memory": {"enabled": False},
            "rhythm": {"enabled": False},
            "motif": {"enabled": False},
            "backbone": {"type": "none", "layers": 1, "dropout": 0.0},
        },
    ],
)
def test_temporal_hybrid_lane_scan_module_variants_match_eager(overrides):
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    base = {
        "objective": "joint",
        "backbone_type": "ssm_lite",
        "heads": {"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        "anomaly": {"mode": "nll"},
        "in_channels": 17,
    }
    base.update(overrides)
    model = _hybrid(**base, scan={"profile_subphases": True})
    batch = next(iter(TemporalStreamLaneLoader(_lane_semantic_data(), stream_lanes=2, chunk_size=2)))
    actual_out, actual_state = model._forward_with_state(batch, None)

    expected_out, expected_state = _lane_batch_reference_forward_with_state(model, batch)

    _assert_outputs_equal(actual_out, expected_out)
    _assert_state_equal(actual_state, expected_state)


def test_temporal_id_memory_batched_gru_cell_path_matches_row_loop():
    from graphids.core.models.temporal._hybrid.contexts import TemporalIdMemory

    memory = TemporalIdMemory(num_ids=4, hidden=8)
    message = torch.randn(5, 8)
    src_rows = torch.randn(5, 8)
    dst_rows = torch.randn(5, 8)
    same_id = torch.tensor([False, True, False, True, False])

    src_expected = memory.update_cell(message, src_rows)
    dst_input = torch.where(same_id.view(-1, 1), src_expected, dst_rows)
    dst_expected = memory.update_cell(message, dst_input)

    src_actual = memory.update_cell(message, src_rows)
    dst_input_actual = torch.where(same_id.view(-1, 1), src_actual, dst_rows)
    dst_actual = memory.update_cell(message, dst_input_actual)

    assert torch.allclose(src_actual, src_expected, atol=1e-6)
    assert torch.allclose(dst_actual, dst_expected, atol=1e-6)


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


def test_temporal_hybrid_phase_timing_logs_training_step_breakdown(monkeypatch):
    model = _hybrid(
        objective="joint",
        backbone_type="ssm_lite",
        heads={"classification": True, "next_id": True, "iat": True, "payload_delta": True},
        profiling={"phase_timing": True, "warmup_steps": 0, "log_every_n_steps": 1},
    )
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, *args, **kwargs: logged.setdefault(name, value))

    loss = model.training_step(_temporal_batch(), 0)

    assert torch.isfinite(loss)
    for name in [
        "profile/train_phase_encode_ms",
        "profile/train_phase_scan_ms",
        "profile/train_phase_heads_ms",
        "profile/train_phase_forward_ms",
        "profile/train_phase_loss_ms",
        "profile/train_phase_heads_loss_ms",
    ]:
        assert name in logged
        assert float(logged[name]) >= 0.0


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


def test_temporal_hybrid_lane_batch_matches_independent_context_free_streams():
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    model = _hybrid(
        backbone_type="ssm_lite",
        memory={"enabled": False},
        rhythm={"enabled": False},
        motif={"enabled": False},
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


@pytest.mark.parametrize("kernel", ["bad", "", None])
def test_temporal_hybrid_rejects_unknown_rhythm_kernel(kernel):
    if kernel is None:
        assert _hybrid(scan={"rhythm_kernel": kernel}).hparams.scan["rhythm_kernel"] == "torch"
    else:
        with pytest.raises(ValueError, match="rhythm_kernel"):
            _hybrid(scan={"rhythm_kernel": kernel})


def test_triton_rhythm_features_and_repeated_updates():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from graphids.core.models.temporal._hybrid.kernels.rhythm import (
        rhythm_features_update_triton,
    )

    rhythm = _hybrid(rhythm={"enabled": True}).rhythm
    rows = torch.tensor([[0, 0, 0], [1, 2, 0], [4, 3, 6], [2, 1, -1]], device="cuda", dtype=torch.float32)
    expected = rows.repeat(3, 1)
    actual = expected.clone()
    # Repeated logical IDs belong to disjoint lane slices; vectors are strided.
    src = torch.tensor([[0, 2], [5, 5], [10, 8]], device="cuda")
    dst = torch.tensor([[1, 2], [5, 7], [11, 9]], device="cuda")
    iats = torch.tensor([[-2., 1.], [0., 4.], [3., -1.]], device="cuda")
    for _ in range(3):
        for step in range(2):
            si, di, value = src[:, step], dst[:, step], iats[:, step]
            same = si == di
            sr, dr = expected[si], expected[di]
            features = rhythm._features_batch_pair(sr, dr)
            sn, dn = rhythm._updated_rows_indexed_batch(sr, dr, value, same)
            expected[si] = sn
            expected[di] = dn
            result = rhythm_features_update_triton(actual, si, di, value, same)
            torch.testing.assert_close(result, features, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    empty = src.new_empty(0)
    assert rhythm_features_update_triton(actual, empty, empty, iats.new_empty(0), empty.bool()).shape == (0, 6)


@pytest.mark.parametrize("dim", [1, 3, 4, 5])
def test_triton_context_time_features(dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from graphids.core.models.temporal._hybrid.encoders import TemporalTimeEncoder
    from graphids.core.models.temporal._hybrid.kernels.time_features import (
        memory_time_features_triton,
        time_features_triton,
    )

    encoder = TemporalTimeEncoder(dim).cuda()
    values = torch.tensor([[-1., 0., 1e-8, 1e-4, 1., 10000.]], device="cuda").repeat(3, 1)[:, ::2]
    actual = time_features_triton(values, encoder.freqs, dim)
    torch.testing.assert_close(actual, encoder(values).flatten(1), rtol=1e-5, atol=1e-6)
    last = torch.tensor([-1., 4., 2., 8.], device="cuda")
    src = torch.tensor([[0, 2], [1, 3], [2, 2]], device="cuda")[:, 0]
    dst = torch.tensor([[0, 2], [3, 3], [2, 2]], device="cuda")[:, 0]
    now = torch.tensor([[1., 0.], [3., 0.], [12., 0.]], device="cuda")[:, 0]
    elapsed = torch.stack([torch.where(last[si] >= 0, now - last[si], 0) for si in (src, dst)], -1)
    result = memory_time_features_triton(last, src, dst, now, encoder.freqs, dim)
    torch.testing.assert_close(result, encoder(elapsed).flatten(1), rtol=1e-5, atol=1e-6)
    assert time_features_triton(values[:0], encoder.freqs, dim).shape == (0, 3 * dim)


@pytest.mark.parametrize("length", [1, 3, 4])
def test_triton_context_motif_writer(length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from graphids.core.models.temporal._hybrid.kernels.rhythm import (
        rhythm_features_update_triton,
    )

    ids = torch.arange(4 * length, device="cuda").reshape(4, length)
    iats = ids.float()
    expected_ids, expected_iats = ids.clone(), iats.clone()
    lanes = torch.tensor([0, 2], device="cuda")
    src = torch.tensor([0, 8], device="cuda")
    dst = torch.tensor([0, 9], device="cuda")
    dest = torch.tensor([0, 1], device="cuda")
    values = torch.tensor([-2., 3.], device="cuda")
    state = torch.zeros(16, 3, device="cuda")
    for _ in range(3):
        if length > 1:
            expected_ids[lanes, :-1] = expected_ids[lanes, 1:]
            expected_iats[lanes, :-1] = expected_iats[lanes, 1:]
        expected_ids[lanes, -1] = dest
        expected_iats[lanes, -1] = values
        rhythm_features_update_triton(
            state, src, dst, values, src == dst,
            motif_ids=ids, motif_iats=iats, lane_idx=lanes, dst_idx=dest,
        )
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(iats, expected_iats)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("options", [
    {"time_kernel": "triton"},
    {"rhythm_kernel": "triton", "motif_writer": "triton"},
    {"rhythm_kernel": "triton", "time_kernel": "triton", "motif_writer": "triton"},
])
def test_triton_context_lane_outputs_states_and_gradients(device, options):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        pytest.importorskip("triton")
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    params = dict(
        objective="joint", backbone_type="ssm_lite", in_channels=17,
        memory={"type": "tgn", "enabled": True, "time_encoding_dim": 3},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 5},
    )
    actual_model = _hybrid(**params, scan=options).to(device)
    reference = _hybrid(**params).to(device)
    reference.load_state_dict(actual_model.state_dict())
    actual_state = expected_state = None
    actual_loss = expected_loss = 0
    for batch in TemporalStreamLaneLoader(_uneven_lane_semantic_data(), stream_lanes=2, chunk_size=3):
        batch = batch.to(device)
        actual, actual_state = actual_model._forward_with_state(batch, actual_state)
        expected, expected_state = reference._forward_with_state(batch, expected_state)
        _assert_outputs_equal(actual, expected)
        _assert_state_equal(actual_state, expected_state)
        actual_loss = actual_loss + actual["features"].square().sum() + actual["logits"].square().sum()
        expected_loss = expected_loss + expected["features"].square().sum() + expected["logits"].square().sum()
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (_, ref_parameter) in zip(actual_model.named_parameters(), reference.named_parameters()):
        if ref_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-4, atol=2e-5, msg=name)
    if device == "cuda":
        for kind in ("rhythm", "memory_time", "motif_time"):
            assert not getattr(actual_model, f"_{kind}_kernel_failed", False)


@pytest.mark.parametrize("options", [
    {"time_kernel": "bad"}, {"motif_writer": "bad"}, {"motif_writer": "triton"},
])
def test_triton_context_scan_validation(options):
    with pytest.raises(ValueError):
        _hybrid(scan=options)


@pytest.mark.parametrize("backbone", ["gru", "ssm_lite"])
def test_continuous_lane_chunks_match_full_recording_reference(backbone):
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    data = _lane_semantic_data()
    model = _hybrid(in_channels=17, backbone_type=backbone, rhythm={"enabled": True},
                    motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4}).eval()
    state = None
    observed = {}
    with torch.no_grad():
        for batch in TemporalStreamLaneLoader(data, stream_lanes=4, chunk_size=2):
            out, state = model._forward_with_state(batch, state)
            for idx, features in zip(batch.source_index[batch.valid_mask].tolist(),
                                     out["features"][batch.valid_mask.reshape(-1)], strict=True):
                observed[idx] = features
        for sid in data.stream_id.unique():
            mask = data.stream_id == sid
            temporal = _slice_temporal(data, mask)
            expected, _ = _reference_forward_with_state(model, temporal)
            actual = torch.stack([observed[i] for i in mask.nonzero().flatten().tolist()])
            torch.testing.assert_close(actual, expected["features"], rtol=1e-5, atol=1e-6)
        assert state["rhythm"].eq(0).all()
        assert state["memory"].eq(0).all()
        assert state["motif_ids"].eq(0).all()


def test_warmup_only_batches_skip_optimizer_and_preserve_weights(tmp_path):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    data = _lane_semantic_data()
    dataset = TemporalStreamLaneLoader(data, stream_lanes=2, chunk_size=1,
                                      stream_execution="segmented", segment_events=2, segment_warmup_events=2)
    # Exercise real replay batches with no scored lanes.
    batches = [b for b in dataset if b.warmup_mask.any() and not b.is_scored.any()]
    assert batches
    model = _hybrid(in_channels=17)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    steps = []
    class StepCounter(pl.Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            steps.append(trainer.global_step)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                         callbacks=[StepCounter()], logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         default_root_dir=tmp_path, limit_val_batches=0)
    trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
    assert not steps
    assert model._train_state is not None
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)
