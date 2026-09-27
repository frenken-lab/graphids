from __future__ import annotations

import pytest
import torch
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
        in_channels=4,
        num_classes=2,
    )


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
