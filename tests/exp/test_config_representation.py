from __future__ import annotations

import pytest


def test_experiment_config_defaults_to_temporal_representation():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(experiment_name="demo", dataset="toy")
    run = cfg.build_run(name="demo-run", stage="fit")

    assert representation_kind(cfg.representation_cfg) == "temporal"
    assert representation_kind(run.representation_cfg) == "temporal"
    assert run.mlflow_tags()["graphids.phase"] == "fit"
    assert run.mlflow_tags()["graphids.group"] == "fit"
    assert run.mlflow_tags()["graphids.variant"] == "demo-run"


def test_hybrid_result_views_load():
    from graphids.exp.results import load_result_view

    supervised = load_result_view("hybrid_supervised_final")
    anomaly = load_result_view("hybrid_anomaly_final")
    joint = load_result_view("hybrid_joint_final")

    assert supervised["filter"]["plan_id"] == "temporal_supervised_hybrid_final_2026_09_27"
    assert anomaly["filter"]["plan_id"] == "temporal_anomaly_hybrid_final_2026_09_27"
    assert joint["filter"]["plan_id"] == "temporal_joint_hybrid_final_2026_09_27"
    assert "test/test/auroc_macro" in supervised["metrics"]
    assert "test/auroc_macro" in anomaly["metrics"]


def test_experiment_config_from_yaml_resolves_git_sha(monkeypatch, tmp_path):
    import graphids.exp.config as config_mod
    from graphids.exp.config import ExperimentConfig

    monkeypatch.setattr(config_mod, "_current_git_sha", lambda: "abc123")
    path = tmp_path / "experiment.yml"
    path.write_text(
        """
experiment_name: demo
dataset: toy
config: {}
resources: {}
""".lstrip()
    )

    cfg = ExperimentConfig.from_yaml(path)
    run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

    assert cfg.git_sha == "abc123"
    assert run.git_sha == "abc123"


def test_experiment_config_from_yaml_preserves_explicit_git_sha(monkeypatch, tmp_path):
    import graphids.exp.config as config_mod
    from graphids.exp.config import ExperimentConfig

    monkeypatch.setattr(config_mod, "_current_git_sha", lambda: "abc123")
    path = tmp_path / "experiment.yml"
    path.write_text(
        """
experiment_name: demo
dataset: toy
git_sha: explicit456
config: {}
resources: {}
""".lstrip()
    )

    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.git_sha == "explicit456"


def test_temporal_smoke_configs_resolve_without_window_or_budget_knobs():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.ray_backend import build_component

    for path, model_type in (
        ("configs/experiments/temporal_event_classifier_smoke.yml", "temporal_event_classifier"),
        ("configs/experiments/rnn_temporal_smoke.yml", "temporal_rnn_classifier"),
        ("configs/experiments/gat_temporal_smoke.yml", "temporal_gat"),
        ("configs/experiments/vgae_temporal_smoke.yml", "temporal_vgae"),
    ):
        raw = open(path).read()
        assert "window_size" not in raw
        assert "stride" not in raw
        assert "dynamic_batching" not in raw

        cfg = ExperimentConfig.from_yaml(path)
        run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

        assert representation_kind(run.representation_cfg) == "temporal"
        assert run.payload.model["type"] == model_type
        data = build_component(run.payload.data)
        from graphids.core.data.datamodule.temporal import TemporalDataModule

        assert isinstance(data, TemporalDataModule)
        assert representation_kind(data.source.representation_cfg) == "temporal"
        expected_batch_size = 64 if path.endswith(("temporal_hybrid_ssm_smoke.yml", "temporal_hybrid_anomaly_smoke.yml")) else 512
        assert data.batch_size == expected_batch_size
        assert data.source.val_warmup_events == 64
        assert data.source.test_warmup_events == 64
        assert data.source.train_source_mode == "mixed"


def test_temporal_data_config_accepts_attack_free_train_source_mode():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.ray_backend import build_component

    data = build_component(
        {
            "type": "temporal_dm",
            "source": {
                "type": "can_bus",
                "dataset": "hcrl_sa",
                "seed": 42,
                "train_source_mode": "attack_free",
                "representation_cfg": {"kind": "temporal"},
            },
            "batch_size": 128,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }
    )

    assert data.source.train_source_mode == "attack_free"
    assert representation_kind(data.source.representation_cfg) == "temporal"
    assert data.batch_size == 128
    assert data.num_workers == 2
    assert data.pin_memory is True
    assert data.persistent_workers is True


def test_temporal_data_config_accepts_and_validates_stream_lanes():
    from pydantic import ValidationError

    from graphids.exp.ray_backend import build_component
    from graphids.primitives import can_bus, temporal_dm

    source = can_bus(dataset="hcrl_sa", seed=42, representation_cfg={"kind": "temporal"})
    cfg = temporal_dm(source=source, batch_mode="stream_lanes", stream_lanes=4, chunk_size=16)
    data = cfg.build()

    assert data.batch_mode == "stream_lanes"
    assert data.stream_lanes == 4
    assert data.chunk_size == 16

    built = build_component(
        {
            "type": "temporal_dm",
            "source": {
                "type": "can_bus",
                "dataset": "hcrl_sa",
                "seed": 42,
                "representation_cfg": {"kind": "temporal"},
            },
            "batch_mode": "stream_lanes",
            "stream_lanes": 4,
            "chunk_size": 16,
        }
    )
    assert built.batch_mode == "stream_lanes"

    with pytest.raises(ValidationError, match="stream_lanes must be >= 2"):
        temporal_dm(source=source, batch_mode="stream_lanes", stream_lanes=1, chunk_size=16)
    with pytest.raises(ValidationError, match="chunk_size must be positive"):
        temporal_dm(source=source, batch_mode="stream_lanes", stream_lanes=2, chunk_size=0)


def test_temporal_hybrid_smoke_configs_parse():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.ray_backend import build_component

    paths = [
        "configs/experiments/temporal_hybrid_memory_smoke.yml",
        "configs/experiments/temporal_hybrid_gru_smoke.yml",
        "configs/experiments/temporal_hybrid_ssm_smoke.yml",
        "configs/experiments/temporal_hybrid_anomaly_smoke.yml",
    ]
    for path in paths:
        cfg = ExperimentConfig.from_yaml(path)
        run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

        assert representation_kind(run.representation_cfg) == "temporal"
        assert run.payload.model["type"] == "temporal_hybrid"
        data = build_component(run.payload.data)
        from graphids.core.data.datamodule.temporal import TemporalDataModule

        assert isinstance(data, TemporalDataModule)
        assert representation_kind(data.source.representation_cfg) == "temporal"
        expected_batch_size = 64 if path.endswith(("temporal_hybrid_ssm_smoke.yml", "temporal_hybrid_anomaly_smoke.yml")) else 512
        assert data.batch_size == expected_batch_size
        assert "compile" not in run.payload.model

    anomaly_cfg = ExperimentConfig.from_yaml("configs/experiments/temporal_hybrid_anomaly_smoke.yml")
    anomaly_run = anomaly_cfg.build_run(
        name=anomaly_cfg.experiment_name,
        stage=anomaly_cfg.stage,
        config=anomaly_cfg.config,
    )
    assert anomaly_run.payload.data["source"]["train_source_mode"] == "attack_free"
    assert anomaly_run.payload.loss_fn is None


def test_temporal_hybrid_primitive_rejects_invalid_objective_head_combos():
    from pydantic import ValidationError

    from graphids.primitives import temporal_hybrid

    with pytest.raises(ValidationError, match="objective='supervised' requires heads.classification=true"):
        temporal_hybrid(objective="supervised", heads={"classification": False})

    with pytest.raises(ValidationError, match="objective='anomaly' requires heads.classification=false"):
        temporal_hybrid(objective="anomaly", heads={"classification": True})

    with pytest.raises(ValidationError, match="objective='anomaly' requires at least one anomaly head"):
        temporal_hybrid(
            objective="anomaly",
            heads={"next_id": False, "iat": False, "payload_delta": False},
        )

    with pytest.raises(ValidationError, match="objective='joint' requires at least one anomaly head"):
        temporal_hybrid(
            objective="joint",
            heads={
                "classification": True,
                "next_id": False,
                "iat": False,
                "payload_delta": False,
            },
        )

    with pytest.raises(ValidationError, match="memory.time_encoding_dim must be non-negative"):
        temporal_hybrid(objective="supervised", memory={"time_encoding_dim": -1})

    with pytest.raises(ValidationError, match="motif.length must be positive"):
        temporal_hybrid(objective="supervised", motif={"enabled": True, "length": 0})

    with pytest.raises(ValidationError, match="anomaly.min_log_scale must be <= anomaly.max_log_scale"):
        temporal_hybrid(objective="supervised", anomaly={"min_log_scale": 1.0, "max_log_scale": 0.0})

    with pytest.raises(ValidationError, match="motif.embedding_dim must be positive"):
        temporal_hybrid(objective="supervised", motif={"enabled": True, "embedding_dim": 0})


def test_temporal_hybrid_primitive_accepts_rich_modular_options():
    from graphids.primitives import temporal_hybrid

    cfg = temporal_hybrid(
        objective="joint",
        memory={"time_encoding_dim": 4},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
    )

    assert cfg.memory.time_encoding_dim == 4
    assert cfg.anomaly.mode == "nll"
    assert cfg.rhythm.enabled is True
    assert cfg.motif.length == 3


def test_temporal_hybrid_diagnostic_profile_uses_profiler_without_compile():
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_profile.yml"
    )
    run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

    assert run.payload.model["type"] == "temporal_hybrid"
    assert "compile" not in run.payload.model
    assert run.payload.trainer["limit_train_batches"] == 100
    assert run.payload.trainer["limit_val_batches"] == 25
    profiler = run.payload.callbacks["profiler"]
    assert profiler["class_path"] == "graphids.core.callbacks.TemporalBatchProfilerCallback"
    assert profiler["init_args"] == {
        "prefix": "profile",
        "warmup_batches": 5,
        "log_every_n_batches": 10,
        "sync_cuda": True,
    }


def test_temporal_hybrid_lane_profile_configs_parse():
    from graphids.exp.config import ExperimentConfig

    baseline = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_profile.yml"
    )
    lane = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile.yml"
    )
    amp = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_amp_profile.yml"
    )
    lane16 = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml"
    )
    lane32 = ExperimentConfig.from_yaml(
        "configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_profile.yml"
    )

    baseline_run = baseline.build_run(name=baseline.experiment_name, stage=baseline.stage, config=baseline.config)
    lane_run = lane.build_run(name=lane.experiment_name, stage=lane.stage, config=lane.config)
    amp_run = amp.build_run(name=amp.experiment_name, stage=amp.stage, config=amp.config)
    lane16_run = lane16.build_run(name=lane16.experiment_name, stage=lane16.stage, config=lane16.config)
    lane32_run = lane32.build_run(name=lane32.experiment_name, stage=lane32.stage, config=lane32.config)

    assert baseline_run.payload.data.get("batch_mode", "events") == "events"
    assert lane_run.payload.data["batch_mode"] == "stream_lanes"
    assert lane_run.payload.data["stream_lanes"] >= 2
    assert lane_run.payload.data["chunk_size"] >= 1
    assert lane_run.payload.trainer["enable_checkpointing"] is False
    assert amp_run.payload.trainer["precision"] == "16-mixed"
    assert lane_run.payload.data["stream_lanes"] * lane_run.payload.data["chunk_size"] == 512
    assert lane16_run.payload.data["batch_mode"] == "stream_lanes"
    assert lane16_run.payload.data["stream_lanes"] == 16
    assert lane16_run.payload.data["chunk_size"] == 32
    assert lane16_run.payload.data["stream_lanes"] * lane16_run.payload.data["chunk_size"] == 512
    assert lane32_run.payload.data["batch_mode"] == "stream_lanes"
    assert lane32_run.payload.data["stream_lanes"] == 32
    assert lane32_run.payload.data["chunk_size"] == 16
    assert lane32_run.payload.data["stream_lanes"] * lane32_run.payload.data["chunk_size"] == 512


def test_config_string_placeholders_resolve_against_run_paths(monkeypatch, tmp_path):
    import graphids.paths as paths_mod
    from graphids.exp.config import ExperimentConfig

    monkeypatch.setattr(paths_mod, "trial_dir", lambda: tmp_path / "runs")
    cfg = ExperimentConfig(
        experiment_name="demo_train",
        dataset="toy",
        config={
            "ckpt_path": "{trial_dir}/{dataset}/demo_train/demo_train/checkpoints/best_model.ckpt",
            "callbacks": {
                "best": {
                    "class_path": "graphids.core.callbacks.Sha256ModelCheckpoint",
                    "init_args": {"dirpath": "{run_dir}/checkpoints"},
                }
            },
        },
    )

    run = cfg.build_run(name=cfg.experiment_name, stage="fit", config=cfg.config)

    assert run.payload.ckpt_path == str(tmp_path / "runs/toy/demo_train/demo_train/checkpoints/best_model.ckpt")
    assert run.payload.callbacks["best"]["init_args"]["dirpath"] == str(run.outputs.checkpoint_path())


def test_final_temporal_configs_parse_and_encode_protocols(monkeypatch, tmp_path):
    from pathlib import Path

    import graphids.paths as paths_mod
    from graphids.exp.config import ExperimentConfig

    monkeypatch.setattr(paths_mod, "trial_dir", lambda: tmp_path / "runs")
    final_paths = sorted(Path("configs/experiments/final").glob("**/*.yml"))

    assert len(final_paths) == 80

    supervised = [p for p in final_paths if "/supervised/" in str(p)]
    anomaly = [p for p in final_paths if "/anomaly/" in str(p)]
    joint = [p for p in final_paths if "/joint/" in str(p)]
    assert len(supervised) == 50
    assert len(anomaly) == 20
    assert len(joint) == 10

    train_runs = {}
    test_runs = {}
    for path in final_paths:
        cfg = ExperimentConfig.from_yaml(path)
        run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)
        source = run.payload.data["source"]
        trainer = run.payload.trainer

        assert run.payload.data["type"] == "temporal_dm"
        assert source["dataset"] == cfg.dataset
        assert source["representation_cfg"] == {"kind": "temporal"}
        assert trainer["devices"] == 1
        assert trainer["enable_progress_bar"] is False

        if "supervised" in path.parts:
            assert cfg.plan_id in {
                "temporal_supervised_final_2026_09_26",
                "temporal_supervised_hybrid_final_2026_09_27",
            }
            assert source["train_source_mode"] == "mixed"
            assert run.payload.model["type"] in {
                "temporal_event_classifier",
                "temporal_gat",
                "temporal_hybrid",
                "temporal_rnn_classifier",
            }
            if run.payload.model["type"] == "temporal_hybrid":
                assert run.payload.model["objective"] == "supervised"
                assert run.payload.model["memory"]["enabled"] is True
                assert run.payload.model["backbone"]["type"] in {"gru", "ssm_lite"}
                assert "compile" not in run.payload.model
            assert run.payload.loss_fn == {"type": "ce"}
        elif "anomaly" in path.parts:
            assert cfg.plan_id in {
                "temporal_anomaly_final_2026_09_26",
                "temporal_anomaly_hybrid_final_2026_09_27",
            }
            assert source["train_source_mode"] == "attack_free"
            assert run.payload.model["type"] in {"temporal_vgae", "temporal_hybrid"}
            if run.payload.model["type"] == "temporal_hybrid":
                assert run.payload.model["objective"] == "anomaly"
                assert run.payload.model["anomaly"]["mode"] == "nll"
                assert run.payload.model["rhythm"]["enabled"] is True
                assert run.payload.model["motif"]["enabled"] is True
                assert "compile" not in run.payload.model
            assert run.payload.loss_fn is None
        else:
            assert "joint" in path.parts
            assert cfg.plan_id == "temporal_joint_hybrid_final_2026_09_27"
            assert source["train_source_mode"] == "mixed"
            assert run.payload.model["type"] == "temporal_hybrid"
            assert run.payload.model["objective"] == "joint"
            assert run.payload.model["anomaly"]["mode"] == "nll"
            assert run.payload.model["rhythm"]["enabled"] is True
            assert run.payload.model["motif"]["enabled"] is True
            assert "compile" not in run.payload.model
            assert run.payload.loss_fn == {"type": "ce"}

        if cfg.stage == "fit":
            assert trainer["max_epochs"] == 20
            assert trainer["enable_checkpointing"] is True
            callback = run.payload.callbacks["best"]
            assert callback["class_path"] == "graphids.core.callbacks.Sha256ModelCheckpoint"
            assert callback["monitor"] == "val_loss"
            assert callback["init_args"]["dirpath"] == str(run.outputs.checkpoint_path())
            assert callback["init_args"]["filename"] == "best_model"
            train_runs[cfg.experiment_name] = run
        else:
            assert trainer["enable_checkpointing"] is False
            assert run.payload.ckpt_path is not None
            assert "{" not in run.payload.ckpt_path
            assert run.payload.ckpt_path.endswith("/checkpoints/best_model.ckpt")
            test_runs[cfg.experiment_name] = run

    assert len(train_runs) == 40
    assert len(test_runs) == 40
    for test_name, test_run in test_runs.items():
        train_name = test_name.removesuffix("_test") + "_train"
        assert train_name in train_runs
        assert test_run.payload.ckpt_path == str(train_runs[train_name].outputs.checkpoint_path() / "best_model.ckpt")
