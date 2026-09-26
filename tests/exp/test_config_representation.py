from __future__ import annotations


def test_experiment_config_defaults_to_temporal_representation():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(experiment_name="demo", dataset="toy")
    run = cfg.build_run(name="demo-run", stage="fit")

    assert representation_kind(cfg.representation_cfg) == "temporal"
    assert representation_kind(run.representation_cfg) == "temporal"


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
        assert data.batch_size == 512
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
        }
    )

    assert data.source.train_source_mode == "attack_free"
    assert representation_kind(data.source.representation_cfg) == "temporal"


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
        assert data.batch_size == 512

    anomaly_cfg = ExperimentConfig.from_yaml("configs/experiments/temporal_hybrid_anomaly_smoke.yml")
    anomaly_run = anomaly_cfg.build_run(
        name=anomaly_cfg.experiment_name,
        stage=anomaly_cfg.stage,
        config=anomaly_cfg.config,
    )
    assert anomaly_run.payload.data["source"]["train_source_mode"] == "attack_free"
    assert anomaly_run.payload.loss_fn is None


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

    assert len(final_paths) == 40

    supervised = [p for p in final_paths if "/supervised/" in str(p)]
    anomaly = [p for p in final_paths if "/anomaly/" in str(p)]
    assert len(supervised) == 30
    assert len(anomaly) == 10

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
            assert cfg.plan_id == "temporal_supervised_final_2026_09_26"
            assert source["train_source_mode"] == "mixed"
            assert run.payload.model["type"] in {
                "temporal_event_classifier",
                "temporal_gat",
                "temporal_rnn_classifier",
            }
            assert run.payload.loss_fn == {"type": "ce"}
        else:
            assert cfg.plan_id == "temporal_anomaly_final_2026_09_26"
            assert source["train_source_mode"] == "attack_free"
            assert run.payload.model["type"] == "temporal_vgae"
            assert run.payload.loss_fn is None

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

    assert len(train_runs) == 20
    assert len(test_runs) == 20
    for test_name, test_run in test_runs.items():
        train_name = test_name.removesuffix("_test") + "_train"
        assert train_name in train_runs
        assert test_run.payload.ckpt_path == str(train_runs[train_name].outputs.checkpoint_path() / "best_model.ckpt")
