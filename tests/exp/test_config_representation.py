from __future__ import annotations


def test_experiment_config_defaults_to_temporal_representation():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(experiment_name="demo", dataset="toy")
    run = cfg.build_run(name="demo-run", stage="fit")

    assert representation_kind(cfg.representation_cfg) == "temporal"
    assert representation_kind(run.representation_cfg) == "temporal"


def test_temporal_smoke_configs_resolve_without_window_or_budget_knobs():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.ray_backend import build_component

    for path, model_type in (
        ("configs/experiments/temporal_event_classifier_smoke.yml", "temporal_event_classifier"),
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
