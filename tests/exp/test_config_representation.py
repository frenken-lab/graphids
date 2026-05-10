from __future__ import annotations


def test_experiment_config_propagates_representation_cfg():
    from graphids.core.data.preprocessing.representations import (
        SnapshotRepresentationCfg,
    )
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(
        experiment_name="demo",
        dataset="toy",
        representation_cfg=SnapshotRepresentationCfg(window_size=17, stride=5),
    )
    run = cfg.build_run(name="demo-run", stage="extract")

    assert run.representation_cfg.window_size == 17
    assert run.representation_cfg.stride == 5
    assert run.payload.dataset == "toy"
    assert run.payload.val_fraction == 0.2


def test_runconfig_reports_representation_in_metadata():
    from graphids.core.data.preprocessing.representations import (
        SnapshotRepresentationCfg,
    )
    from graphids.exp.config import (
        ExtractRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )

    run = RunConfig(
        name="demo",
        stage="extract",
        outputs=OutputConfig(run_dir="/tmp/graphids-demo"),
        resources=ResourceConfig(),
        representation_cfg=SnapshotRepresentationCfg(window_size=11, stride=3),
        payload=ExtractRunPayload(dataset="toy", output_dir="/tmp/out"),
    )

    assert run.mlflow_tags()["graphids.representation"] == "snapshot"
    assert run.mlflow_hparams(backend="local")["graphids.representation"] == "snapshot"
