import pytest
from pydantic import ValidationError

from graphids.exp.config import ExperimentConfig
from graphids.exp.runtime import resolve_spec


def test_snapshot_model_and_data_specs_resolve_without_building_cuda():
    model = resolve_spec({"type": "vehicle_graph_mamba_joint", "width": 64})
    data = resolve_spec(
        {
            "type": "can_snapshot_dm",
            "manifest_path": "reviewed.json",
            "datasets": ["set_01"],
            "representation_cfg": {"kind": "can_snapshot", "bin_seconds": 0.01},
        }
    )
    assert model.width == 64 and model.backend == "official"
    assert data.representation_cfg.kind == "can_snapshot"
    assert data.stream_batch_size == 1
    with pytest.raises(ValidationError):
        resolve_spec(
            {
                "type": "can_snapshot_dm",
                "manifest_path": "x",
                "datasets": ["set_01"],
                "stream_lanes": 16,
            }
        )


def test_snapshot_representation_mismatch_is_rejected():
    cfg = ExperimentConfig.from_yaml("configs/experiments/graph_mamba_joint.yml")
    changed = dict(cfg.config["data"])
    changed["representation_cfg"] = {"kind": "can_snapshot", "bin_seconds": 0.02}
    with pytest.raises(ValueError, match="must match"):
        cfg.build_run(name="test", stage="fit", config={"data": changed})


def test_presets_target_supported_gpu_clusters():
    for name, cluster in [
        ("graph_mamba_joint", "ascend"),
        ("graph_mamba_joint_cardinal", "cardinal"),
        ("graph_mamba_joint_parallel", "ascend"),
    ]:
        cfg = ExperimentConfig.from_yaml(f"configs/experiments/{name}.yml")
        run = cfg.build_run(name=name, stage=cfg.stage)
        assert run.resources.cluster == cluster
        assert run.payload.model["type"] == "vehicle_graph_mamba_joint"
        assert run.representation_cfg.kind == "can_snapshot"


def test_removed_model_names_have_migration_errors():
    from graphids.migration import reject_retired

    for name in [
        "temporal_event_classifier",
        "temporal_rnn_classifier",
        "temporal_gat",
        "temporal_vgae",
        "temporal_hybrid",
        "temporal_dm",
    ]:
        with pytest.raises(ValueError, match="graphids-temporal-legacy"):
            reject_retired({"type": name})


def test_learning_run_uses_full_epochs_and_source_checkpoint_selection():
    cfg = ExperimentConfig.from_yaml(
        "configs/experiments/graph_mamba_ctat_learning.yml"
    )
    run = cfg.build_run(name="learning", stage="fit")
    assert run.payload.trainer["max_epochs"] == 10
    assert isinstance(run.payload.trainer["limit_train_batches"], float)
    assert run.payload.trainer["limit_train_batches"] == 1.0
    checkpoint = run.payload.callbacks["checkpoint"]["init_args"]
    assert checkpoint["monitor"] == "val/zero-shot/classification_ap_macro"
    assert checkpoint["mode"] == "max"
    assert checkpoint["save_top_k"] == 1 and checkpoint["save_last"]
    assert not checkpoint["enable_version_counter"]
    assert run.payload.data["datasets"] == ["set_01"]
