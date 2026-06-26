from __future__ import annotations


def test_experiment_config_defaults_to_temporal_representation():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(experiment_name="demo", dataset="toy")
    run = cfg.build_run(name="demo-run", stage="extract")

    assert representation_kind(cfg.representation_cfg) == "temporal"
    assert representation_kind(run.representation_cfg) == "temporal"


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


def test_fit_config_rejects_representation_drift():
    import pytest

    from graphids.core.data.preprocessing.representations import (
        SnapshotRepresentationCfg,
        SnapshotSequenceRepresentationCfg,
    )
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig(
        experiment_name="demo",
        dataset="set_01",
        representation_cfg=SnapshotRepresentationCfg(window_size=100, stride=100),
        config={
            "data": {
                "type": "graph_dm",
                "source": {
                    "type": "can_bus",
                    "dataset": "set_01",
                    "seed": 42,
                    "representation_cfg": SnapshotSequenceRepresentationCfg(
                        window_size=100,
                        stride=100,
                        sequence_length=3,
                        sequence_stride=1,
                    ),
                },
            },
            "model": {"type": "gat", "sequence_pool": "gru"},
        },
    )

    with pytest.raises(ValueError, match="must match top-level representation_cfg"):
        cfg.build_run(name="demo-run", stage="fit")


def test_snapshot_sequence_smoke_config_resolves_to_fit_payload():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp import runtime
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/experiments/gat_snapshot_sequence_smoke.yml")
    run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

    assert representation_kind(run.representation_cfg) == "snapshot_sequence"
    assert run.payload.model["sequence_pool"] == "gru"

    data = runtime._build_component(run.payload.data)
    assert representation_kind(data.source.representation_cfg) == "snapshot_sequence"
    assert data.batch_size == 1
    assert data.num_workers == 0
    assert data.dynamic_batching is True
    assert data.require_cache is True
    assert data.min_steps_per_epoch == 2


def test_snapshot_sequence_real_config_uses_budgeted_batches():
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp import runtime
    from graphids.exp.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/experiments/gat_snapshot_sequence_real.yml")
    run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

    data = runtime._build_component(run.payload.data)
    assert representation_kind(data.source.representation_cfg) == "snapshot_sequence"
    assert data.dynamic_batching is True
    assert data.min_steps_per_epoch == 64


def test_snapshot_sequence_cache_config_resolves_to_cache_payload(monkeypatch):
    from graphids.core.data.preprocessing.representations import representation_kind
    from graphids.exp import runtime
    from graphids.exp.config import CacheRunPayload, ExperimentConfig

    monkeypatch.setenv("GRAPHIDS_LAKE_ROOT", "/tmp/graphids-lake")
    cfg = ExperimentConfig.from_yaml("configs/experiments/gat_snapshot_sequence_cache.yml")
    run = cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)

    assert run.stage == "cache"
    assert isinstance(run.payload, CacheRunPayload)
    assert representation_kind(run.representation_cfg) == "snapshot_sequence"
    data = runtime._build_component(run.payload.data)
    assert representation_kind(data.source.representation_cfg) == "snapshot_sequence"
    assert (
        data.source.cache_root_path().name
        == "snapshot_sequence_32e9bbb5f88a_voc_train_val_0.2_gap_2"
    )


def test_runtime_runs_snapshot_sequence_training_smoke(monkeypatch, tmp_path):
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch_geometric.data import Batch, Data

    from graphids.exp import runtime
    from graphids.exp.config import FitRunPayload, OutputConfig, RunConfig

    def graph(sequence_id: int) -> Data:
        steps = 3
        nodes_per_step = 3
        num_nodes = steps * nodes_per_step
        x = torch.rand(num_nodes, 5)
        node_id = torch.arange(num_nodes) % 12
        node_sequence_step = torch.arange(steps).repeat_interleave(nodes_per_step)
        edges = []
        for step in range(steps):
            start = step * nodes_per_step
            src = torch.arange(start, start + nodes_per_step - 1)
            dst = torch.arange(start + 1, start + nodes_per_step)
            edges.append(torch.stack([src, dst]))
        edge_index = torch.cat(edges, dim=1)
        edge_sequence_step = node_sequence_step[edge_index[0]]
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.rand(edge_index.shape[1], 11),
            node_id=node_id,
            y=torch.tensor([sequence_id % 2]),
            node_sequence_step=node_sequence_step,
            edge_sequence_step=edge_sequence_step,
            sequence_id=torch.tensor([sequence_id]),
            sequence_length=torch.tensor([steps]),
            sequence_stride=torch.tensor([1]),
        )

    class SequenceSmokeData(pl.LightningDataModule):
        num_ids = 12
        in_channels = 5
        num_classes = 2
        test_datasets = {}

        def setup(self, stage=None):
            self.batch = Batch.from_data_list([graph(0), graph(1)])

        def train_dataloader(self):
            return TorchDataLoader([self.batch], batch_size=None)

        def val_dataloader(self):
            return TorchDataLoader([self.batch], batch_size=None)

    original_build = runtime._build_component

    def build_component(spec, **build_kwargs):
        if isinstance(spec, dict) and spec.get("type") == "sequence_smoke_data":
            return SequenceSmokeData()
        return original_build(spec, **build_kwargs)

    monkeypatch.setattr(runtime, "_build_component", build_component)
    run = RunConfig(
        name="sequence-smoke",
        stage="fit",
        dataset="synthetic",
        payload=FitRunPayload(
            data={"type": "sequence_smoke_data"},
            model={"type": "gat", "scale": "small", "sequence_pool": "gru"},
            loss_fn={"type": "ce"},
            trainer={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
            },
        ),
        outputs=OutputConfig(run_dir=tmp_path / "sequence-smoke"),
    )

    result = runtime.run_stage(run)
    assert result["stage"] == "fit"
    assert result["trainer"] == "Trainer"
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "val_auroc"} <= set(
        result["metrics"]
    )
