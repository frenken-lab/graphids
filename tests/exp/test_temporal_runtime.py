from __future__ import annotations


def _install_temporal_smoke_data(monkeypatch):
    import lightning.pytorch as pl
    import torch
    from torch_geometric.data import TemporalData
    from torch_geometric.loader import TemporalDataLoader

    from graphids.exp import runtime

    def temporal(labels: list[int], *, scored: list[bool] | None = None) -> TemporalData:
        n = len(labels)
        event_id = torch.arange(n, dtype=torch.long)
        src = torch.arange(n, dtype=torch.long) % 3
        dst = (src + torch.tensor([1, 0, 1, 0, 1, 0][:n], dtype=torch.long)).clamp_max(3)
        msg = torch.stack(
            [
                torch.tensor([float(label), float(i % 2), float(i), 1.0])
                for i, label in enumerate(labels)
            ]
        )
        return TemporalData(
            src=src,
            dst=dst,
            t=event_id.float(),
            msg=msg,
            y=torch.tensor(labels, dtype=torch.long),
            attack_type=torch.tensor(labels, dtype=torch.long),
            stream_id=torch.zeros(n, dtype=torch.long),
            reset_after=torch.tensor([False] * (n - 1) + [True]),
            event_id=event_id,
            is_scored=torch.tensor(scored or [True] * n, dtype=torch.bool),
        )

    class TemporalSmokeData(pl.LightningDataModule):
        def setup(self, stage=None):
            self.train = temporal([0, 1, 0, 1])
            self.val = temporal([0, 1, 0], scored=[False, True, True])
            self.test_data = {"holdout": temporal([0, 1])}

        @property
        def num_ids(self):
            return 3

        @property
        def in_channels(self):
            return 4

        @property
        def num_classes(self):
            return 2

        def train_dataloader(self):
            return TemporalDataLoader(self.train, batch_size=2)

        def val_dataloader(self):
            return TemporalDataLoader(self.val, batch_size=2)

        def test_dataloader(self):
            return [TemporalDataLoader(self.test_data["holdout"], batch_size=2)]

    original_build = runtime._build_component

    def build_component(spec, **build_kwargs):
        if isinstance(spec, dict) and spec.get("type") == "temporal_smoke_data":
            return TemporalSmokeData()
        return original_build(spec, **build_kwargs)

    monkeypatch.setattr(runtime, "_build_component", build_component)
    return runtime


def _run_fit_smoke(tmp_path, *, model: dict, loss_fn: dict):
    from graphids.exp import runtime
    from graphids.exp.config import FitRunPayload, OutputConfig, RunConfig

    run = RunConfig(
        name=f"temporal-{model['type']}-smoke",
        stage="fit",
        dataset="synthetic",
        payload=FitRunPayload(
            data={"type": "temporal_smoke_data"},
            model=model,
            loss_fn=loss_fn,
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
        outputs=OutputConfig(run_dir=tmp_path / f"temporal-{model['type']}-smoke"),
    )

    return runtime.run_stage(run)


def test_runtime_runs_temporal_event_classifier_smoke(monkeypatch, tmp_path):
    _install_temporal_smoke_data(monkeypatch)
    result = _run_fit_smoke(
        tmp_path,
        model={"type": "temporal_event_classifier", "hidden": 8, "layers": 2, "embedding_dim": 4},
        loss_fn={"type": "ce"},
    )
    assert result["stage"] == "fit"
    assert result["trainer"] == "Trainer"
    assert {"train_loss", "train_acc", "val_loss", "val_acc"} <= set(result["metrics"])


def test_runtime_runs_gat_on_temporal_data(monkeypatch, tmp_path):
    _install_temporal_smoke_data(monkeypatch)
    result = _run_fit_smoke(
        tmp_path,
        model={
            "type": "gat",
            "scale": "small",
            "gradient_checkpointing": False,
        },
        loss_fn={"type": "ce"},
    )
    assert {"train_loss", "train_acc", "val_loss", "val_acc"} <= set(result["metrics"])


def test_runtime_runs_vgae_on_temporal_data(monkeypatch, tmp_path):
    _install_temporal_smoke_data(monkeypatch)
    result = _run_fit_smoke(
        tmp_path,
        model={"type": "vgae", "scale": "small"},
        loss_fn={
            "type": "vgae_task",
            "kl_weight": 0.01,
            "canid_weight": 0.1,
            "nbr_weight": 0.1,
            "edge_weight": 0.0,
        },
    )
    assert {"train_loss", "train_recon", "val_loss", "val_recon_mean"} <= set(result["metrics"])
