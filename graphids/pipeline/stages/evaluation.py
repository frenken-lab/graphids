"""Evaluation stage: runs inference on validation and test data using Lightning test loop.

All models evaluate via trainer.test(module, dataloaders=...) — one pattern.
Dispatcher iterates EVAL_ORDER and calls per-model eval functions.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.core.preprocessing import CANBusDataModule

from .callbacks import save_attention, save_dqn_policy, save_embeddings
from .data_loading import cache_predictions, cleanup
from .eval_inference import (
    capture_gat_artifacts,
    capture_vgae_artifacts,
    find_vgae_threshold,
    run_fusion_inference,
    test_model,
)
from .modules import GATModule, VGAEModule
from .trainer_factory import load_frozen_cfg, load_model

log = structlog.get_logger()

EVAL_ORDER = ["gat", "vgae", "fusion", "temporal"]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def evaluate(cfg) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Uses Lightning's trainer.test() with torchmetrics for metric computation.
    Returns {"metrics": {...}} with per-model and per-test-scenario results.
    """
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup()  # both fit + test
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}

    eval_fns = {
        "gat": eval_gat,
        "vgae": eval_vgae,
        "fusion": eval_fusion,
        "temporal": eval_temporal,
    }

    for model_name in EVAL_ORDER:
        ckpt_key = "dqn" if model_name == "fusion" else model_name
        if model_name == "temporal" and not cfg.temporal.enabled:
            continue
        ckpt = cfg.checkpoints.get(ckpt_key)
        if not ckpt or not Path(ckpt).exists():
            continue
        # Fusion needs all 3 checkpoints
        if model_name == "fusion" and not _fusion_checkpoints_exist(cfg):
            continue

        result = eval_fns[model_name](cfg, val_data, test_scenarios, device)
        if result is None:
            continue

        all_metrics[model_name] = result["val_metrics"]
        if result.get("test_metrics"):
            test_metrics[model_name] = result["test_metrics"]
        if result.get("artifacts"):
            artifacts[model_name] = result["artifacts"]

    # CKA (KD runs only)
    if any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        try:
            from .cka import compute_and_save_cka
            compute_and_save_cka(cfg, val_data, device, Path.cwd())
        except Exception as e:
            log.warning("cka_failed", error=str(e))

    # Persist artifacts
    save_embeddings(
        Path.cwd(),
        artifacts.get("vgae"),
        artifacts.get("gat"),
    )
    save_attention(Path.cwd(), artifacts.get("gat"))
    save_dqn_policy(Path.cwd(), artifacts.get("fusion"))

    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    cleanup()
    return {"metrics": all_metrics}


def _fusion_checkpoints_exist(cfg) -> bool:
    return all(
        Path(cfg.checkpoints.get(k, "")).exists()
        for k in ("dqn", "vgae", "gat")
    )


# ---------------------------------------------------------------------------
# Per-model eval functions
# ---------------------------------------------------------------------------


def eval_gat(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate GAT via trainer.test() + capture artifacts."""
    gat_model = load_model(cfg, "gat", "curriculum", device)
    module = GATModule(cfg)
    module.model = gat_model

    val_metrics = test_model(module, val_data)
    _log_metrics("GAT", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tdata)

    gat_result = capture_gat_artifacts(gat_model, val_data, device)

    del gat_model
    cleanup()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": gat_result,
    }


def eval_vgae(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate VGAE: threshold search on val, then test with threshold."""
    vgae_model = load_model(cfg, "vgae", "autoencoder", device)
    module = VGAEModule(cfg)
    module.model = vgae_model

    threshold, youden_j = find_vgae_threshold(module, val_data)
    module.test_threshold = threshold
    module._test_errors.clear()
    module._test_labels.clear()
    module.test_metrics.reset()

    val_metrics = test_model(module, val_data)
    val_metrics["core"]["optimal_threshold"] = threshold
    val_metrics["core"]["youden_j"] = youden_j
    _log_metrics("VGAE", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            module._test_errors.clear()
            module._test_labels.clear()
            scenario_metrics[name] = test_model(module, tdata)

    vgae_result = capture_vgae_artifacts(vgae_model, val_data, device)

    del vgae_model
    cleanup()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": vgae_result,
    }


def eval_fusion(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate fusion agent via Lightning test loop."""
    from .fusion import BanditFusionModule, DQNFusionModule

    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", "curriculum", device)
    models = {"vgae": vgae, "gat": gat}

    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)
    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    method = fusion_cfg.fusion.method if hasattr(fusion_cfg, "fusion") else cfg.fusion.method

    # Load agent
    if method == "bandit":
        from graphids.core.models.bandit import NeuralLinUCBAgent
        agent = NeuralLinUCBAgent.from_config(fusion_cfg, device=str(device))
        module = BanditFusionModule(agent)
    else:
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent.from_config(fusion_cfg, device=str(device), inference=True)
        module = DQNFusionModule(agent)

    agent.load_checkpoint(torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True))

    # MLP / weighted_avg: load their Lightning modules directly
    if method == "mlp":
        from graphids.core.models.fusion_baselines import MLPFusionModule
        from graphids.core.models.registry import fusion_state_dim
        module = MLPFusionModule(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
        ckpt = torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True)
        module.model.load_state_dict(ckpt["model"])
    elif method == "weighted_avg":
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule(lr=cfg.fusion.lr)
        ckpt = torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True)
        module.weight.data = ckpt["weight"]

    # Eval via trainer.test()
    val_loader = DataLoader(
        TensorDataset(val_cache["states"], val_cache["labels"]),
        batch_size=256, shuffle=False,
    )
    val_metrics = test_model(module, val_loader)
    _log_metrics("Fusion", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
            tl = DataLoader(
                TensorDataset(tc["states"], tc["labels"]),
                batch_size=256, shuffle=False,
            )
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tl)

    # Artifact capture (q-values / policy JSON) — only for DQN/bandit
    fusion_result = None
    if method in ("dqn", "bandit"):
        fusion_result = run_fusion_inference(agent, val_cache)

    del vgae, gat
    cleanup()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": fusion_result,
    }


def eval_temporal(cfg, val_data, test_scenarios, device) -> dict | None:
    """Evaluate temporal model via Lightning test loop."""
    try:
        from graphids.core.models.temporal import TemporalGraphClassifier
        from graphids.core.preprocessing._temporal import TemporalGrouper

        from .temporal import (
            TemporalGraphDataset,
            TemporalLightningModule,
            collate_temporal,
        )

        gat = load_model(cfg, "gat", "curriculum", device)

        # Probe embedding dim
        with torch.no_grad():
            probe = val_data[0].clone().to(device)
            _, emb = gat(probe, return_embedding=True)
            spatial_dim = emb.shape[-1]

        tc = cfg.temporal
        temporal_model = TemporalGraphClassifier(
            spatial_encoder=gat, spatial_dim=spatial_dim,
            temporal_hidden=tc.temporal_hidden, temporal_heads=tc.temporal_heads,
            temporal_layers=tc.temporal_layers, max_seq_len=tc.temporal_window,
            freeze_spatial=True, num_classes=2,
        ).to(device)
        temporal_model.load_state_dict(torch.load(
            cfg.checkpoints["temporal"], map_location="cpu", weights_only=True,
        ))

        module = TemporalLightningModule(temporal_model, cfg)

        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        val_sequences = grouper.group(val_data)
        if not val_sequences:
            return None

        val_ds = TemporalGraphDataset(val_sequences, device)
        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
        val_metrics = test_model(module, val_loader)
        _log_metrics("Temporal", val_metrics)

        del temporal_model, gat
        cleanup()
        return {"val_metrics": val_metrics, "test_metrics": {}, "artifacts": None}
    except Exception as e:
        log.warning("temporal_eval_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_metrics(name: str, metrics: dict) -> None:
    log.info(
        "val_metrics", model=name,
        **{k: round(v, 4) for k, v in metrics["core"].items() if isinstance(v, float)},
    )
