"""Post-training artifact generation. jsonargparse reads __init__ for config."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch

from graphids.config import cache_dir, data_dir
from graphids.core.models.registry import get_module_cls

log = structlog.get_logger()


class Analyzer:
    """Generate analysis artifacts from trained checkpoints.

    Loads models and data, runs inference, saves results (NPZ/JSON/Parquet).
    All checkpoint paths are explicit config — no hidden derivation.
    """

    def __init__(
        self,
        # --- required ---
        ckpt_path: str,
        dataset: str,
        model_type: str,
        # --- paths ---
        lake_root: str = "experimentruns",
        output_dir: str = "artifacts/",
        # --- artifact toggles ---
        embeddings: bool = True,
        attention: bool = False,
        cka: bool = False,
        landscape: bool = False,
        fusion_policy: bool = False,
        # --- CKA ---
        cka_teacher_ckpt: str = "",
        cka_max_samples: int = 500,
        # --- landscape ---
        landscape_resolution: int = 51,
        landscape_scale: float = 1.0,
        landscape_max_graphs: int = 500,
        # --- embeddings ---
        embedding_max_samples: int = 2000,
        attention_max_samples: int = 50,
        # --- data ---
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.ckpt_path = ckpt_path
        self.dataset = dataset
        self.model_type = model_type
        self.lake_root = lake_root
        self.output_dir = Path(output_dir)
        self.embeddings = embeddings
        self.attention = attention
        self.cka = cka
        self.landscape = landscape
        self.fusion_policy = fusion_policy
        self.cka_teacher_ckpt = cka_teacher_ckpt
        self.cka_max_samples = cka_max_samples
        self.landscape_resolution = landscape_resolution
        self.landscape_scale = landscape_scale
        self.landscape_max_graphs = landscape_max_graphs
        self.embedding_max_samples = embedding_max_samples
        self.attention_max_samples = attention_max_samples
        self.batch_size = batch_size
        self.seed = seed

        # fail-loud validation
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        if self.cka and not self.cka_teacher_ckpt:
            raise ValueError("cka=true requires cka_teacher_ckpt")
        if self.cka and not Path(self.cka_teacher_ckpt).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.cka_teacher_ckpt}")

    def run(self) -> None:
        """Load models, load data, generate all enabled artifacts."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("analyzer_start", model_type=self.model_type, dataset=self.dataset,
                 output_dir=str(self.output_dir))

        # Load model from checkpoint
        module = get_module_cls(self.model_type).load_from_checkpoint(
            self.ckpt_path, map_location=device,
        )
        module.eval()
        model = module.model
        hparams = module.hparams

        # Load validation data
        from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
        root = cache_dir(self.lake_root, self.dataset)
        raw = data_dir(self.lake_root, self.dataset)
        val_ds = CANBusDataset(root=root, raw_dir=raw, split="val", seed=self.seed)
        val_data = list(val_ds)
        log.info("data_loaded", n_val=len(val_data))

        if self.embeddings:
            from .embeddings import collect_and_save_embeddings
            log.info("artifact_start", artifact="embeddings")
            collect_and_save_embeddings(
                model, val_data, device, self.output_dir,
                model_type=self.model_type,
                max_samples=self.embedding_max_samples,
                batch_size=self.batch_size,
            )

        if self.attention:
            from .embeddings import collect_and_save_attention
            log.info("artifact_start", artifact="attention")
            collect_and_save_attention(
                model, val_data, device, self.output_dir,
                max_samples=self.attention_max_samples,
            )

        if self.cka:
            from .cka import compute_and_save_cka
            log.info("artifact_start", artifact="cka")
            teacher_module = get_module_cls("gat").load_from_checkpoint(
                self.cka_teacher_ckpt, map_location=device,
            )
            teacher_module.eval()
            compute_and_save_cka(
                model, teacher_module.model, val_data, device, self.output_dir,
                max_samples=self.cka_max_samples,
            )
            del teacher_module
            torch.cuda.empty_cache()

        if self.landscape:
            from .loss_landscape import compute_and_save_loss_landscape
            log.info("artifact_start", artifact="landscape")
            compute_and_save_loss_landscape(
                model, self.model_type, val_data, device, self.output_dir, hparams,
                resolution=self.landscape_resolution,
                scale=self.landscape_scale,
                max_graphs=self.landscape_max_graphs,
                dataset=self.dataset,
                seed=self.seed,
            )

        if self.fusion_policy:
            from .fusion_policy import save_fusion_policy
            log.info("artifact_start", artifact="fusion_policy")
            agent = module.agent
            # FusionDataModule provides pre-computed states
            from graphids.core.preprocessing.datamodule import FusionDataModule
            dm = FusionDataModule(
                dataset=self.dataset, lake_root=self.lake_root, seed=self.seed,
            )
            dm.setup("test")
            states = dm.val_cache["states"].to(device)
            labels = dm.val_cache["labels"]
            result = agent.predict(states)
            import numpy as np
            save_fusion_policy(
                self.output_dir,
                alphas=result["alphas"].cpu().numpy(),
                labels=labels.numpy(),
                q_values=agent.q_values(result["norm_states"]).cpu().numpy(),
            )

        log.info("analyzer_done", output_dir=str(self.output_dir))
