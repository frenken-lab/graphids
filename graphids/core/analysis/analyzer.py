"""Post-training artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from graphids._otel import get_logger
from graphids.config.topology import cache_dir, data_dir
from graphids.core.models.base import safe_load_checkpoint

log = get_logger(__name__)


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
        model_type: Literal["vgae", "dgi", "gat", "fusion"],
        # --- paths ---
        lake_root: str | None = None,
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
        window_size: int = 100,
        stride: int = 100,
        batch_size: int = 256,
        seed: int = 42,
        # --- fusion (only needed when fusion_policy=true) ---
        vgae_ckpt_path: str = "",
        gat_ckpt_path: str = "",
    ):
        if lake_root is None:
            from graphids.config.settings import get_settings

            lake_root = get_settings().lake_root
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
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.seed = seed
        self.vgae_ckpt_path = vgae_ckpt_path
        self.gat_ckpt_path = gat_ckpt_path

        # Runtime file-existence checks (schema deps validated by AnalysisSpec).
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        if self.cka and not Path(self.cka_teacher_ckpt).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.cka_teacher_ckpt}")

    def run(self) -> None:
        """Load models, load data, generate all enabled artifacts."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "analyzer_start",
            model_type=self.model_type,
            dataset=self.dataset,
            output_dir=str(self.output_dir),
        )

        from graphids.core.models.base import eval_mode

        # Load model from checkpoint (with migration guard)
        module = safe_load_checkpoint(self.model_type, self.ckpt_path, map_location=device)
        with eval_mode(module):
            model = module.model
            hparams = module.hparams

            # Load validation data
            val_data = self._load_val_data()

            if self.embeddings:
                from .embeddings import collect_and_save_embeddings

                log.info("artifact_start", artifact="embeddings")
                collect_and_save_embeddings(
                    model,
                    val_data,
                    device,
                    self.output_dir,
                    model_type=self.model_type,
                    max_samples=self.embedding_max_samples,
                    batch_size=self.batch_size,
                )

            if self.attention:
                from .embeddings import collect_and_save_attention

                log.info("artifact_start", artifact="attention")
                collect_and_save_attention(
                    model,
                    val_data,
                    device,
                    self.output_dir,
                    max_samples=self.attention_max_samples,
                )

            if self.cka:
                from .cka import compute_and_save_cka

                compute_and_save_cka(
                    model,
                    val_data,
                    device,
                    self.output_dir,
                    teacher_ckpt=self.cka_teacher_ckpt,
                    max_samples=self.cka_max_samples,
                )

            if self.landscape:
                from .loss_landscape import compute_and_save_loss_landscape

                log.info("artifact_start", artifact="landscape")
                compute_and_save_loss_landscape(
                    model,
                    self.model_type,
                    val_data,
                    device,
                    self.output_dir,
                    hparams,
                    resolution=self.landscape_resolution,
                    scale=self.landscape_scale,
                    max_graphs=self.landscape_max_graphs,
                    dataset=self.dataset,
                    seed=self.seed,
                )

            if self.fusion_policy:
                from .fusion_policy import run_fusion_policy

                run_fusion_policy(
                    module=module,
                    dataset=self.dataset,
                    lake_root=self.lake_root,
                    seed=self.seed,
                    vgae_ckpt_path=self.vgae_ckpt_path,
                    gat_ckpt_path=self.gat_ckpt_path,
                    window_size=self.window_size,
                    stride=self.stride,
                    output_dir=self.output_dir,
                    device=device,
                )

        log.info("analyzer_done", output_dir=str(self.output_dir))

    def _load_val_data(self) -> list:
        """Load validation dataset using same window_size/stride as training."""
        from graphids.core.data.datasets.can_bus import CANBusDataset

        root = cache_dir(self.lake_root, self.dataset)
        raw = data_dir(self.lake_root, self.dataset)
        # val shares train's tensor (split_tag defaults to "train"); we only
        # read the cache, so source_dirs isn't needed here. val_fraction
        # must match training (0.2 — the CANBusSource default).
        val_ds = CANBusDataset(
            root=root,
            raw_dir=raw,
            split="val",
            val_fraction=0.2,
            seed=self.seed,
            window_size=self.window_size,
            stride=self.stride,
        )
        val_data = list(val_ds)
        log.info("data_loaded", n_val=len(val_data))
        return val_data
