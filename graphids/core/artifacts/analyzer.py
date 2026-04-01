"""Post-training artifact generation. jsonargparse reads __init__ for config."""

from __future__ import annotations

import os
from pathlib import Path

import structlog
import torch

from graphids.config import cache_dir, data_dir
from graphids.core.models._training import safe_load_checkpoint
from .tasks import run_attention, run_cka, run_embeddings, run_fusion_policy, run_landscape
from .validation import validate_inputs

log = structlog.get_logger()


class Analyzer:
    """Generate analysis artifacts from trained checkpoints.

    Loads models and data, runs inference, saves results (NPZ/JSON/Parquet).
    All checkpoint paths are explicit config — no hidden derivation.
    """

    def __init__(
        self,
        # --- required (no defaults → jsonargparse enforces) ---
        ckpt_path: str,
        dataset: str,
        model_type: str,
        # --- paths ---
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
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

        validate_inputs(
            ckpt_path=self.ckpt_path,
            cka=self.cka,
            cka_teacher_ckpt=self.cka_teacher_ckpt,
            fusion_policy=self.fusion_policy,
            vgae_ckpt_path=self.vgae_ckpt_path,
            gat_ckpt_path=self.gat_ckpt_path,
        )

    def run(self) -> None:
        """Load models, load data, generate all enabled artifacts."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("analyzer_start", model_type=self.model_type, dataset=self.dataset,
                 output_dir=str(self.output_dir))

        # Load model from checkpoint (with migration guard)
        module = safe_load_checkpoint(self.model_type, self.ckpt_path, map_location=device)
        module.eval()
        model = module.model
        hparams = module.hparams

        # Load validation data
        val_data = self._load_val_data()

        if self.embeddings:
            run_embeddings(
                model=model,
                val_data=val_data,
                device=device,
                output_dir=self.output_dir,
                model_type=self.model_type,
                max_samples=self.embedding_max_samples,
                batch_size=self.batch_size,
            )

        if self.attention:
            run_attention(
                model=model,
                val_data=val_data,
                device=device,
                output_dir=self.output_dir,
                max_samples=self.attention_max_samples,
            )

        if self.cka:
            run_cka(
                model=model,
                val_data=val_data,
                device=device,
                output_dir=self.output_dir,
                teacher_ckpt=self.cka_teacher_ckpt,
                max_samples=self.cka_max_samples,
            )

        if self.landscape:
            run_landscape(
                model=model,
                model_type=self.model_type,
                val_data=val_data,
                device=device,
                output_dir=self.output_dir,
                hparams=hparams,
                resolution=self.landscape_resolution,
                scale=self.landscape_scale,
                max_graphs=self.landscape_max_graphs,
                dataset=self.dataset,
                seed=self.seed,
            )

        if self.fusion_policy:
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
        from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
        root = cache_dir(self.lake_root, self.dataset)
        raw = data_dir(self.lake_root, self.dataset)
        val_ds = CANBusDataset(
            root=root, raw_dir=raw, split="val", seed=self.seed,
            window_size=self.window_size, stride=self.stride,
        )
        val_data = list(val_ds)
        log.info("data_loaded", n_val=len(val_data))
        return val_data

