"""Extract fusion state vectors from upstream VGAE + GAT checkpoints.

Operation layer — argparse surface lives in
``graphids.commands.extract_fusion_states``.

Short GPU job (~2 min). Saves cached states to disk so fusion training
can run on CPU without loading upstream models.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from graphids.log import get_logger

log = get_logger(__name__)

FUSION_STATES_DIR = "fusion_states"
TRAIN_FILENAME = "train_states.pt"
VAL_FILENAME = "val_states.pt"


def extract_fusion_states(
    *,
    vgae_ckpt: str,
    gat_ckpt: str,
    dataset: str,
    output_dir: str,
    max_samples: int = 150_000,
    max_val_samples: int = 30_000,
    batch_size: int = 256,
    seed: int = 42,
    window_size: int = 100,
    stride: int = 100,
    val_fraction: float = 0.2,
) -> None:
    """Load VGAE + GAT checkpoints, cache fusion states to ``output_dir``.

    Reads the dataset via ``load_datasets``, runs predictions through both
    upstream models via ``FusionDataModule.cache_predictions``, then writes
    ``train_states.pt`` / ``val_states.pt`` under ``{output_dir}/fusion_states``.
    """
    from graphids.core.data.datamodule.fusion import FusionDataModule
    from graphids.core.data.datamodule.graph import load_datasets
    from graphids.core.data.datasets.can_bus import CANBusDataset
    from graphids.core.models._training import load_inner_model

    lake_root = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading_upstream_models", vgae=vgae_ckpt, gat=gat_ckpt, device=str(device))
    vgae, _ = load_inner_model("vgae", Path(vgae_ckpt), device)
    gat, _ = load_inner_model("gat", Path(gat_ckpt), device)
    models = {"vgae": vgae, "gat": gat}

    train_ds, val_ds, _ = load_datasets(
        dataset=dataset,
        lake_root=lake_root,
        seed=seed,
        window_size=window_size,
        stride=stride,
        train_val_split=1.0 - val_fraction,
        dataset_cls=CANBusDataset,
    )

    log.info("extracting_train_states", n_graphs=len(train_ds), max_samples=max_samples)
    train_cache = FusionDataModule.cache_predictions(
        models,
        list(train_ds),
        device,
        max_samples,
        batch_size=batch_size,
    )

    log.info("extracting_val_states", n_graphs=len(val_ds), max_samples=max_val_samples)
    val_cache = FusionDataModule.cache_predictions(
        models,
        list(val_ds),
        device,
        max_val_samples,
        batch_size=batch_size,
    )

    # Move to CPU for disk serialization
    train_cache = {k: v.cpu() for k, v in train_cache.items()}
    val_cache = {k: v.cpu() for k, v in val_cache.items()}

    out = Path(output_dir) / FUSION_STATES_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, out / TRAIN_FILENAME)
    torch.save(val_cache, out / VAL_FILENAME)

    log.info(
        "fusion_states_saved",
        output_dir=str(out),
        train_states=list(train_cache["states"].shape),
        val_states=list(val_cache["states"].shape),
    )
