"""Extract fusion state vectors from upstream VGAE + GAT checkpoints.

Short GPU job (~2 min). Saves cached states to disk so fusion training
can run on CPU without loading upstream models.

Usage:
    python -m graphids extract-fusion-states \
        --vgae-ckpt /path/to/vgae/best_model.ckpt \
        --gat-ckpt /path/to/gat/best_model.ckpt \
        --dataset set_01 \
        --output-dir /path/to/fusion_run/seed_42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from graphids.log import get_logger

log = get_logger(__name__)

FUSION_STATES_DIR = "fusion_states"
TRAIN_FILENAME = "train_states.pt"
VAL_FILENAME = "val_states.pt"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Extract fusion state vectors from upstream checkpoints",
    )
    parser.add_argument("--vgae-ckpt", required=True, help="Path to VGAE best_model.ckpt")
    parser.add_argument("--gat-ckpt", required=True, help="Path to GAT best_model.ckpt")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. set_01)")
    parser.add_argument("--output-dir", required=True, help="Run directory to save states into")
    parser.add_argument("--max-samples", type=int, default=150_000)
    parser.add_argument("--max-val-samples", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    extract_and_save(
        vgae_ckpt=args.vgae_ckpt,
        gat_ckpt=args.gat_ckpt,
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        window_size=args.window_size,
        stride=args.stride,
        val_fraction=args.val_fraction,
    )


def extract_and_save(
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
    lake_root: str | None = None,
) -> Path:
    """Extract fusion states and save to disk. Returns the output directory."""
    import types

    from graphids.core.models._training import load_inner_model
    from graphids.core.preprocessing.datamodule import FusionDataModule, load_datasets

    lake_root = lake_root or os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading_upstream_models", vgae=vgae_ckpt, gat=gat_ckpt, device=str(device))
    vgae, _ = load_inner_model("vgae", Path(vgae_ckpt), device)
    gat, _ = load_inner_model("gat", Path(gat_ckpt), device)
    models = {"vgae": vgae, "gat": gat}

    cfg = types.SimpleNamespace(
        dataset=dataset, lake_root=lake_root, seed=seed,
        preprocessing=types.SimpleNamespace(
            window_size=window_size, stride=stride,
            train_val_split=1.0 - val_fraction,
        ),
    )
    train_ds, val_ds, _ = load_datasets(cfg)

    log.info("extracting_train_states", n_graphs=len(train_ds), max_samples=max_samples)
    train_cache = FusionDataModule.cache_predictions(
        models, list(train_ds), device, max_samples, batch_size=batch_size,
    )

    log.info("extracting_val_states", n_graphs=len(val_ds), max_samples=max_val_samples)
    val_cache = FusionDataModule.cache_predictions(
        models, list(val_ds), device, max_val_samples, batch_size=batch_size,
    )

    # Move to CPU for disk serialization
    train_cache = {k: v.cpu() for k, v in train_cache.items()}
    val_cache = {k: v.cpu() for k, v in val_cache.items()}

    out = Path(output_dir) / FUSION_STATES_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, out / TRAIN_FILENAME)
    torch.save(val_cache, out / VAL_FILENAME)

    log.info("fusion_states_saved",
             output_dir=str(out),
             train_states=list(train_cache["states"].shape),
             val_states=list(val_cache["states"].shape))

    del vgae, gat, models
    return out
