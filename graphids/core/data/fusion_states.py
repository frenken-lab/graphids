"""Extract and cache state vectors from any model with ``extract_features``.

General-purpose: given a dict of ``{name: model}``, runs each model's
``extract_features(batch, device)`` method on the dataset, concatenates
the per-model feature vectors, and saves to disk.

Models must implement ``extract_features(batch, device) → Tensor[N, D]``
on their module class (VGAE, GAT, or any future model).

CLI surface: ``python -m graphids extract-fusion-states``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from graphids._otel import get_logger
from graphids.core.data.sampler import make_graph_loader

log = get_logger(__name__)

FUSION_STATES_DIR = "fusion_states"
TRAIN_FILENAME = "train_states.pt"
VAL_FILENAME = "val_states.pt"


def extract_states(
    models: dict[str, torch.nn.Module],
    data,
    device: torch.device,
    max_samples: int = 150_000,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Run ``model.extract_features`` for each model, concatenate feature vectors.

    Args:
        models: ``{name: model}`` — each model must have ``extract_features(batch, device)``.
        data: list of PyG Data objects.
        device: target device.
        max_samples: cap on number of graphs to process.
        batch_size: batch size for the graph loader.

    Returns:
        ``{"states": Tensor[N, D_total], "labels": Tensor[N]}``
    """
    for model in models.values():
        model.eval()

    capped = data[:max_samples]
    loader = make_graph_loader(capped, batch_size=batch_size)

    states, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            feats = [model.extract_features(batch, device) for model in models.values()]
            states.append(torch.cat(feats, dim=1))
            labels.append(batch.y)

    return {"states": torch.cat(states), "labels": torch.cat(labels)}


def extract_fusion_states(
    *,
    checkpoints: dict[str, str],
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
    """Load model checkpoints, extract and cache fusion states to ``output_dir``.

    Args:
        checkpoints: ``{model_type: ckpt_path}`` e.g. ``{"vgae": "/path/to/best.ckpt", "gat": "..."}``.
    """
    from graphids.config.settings import get_settings
    from graphids.core.data.datamodule.graph import load_datasets
    from graphids.core.data.datasets.can_bus import CANBusDataset
    from graphids.core.models.base import load_inner_model

    lake_root = get_settings().lake_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for model_type, ckpt_path in checkpoints.items():
        log.info("loading_model", model_type=model_type, ckpt=ckpt_path)
        model, _ = load_inner_model(model_type, Path(ckpt_path), device)
        models[model_type] = model

    train_ds, val_ds, _ = load_datasets(
        dataset=dataset, lake_root=lake_root, seed=seed,
        window_size=window_size, stride=stride,
        train_val_split=1.0 - val_fraction, dataset_cls=CANBusDataset,
    )

    log.info("extracting_train", n_graphs=len(train_ds), max_samples=max_samples)
    train_cache = extract_states(models, list(train_ds), device, max_samples, batch_size)

    log.info("extracting_val", n_graphs=len(val_ds), max_samples=max_val_samples)
    val_cache = extract_states(models, list(val_ds), device, max_val_samples, batch_size)

    train_cache = {k: v.cpu() for k, v in train_cache.items()}
    val_cache = {k: v.cpu() for k, v in val_cache.items()}

    out = Path(output_dir) / FUSION_STATES_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, out / TRAIN_FILENAME)
    torch.save(val_cache, out / VAL_FILENAME)

    log.info("states_saved", output_dir=str(out),
             train_shape=list(train_cache["states"].shape),
             val_shape=list(val_cache["states"].shape))
