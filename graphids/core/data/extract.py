"""Extract and cache fusion features as a TensorDict.

Each upstream model implements ``extract_features(batch, device) -> dict[str, Tensor]``
returning per-graph named feature tensors. This module collects those dicts
under the model name (``vgae``, ``gat``, ...), stacks across batches, and saves
the resulting nested TensorDict to disk. No flat state vector, no offsets —
the fusion side reads keys directly.

Invoked as the first row of ``configs/plans/fusion.jsonnet`` (an ``ExtractRow``);
``graphids exec/submit`` dispatch routes through ``orchestrate.run_row`` →
``orchestrate.extract`` → ``extract_states``. Idempotent on ``output_dir``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from structlog import get_logger
from tensordict import TensorDict
from torch_geometric.loader import DataLoader as PyGDataLoader

log = get_logger(__name__)

FUSION_STATES_DIR = "fusion_states"
TRAIN_FILENAME = "train_states.pt"
VAL_FILENAME = "val_states.pt"
# v4 — TensorDict cache shape; per-extractor named feature dicts replace the
# flat [N, 15] state vector + LAYOUT offsets. Old caches are incompatible by
# format (top-level keys changed); the version field forces re-extraction.
# v5: DGI extract_features now returns dict (pos_stats/conf/z_stats), not flat 8D tensor.
# v6: per-graph ``attack_type`` propagated into the cache + ``attack_type_names``
# stashed in the blob so fusion test_step can emit ``auroc_per_attack/{name}``.
CACHE_VERSION = 6


def _extract_states(
    models: dict[str, torch.nn.Module],
    data,
    device: torch.device,
    max_samples: int = 150_000,
    batch_size: int = 256,
) -> TensorDict:
    """Run each model's ``extract_features`` and collect into a TensorDict.

    Returns a TensorDict with shape ``[N]`` and keys:
      - ``<model_name>``: nested TensorDict of that model's named feature tensors.
      - ``labels``: ``[N]`` int.
    """
    from contextlib import ExitStack

    from graphids.core.models.base import eval_mode

    capped = data[:max_samples]
    loader = PyGDataLoader(capped, batch_size=batch_size)

    chunks: list[TensorDict] = []
    labels_chunks: list[torch.Tensor] = []
    attack_chunks: list[torch.Tensor] = []
    with ExitStack() as stack, torch.no_grad():
        for model in models.values():
            stack.enter_context(eval_mode(model))
        for batch in loader:
            # PyG Data.to() is in-place — clone first to keep the source pristine.
            batch = batch.clone().to(device, non_blocking=True)
            n = batch.y.size(0)
            per_model = {name: m.extract_features(batch, device) for name, m in models.items()}
            chunk = TensorDict(per_model, batch_size=[n])
            chunks.append(chunk)
            labels_chunks.append(batch.y)
            at = getattr(batch, "attack_type", None)
            if at is not None:
                attack_chunks.append(at)

    td = torch.cat(chunks, dim=0)
    td["labels"] = torch.cat(labels_chunks)
    if attack_chunks:
        td["attack_type"] = torch.cat(attack_chunks)
    return td


def extract_states(
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
    """Load model checkpoints, extract and cache fusion features.

    Idempotent per-file: each split's cache is checked independently so
    re-running after adding test splits only extracts the missing files.
    """
    from graphids.core.data.datamodule.graph import GraphDataModule
    from graphids.core.data.datasets.can_bus import CANBusSource
    from graphids.core.models.base import safe_load_checkpoint

    # Build DM first so test split names are known before the idempotency check.
    source = CANBusSource(
        name=dataset,
        seed=seed,
        window_size=window_size,
        stride=stride,
        val_fraction=val_fraction,
    )
    dm = GraphDataModule(dataset=source, dynamic_batching=False)
    dm.setup(None)

    out = Path(output_dir) / FUSION_STATES_DIR
    train_path = out / TRAIN_FILENAME
    val_path = out / VAL_FILENAME
    test_paths = {name: out / f"{name}_states.pt" for name in dm.test_datasets.keys()}

    def _version_ok(p: Path) -> bool:
        if not p.exists():
            return False
        try:
            return (
                torch.load(p, map_location="cpu", weights_only=False).get("version")
                == CACHE_VERSION
            )
        except Exception:
            return False

    if all(_version_ok(p) for p in [train_path, val_path, *test_paths.values()]):
        log.info("cache_hit", output_dir=str(out), version=CACHE_VERSION)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for model_type, ckpt_path in checkpoints.items():
        log.info("loading_model", model_type=model_type, ckpt=ckpt_path)
        module = safe_load_checkpoint(model_type, Path(ckpt_path), map_location=device)
        module.to(device).eval()
        models[model_type] = module

    train_ds, val_ds = dm.train_dataset, dm.val_dataset

    # Stash schema's attack code → name map so the fusion test path can emit
    # ``auroc_per_attack/{name}`` keys (looked up in FusionDataModule).
    schema = getattr(type(train_ds), "SCHEMA", None)
    names_map = getattr(schema, "attack_type_names", None) if schema is not None else None
    blob_extras = {"version": CACHE_VERSION, "attack_type_names": dict(names_map or {0: "benign"})}

    out.mkdir(parents=True, exist_ok=True)

    if not _version_ok(train_path):
        log.info("extracting_train", n_graphs=len(train_ds), max_samples=max_samples)
        train_td = _extract_states(models, list(train_ds), device, max_samples, batch_size).cpu()
        torch.save({"td": train_td.to_dict(), **blob_extras}, train_path)

    if not _version_ok(val_path):
        log.info("extracting_val", n_graphs=len(val_ds), max_samples=max_val_samples)
        val_td = _extract_states(models, list(val_ds), device, max_val_samples, batch_size).cpu()
        torch.save({"td": val_td.to_dict(), **blob_extras}, val_path)

    for name, test_ds in dm.test_datasets.items():
        p = test_paths[name]
        if not _version_ok(p):
            n = len(test_ds)
            log.info("extracting_test", split=name, n_graphs=n)
            test_td = _extract_states(models, list(test_ds), device, n, batch_size).cpu()
            torch.save({"td": test_td.to_dict(), **blob_extras}, p)

    log.info("states_saved", output_dir=str(out), version=CACHE_VERSION)
