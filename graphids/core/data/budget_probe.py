"""Profile the budget module's sizing chain on real hardware.

Operation layer — CLI surface lives in ``graphids.cli._slurm``.

Instantiates each (model_type, scale) with random weights, loads cached
datasets, runs _probe_vram() for VRAM measurements, then calibrate_at_budget()
for real T_collation and T_gpu at the operating batch size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from graphids.config.constants import CONFIG_DIR
from graphids.config.jsonnet import render
from graphids.config.paths import cache_dir
from graphids.log import get_logger

log = get_logger(__name__)


def _find_cached_datasets(lake_root: str) -> list[str]:
    """Return dataset names that have cache_metadata.json on disk."""
    from graphids.config.paths import dataset_names

    available = []
    for ds in dataset_names():
        metadata = cache_dir(lake_root, ds) / "cache_metadata.json"
        if metadata.exists():
            available.append(ds)
    return available


def _instantiate_model(
    model_type: str, scale: str, num_ids: int, in_channels: int, conv_type: str | None = None
):
    """Instantiate a model by rendering the model Jsonnet and importing the class directly.

    When *conv_type* is provided it overrides the jsonnet default so the probe
    can measure VRAM for different convolution backends (O(E) vs O(N²)).
    """
    import importlib
    import inspect

    from graphids.config.constants import FAMILY_FOR_MODEL_TYPE
    from graphids.core.losses.build import build_loss

    family = FAMILY_FOR_MODEL_TYPE[model_type]
    model_cfg = render(
        CONFIG_DIR / "models" / "_expand.jsonnet",
        tla={"family": family, "model_type": model_type, "scale": scale},
    )

    class_path = model_cfg["model"]["class_path"]
    init_args = dict(model_cfg["model"].get("init_args", {}))
    init_args["num_ids"] = num_ids
    init_args["in_channels"] = in_channels

    # Override conv_type if caller requests a different backend.
    if conv_type is not None:
        init_args["conv_type"] = conv_type

    # loss_fn is an nn.Module excluded from jsonnet — build a default for probing.
    loss_fn = build_loss(model_type, init_args.pop("loss_config", None), distillation_config=None)
    if loss_fn is not None:
        init_args["loss_fn"] = loss_fn

    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)

    # Filter to params accepted by __init__ (rendered scale configs may carry metadata keys)
    sig = inspect.signature(cls.__init__)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        valid = set(sig.parameters) - {"self"}
        init_args = {k: v for k, v in init_args.items() if k in valid}

    return cls(**init_args)


def _warmup_training_state(model, dataset, device, step_fn) -> None:
    """Allocate optimizer state and torch.compile caches in VRAM.

    Replicates the VRAM footprint of a real training step so that
    subsequent ``torch.cuda.mem_get_info()`` calls reflect actual free
    VRAM during training — not the inflated value before optimizer/compile.

    Uses the LightningModule's own ``configure_optimizers`` to match the
    real optimizer (Adam, lr schedule, weight decay, etc.).
    """
    import torch

    if device.type != "cuda" or step_fn is None:
        return

    from torch_geometric.data import Batch

    from graphids.core.data.budget import _collect_graphs, _extract_loss

    # Build Adam from hparams — configure_optimizers() needs a Trainer
    # (for scheduler T_max), but we only need the optimizer to allocate
    # its m/v state tensors in VRAM.
    hp = model.hparams
    lr = getattr(hp, "lr", 1e-3)
    wd = getattr(hp, "weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # One real fwd+bwd+step to allocate Adam state tensors + compile caches
    warmup_graphs = _collect_graphs(dataset, 500)
    warmup_batch = Batch.from_data_list(warmup_graphs).to(device)
    was_training = model.training
    model.train()
    loss = _extract_loss(step_fn(warmup_batch))
    loss.backward()
    optimizer.step()  # noqa: probe — allocates Adam m/v state in VRAM
    optimizer.zero_grad(set_to_none=True)  # noqa: probe — free grad tensors
    if not was_training:
        model.eval()
    del warmup_batch, loss, warmup_graphs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Keep optimizer alive so its state tensors stay in VRAM.
    # Stash on the model so it isn't garbage collected before node_budget().
    model._probe_optimizer = optimizer

    log.info(
        "warmup_training_state",
        optimizer=type(optimizer).__name__,
        param_count=sum(p.numel() for p in model.parameters()),
        compiled=hasattr(model, "_orig_mod"),
    )


_CALIBRATION_FRACTIONS = [0.25, 0.50, 0.75, 1.0]


def _probe_combo(
    model_type: str,
    scale: str,
    dataset_name: str,
    lake_root: str,
    device,
    conv_type: str | None = None,
) -> list[dict]:
    """Run VRAM probe + multi-point calibration for one (model, scale, dataset, conv_type).

    Measures at multiple fractions of the VRAM budget so downstream plots
    can fit the throughput model (α, β, γ) from raw data points.
    Returns one row per measurement point.
    """
    import torch

    from graphids.config.paths import data_dir
    from graphids.core.data.budget import (
        _probe_vram,
        calibrate_at_budget,
        conv_complexity,
        node_budget,
    )
    from graphids.core.data.datasets.can_bus import CANBusDataset

    # --- Load dataset from cache ---
    root = cache_dir(lake_root, dataset_name)
    raw = data_dir(lake_root, dataset_name)
    ds = CANBusDataset(root=root, raw_dir=raw, split="train")

    sample = ds[0]
    in_channels = sample.x.shape[1]
    num_ids = ds.num_arb_ids

    # --- Instantiate model ---
    model = _instantiate_model(model_type, scale, num_ids, in_channels, conv_type=conv_type)
    model = model.to(device)

    # Resolve effective conv_type (may come from jsonnet default if not overridden).
    effective_conv = conv_type or getattr(model, "hparams", {}).get("conv_type", "gatv2")

    # --- Replicate training VRAM footprint before measuring ---
    step_fn = getattr(model, "_step", None)
    _warmup_training_state(model, ds, device, step_fn)

    # --- VRAM probe (with optimizer + compile state already resident) ---
    bytes_per_node, backward_mult = _probe_vram(model, ds, step_fn=step_fn)

    # --- Node budget (free VRAM now reflects optimizer + compile overhead) ---
    result = node_budget(
        dataset_name,
        lake_root,
        model=model,
        train_dataset=ds,
        conv_type=effective_conv,
    )

    # --- Multi-point calibration ---
    shared = {
        "model_type": model_type,
        "scale": scale,
        "conv_type": effective_conv,
        "complexity": conv_complexity(effective_conv),
        "dataset": dataset_name,
        "bytes_per_node": bytes_per_node,
        "backward_multiplier": round(backward_mult, 2),
        "budget": result.budget,
        "mem_budget": result.mem_budget,
        "mean_nodes": round(result.mean_nodes, 1),
    }

    rows = []
    for frac in _CALIBRATION_FRACTIONS:
        target = max(1, int(result.budget * frac))
        t_collation, t_gpu, n_graphs = calibrate_at_budget(
            model,
            ds,
            target,
            backward_multiplier=backward_mult,
        )
        if t_collation <= 0 and t_gpu <= 0:
            continue
        rows.append(
            {
                **shared,
                "fraction": frac,
                "target_nodes": target,
                "n_graphs": n_graphs,
                "t_collation_ms": round(t_collation * 1000, 2),
                "t_gpu_ms": round(t_gpu * 1000, 2),
            }
        )

    del model
    torch.cuda.empty_cache()

    return rows


def _format_table(results: list[dict]) -> str:
    """Format results as a readable multi-point calibration table."""
    if not results:
        return "No results."

    header = (
        f"{'model':<12} {'scale':<6} {'conv':<12} {'cplx':<5} {'dataset':<10} {'frac':>5} "
        f"{'target':>10} {'graphs':>7} "
        f"{'T_c(ms)':>9} {'T_g(ms)':>9} "
        f"{'B/node':>8} {'bwd':>5} {'budget':>10}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['model_type']:<12} {r['scale']:<6} {r['conv_type']:<12} "
            f"{r['complexity']:<5} {r['dataset']:<10} "
            f"{r['fraction']:>5.2f} "
            f"{r['target_nodes']:>10,} {r['n_graphs']:>7,} "
            f"{r['t_collation_ms']:>9.2f} {r['t_gpu_ms']:>9.2f} "
            f"{r['bytes_per_node']:>8,} {r['backward_multiplier']:>5.2f} "
            f"{r['budget']:>10,}"
        )

    return "\n".join(lines)


def run_probe_budget(
    *,
    model_types: list[str],
    scales: list[str],
    conv_types: list[str] | None = None,
    datasets: list[str] | None,
    lake_root: str,
    json_output: bool = False,
    dry_run: bool = False,
) -> None:
    """Run VRAM probe + multi-point calibration over ``(model, scale, conv_type, dataset)``.

    When ``conv_types`` is ``None``, uses each model's jsonnet default (gatv2).
    When ``datasets`` is ``None``, auto-discovers all datasets that have a
    ``cache_metadata.json`` sidecar under ``lake_root``. Writes results to
    ``{lake_root}/reference/budget_calibration.csv`` unless ``dry_run`` is
    set. Requires CUDA — exits with status 1 otherwise.
    """
    import torch

    if not torch.cuda.is_available():
        print(
            "ERROR: probe-budget requires a GPU. Submit via: scripts/slurm/submit.sh profile-budget",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda")

    resolved_datasets = datasets or _find_cached_datasets(lake_root)
    if not resolved_datasets:
        print(f"ERROR: no cached datasets found under {lake_root}", file=sys.stderr)
        sys.exit(1)

    # None → [None] means "use jsonnet default"; explicit list sweeps conv types.
    resolved_conv_types: list[str | None] = conv_types if conv_types else [None]

    total = len(model_types) * len(scales) * len(resolved_conv_types) * len(resolved_datasets)
    log.info(
        "profile_budget_start",
        combos=total,
        models=model_types,
        scales=scales,
        conv_types=conv_types,
        datasets=resolved_datasets,
    )

    results: list[dict] = []
    errors: list[dict] = []

    for model_type in model_types:
        for scale in scales:
            for ct in resolved_conv_types:
                for dataset_name in resolved_datasets:
                    ct_label = ct or "default"
                    label = f"{model_type}/{scale}/{ct_label}/{dataset_name}"
                    try:
                        rows = _probe_combo(
                            model_type, scale, dataset_name, lake_root, device, conv_type=ct
                        )
                        results.extend(rows)
                        log.info("probe_done", label=label, points=len(rows))
                    except Exception as e:
                        log.error("probe_failed", label=label, error=str(e))
                        errors.append({"label": label, "error": str(e)})

    # --- Write to data lake ---
    out_path = None
    if results and not dry_run:
        import csv
        import os

        from graphids.config.paths import require_lake_write

        require_lake_write()
        out_dir = Path(lake_root) / "reference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "budget_calibration.csv"

        fieldnames = list(results[0].keys())
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(out_path)
        log.info("budget_csv_written", path=str(out_path), rows=len(results))

    # --- Print ---
    if json_output:
        output = {"results": results, "errors": errors}
        print(json.dumps(output, indent=2))
    else:
        print(_format_table(results))
        if errors:
            print(f"\n{len(errors)} probe(s) failed:")
            for e in errors:
                print(f"  {e['label']}: {e['error']}")

    if out_path is not None:
        print(f"\nWritten to {out_path}")
