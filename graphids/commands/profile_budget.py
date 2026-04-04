"""Profile the budget module's sizing chain on real hardware.

Instantiates each (model_type, scale) with random weights, loads cached
datasets, runs _probe_vram() for VRAM measurements, then calibrate_at_budget()
for real T_collation and T_gpu at the operating batch size.

Usage (via __main__.py):
    python -m graphids probe-budget
    python -m graphids probe-budget --dataset hcrl_ch --model-type vgae
    python -m graphids probe-budget --json

SLURM:
    scripts/submit.sh profile-budget
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphids.log import get_logger

from graphids.config import (
    CONFIG_DIR,
    LAKE_ROOT,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    cache_dir,
)
from graphids.config.yaml_utils import read_yaml

log = get_logger(__name__)


def _find_cached_datasets(lake_root: str) -> list[str]:
    """Return dataset names that have cache_metadata.json on disk."""
    from graphids.config import dataset_names

    available = []
    for ds in dataset_names():
        metadata = cache_dir(lake_root, ds) / "cache_metadata.json"
        if metadata.exists():
            available.append(ds)
    return available


def _instantiate_model(model_type: str, scale: str, num_ids: int, in_channels: int):
    """Instantiate a model by reading config YAMLs and importing the class directly."""
    import importlib
    import inspect

    base_cfg = read_yaml(CONFIG_DIR / "models" / model_type / "base.yaml")
    scale_cfg = read_yaml(CONFIG_DIR / "models" / model_type / "scales" / f"{scale}.yaml")

    class_path = base_cfg["model"]["class_path"]
    init_args = {
        **base_cfg["model"].get("init_args", {}),
        **scale_cfg.get("model", {}).get("init_args", {}),
    }
    init_args["num_ids"] = num_ids
    init_args["in_channels"] = in_channels
    if model_type != "temporal":
        init_args["compile_model"] = False

    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)

    # Filter to params accepted by __init__ (scale YAMLs may have metadata keys)
    sig = inspect.signature(cls.__init__)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        valid = set(sig.parameters) - {"self"}
        init_args = {k: v for k, v in init_args.items() if k in valid}

    return cls(**init_args)


_CALIBRATION_FRACTIONS = [0.25, 0.50, 0.75, 1.0]


def _probe_combo(
    model_type: str, scale: str, dataset_name: str, lake_root: str, device,
) -> list[dict]:
    """Run VRAM probe + multi-point calibration for one (model, scale, dataset).

    Measures at multiple fractions of the VRAM budget so downstream plots
    can fit the throughput model (α, β, γ) from raw data points.
    Returns one row per measurement point.
    """
    import torch

    from graphids.config import data_dir
    from graphids.core.preprocessing.budget import (
        _probe_vram, calibrate_at_budget, node_budget,
    )
    from graphids.core.preprocessing.datasets.can_bus import CANBusDataset

    # --- Load dataset from cache ---
    root = cache_dir(lake_root, dataset_name)
    raw = data_dir(lake_root, dataset_name)
    ds = CANBusDataset(root=root, raw_dir=raw, split="train")

    sample = ds[0]
    in_channels = sample.x.shape[1]
    num_ids = ds.num_arb_ids

    # --- Instantiate model ---
    model = _instantiate_model(model_type, scale, num_ids, in_channels)
    model = model.to(device)

    # --- VRAM probe ---
    step_fn = getattr(model, "_step", None)
    bytes_per_node, backward_mult = _probe_vram(model, ds, step_fn=step_fn)

    # --- Node budget ---
    result = node_budget(
        dataset_name, lake_root, model=model, train_dataset=ds,
        conv_type=getattr(model, "hparams", {}).get("conv_type", "gatv2"),
    )

    # --- Multi-point calibration ---
    shared = {
        "model_type": model_type,
        "scale": scale,
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
            model, ds, target, backward_multiplier=backward_mult,
        )
        if t_collation <= 0 and t_gpu <= 0:
            continue
        rows.append({
            **shared,
            "fraction": frac,
            "target_nodes": target,
            "n_graphs": n_graphs,
            "t_collation_ms": round(t_collation * 1000, 2),
            "t_gpu_ms": round(t_gpu * 1000, 2),
        })

    del model
    torch.cuda.empty_cache()

    return rows


def _format_table(results: list[dict]) -> str:
    """Format results as a readable multi-point calibration table."""
    if not results:
        return "No results."

    header = (
        f"{'model':<12} {'scale':<6} {'dataset':<10} {'frac':>5} "
        f"{'target':>10} {'graphs':>7} "
        f"{'T_c(ms)':>9} {'T_g(ms)':>9} "
        f"{'B/node':>8} {'bwd':>5} {'budget':>10}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['model_type']:<12} {r['scale']:<6} {r['dataset']:<10} "
            f"{r['fraction']:>5.2f} "
            f"{r['target_nodes']:>10,} {r['n_graphs']:>7,} "
            f"{r['t_collation_ms']:>9.2f} {r['t_gpu_ms']:>9.2f} "
            f"{r['bytes_per_node']:>8,} {r['backward_multiplier']:>5.2f} "
            f"{r['budget']:>10,}"
        )

    return "\n".join(lines)


def main(argv: list[str]) -> None:
    """CLI entry point -- called from __main__.py."""
    parser = argparse.ArgumentParser(
        description="Profile sizing chain (VRAM probe + calibration) on GPU"
    )
    parser.add_argument(
        "--dataset", nargs="*", default=None,
        help="Dataset(s) to probe (default: all with caches)",
    )
    parser.add_argument(
        "--model-type", nargs="*", default=None,
        help=f"Model type(s) to probe (default: all = {sorted(VALID_MODEL_TYPES)})",
    )
    parser.add_argument(
        "--scale", nargs="*", default=None,
        help=f"Scale(s) to probe (default: all = {sorted(VALID_SCALES)})",
    )
    parser.add_argument(
        "--lake-root", default=LAKE_ROOT,
        help=f"Lake root path (default: {LAKE_ROOT})",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results but don't write to data lake",
    )
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("ERROR: probe-budget requires a GPU. Submit via: scripts/submit.sh profile-budget",
              file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")

    model_types = args.model_type or sorted(VALID_MODEL_TYPES)
    scales = args.scale or sorted(VALID_SCALES)
    datasets = args.dataset or _find_cached_datasets(args.lake_root)

    if not datasets:
        print(f"ERROR: no cached datasets found under {args.lake_root}", file=sys.stderr)
        sys.exit(1)

    total = len(model_types) * len(scales) * len(datasets)
    log.info("profile_budget_start", combos=total,
             models=model_types, scales=scales, datasets=datasets)

    results = []
    errors = []

    for model_type in model_types:
        for scale in scales:
            for dataset_name in datasets:
                label = f"{model_type}/{scale}/{dataset_name}"
                try:
                    rows = _probe_combo(model_type, scale, dataset_name, args.lake_root, device)
                    results.extend(rows)
                    log.info("probe_done", label=label, points=len(rows))
                except Exception as e:
                    log.error("probe_failed", label=label, error=str(e))
                    errors.append({"label": label, "error": str(e)})

    # --- Write to data lake ---
    out_path = None
    if results and not args.dry_run:
        import csv
        import os

        from graphids.config import require_lake_write

        require_lake_write()
        out_dir = Path(args.lake_root) / "reference"
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
    if args.json:
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
