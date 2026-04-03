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


def _probe_combo(
    model_type: str, scale: str, dataset_name: str, lake_root: str, device,
) -> dict:
    """Run full sizing chain for one (model_type, scale, dataset) combo."""
    import torch

    from graphids.config import data_dir
    from graphids.core.preprocessing.budget import (
        _probe_vram, calibrate_at_budget, compute_resource_profile, node_budget,
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

    # --- Calibrate at operating batch size ---
    t_collation, t_gpu = calibrate_at_budget(
        model, ds, result.budget, backward_multiplier=backward_mult,
    )

    # --- Resource profile ---
    profile = compute_resource_profile(
        result, t_collation_s=t_collation, t_gpu_s=t_gpu,
    )

    del model
    torch.cuda.empty_cache()

    row = {
        "model_type": model_type,
        "scale": scale,
        "dataset": dataset_name,
        "bytes_per_node": bytes_per_node,
        "backward_multiplier": round(backward_mult, 2),
        "budget": result.budget,
        "mem_budget": result.mem_budget,
        "mean_nodes": round(result.mean_nodes, 1),
    }
    if profile is not None:
        row.update({
            "t_collation_ms": round(t_collation * 1000, 1),
            "t_gpu_ms": round(t_gpu * 1000, 1),
            "workers": profile.workers,
            "prefetch_factor": profile.prefetch_factor,
            "cpus": profile.cpus,
            "memory_gb": profile.memory_gb,
            "cg_ratio": round(profile.t_collation_us / profile.t_gpu_us, 2)
                        if profile.t_gpu_us > 0 else None,
        })
    return row


def _format_table(results: list[dict]) -> str:
    """Format results as a readable sizing chain table."""
    if not results:
        return "No results."

    header = (
        f"{'model':<12} {'scale':<6} {'dataset':<10} "
        f"{'B/node':>8} {'bwd':>5} {'budget':>10} "
        f"{'T_c(ms)':>8} {'T_g(ms)':>8} {'W':>3} {'pf':>3} "
        f"{'cpus':>5} {'mem_gb':>6} {'cg':>5}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['model_type']:<12} {r['scale']:<6} {r['dataset']:<10} "
            f"{r['bytes_per_node']:>8,} {r['backward_multiplier']:>5.2f} "
            f"{r['budget']:>10,} "
            f"{r.get('t_collation_ms', ''):>8} "
            f"{r.get('t_gpu_ms', ''):>8} "
            f"{r.get('workers', ''):>3} "
            f"{r.get('prefetch_factor', ''):>3} "
            f"{r.get('cpus', ''):>5} "
            f"{r.get('memory_gb', ''):>6} "
            f"{r.get('cg_ratio', ''):>5}"
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
                    result = _probe_combo(model_type, scale, dataset_name, args.lake_root, device)
                    results.append(result)
                    log.info("probe_done", label=label, **{
                        k: v for k, v in result.items()
                        if k not in ("model_type", "scale", "dataset")
                    })
                except Exception as e:
                    log.error("probe_failed", label=label, error=str(e))
                    errors.append({"label": label, "error": str(e)})

    if args.json:
        output = {"results": results, "errors": errors}
        print(json.dumps(output, indent=2))
    else:
        print(_format_table(results))
        if errors:
            print(f"\n{len(errors)} probe(s) failed:")
            for e in errors:
                print(f"  {e['label']}: {e['error']}")
