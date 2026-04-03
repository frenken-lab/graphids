"""Profile the budget module's cost model on real hardware.

Instantiates each (model_type, scale) with random weights via jsonargparse
(same path as LightningCLI), loads cached datasets, and runs _probe() to
measure bytes_per_node, γ, α, β.

Usage (via __main__.py):
    python -m graphids profile-budget
    python -m graphids profile-budget --dataset hcrl_ch --model-type vgae
    python -m graphids profile-budget --json

SLURM:
    scripts/submit.sh profile-budget
"""

from __future__ import annotations

import argparse
import csv
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
    """Instantiate a model by reading config YAMLs and importing the class directly.

    Avoids jsonargparse subclass resolution — the second --config (scale YAML)
    replaces the model dict wholesale, losing class_path from the base YAML.
    """
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
    """Run _probe for one (model_type, scale, dataset) combo. Returns result dict."""
    import torch

    from graphids.config import data_dir
    from graphids.core.preprocessing.budget import _probe
    from graphids.core.preprocessing.datasets.can_bus import CANBusDataset

    # --- Load dataset from cache ---
    root = cache_dir(lake_root, dataset_name)
    metadata_path = root / "cache_metadata.json"
    stats = json.loads(metadata_path.read_text())["graph_stats"]["node_count"]
    mean_nodes = stats["mean"]

    raw = data_dir(lake_root, dataset_name)
    ds = CANBusDataset(root=root, raw_dir=raw, split="train")

    # Get dataset properties for model instantiation
    sample = ds[0]
    in_channels = sample.x.shape[1]
    num_ids = ds.num_arb_ids

    # --- Instantiate model via jsonargparse (same as LightningCLI) ---
    model = _instantiate_model(model_type, scale, num_ids, in_channels)
    model = model.to(device)
    model.eval()

    # --- Run probe ---
    step_fn = getattr(model, "_step", None)
    bytes_per_node, backward_mult, gamma, alpha, beta = (
        _probe(model, ds, step_fn=step_fn)
    )

    # --- Free GPU memory ---
    del model
    torch.cuda.empty_cache()

    return {
        "model_type": model_type,
        "scale": scale,
        "dataset": dataset_name,
        "bytes_per_node": bytes_per_node,
        "backward_multiplier": round(backward_mult, 2),
        "gamma_us": round(gamma * 1e6, 1),
        "alpha_ms": round(alpha * 1000, 3),
        "beta_us": round(beta * 1e6, 3),
        "mean_nodes": round(mean_nodes, 1),
        "num_ids": num_ids,
        "in_channels": in_channels,
    }


def _format_table(results: list[dict]) -> str:
    """Format results as a readable table."""
    if not results:
        return "No results."

    header = (
        f"{'model_type':<12} {'scale':<6} {'dataset':<10} "
        f"{'B/node':>10} {'bwd_mult':>8} "
        f"{'γ(μs)':>8} {'α(ms)':>8} {'β(μs)':>8} "
        f"{'mean_nodes':>10}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['model_type']:<12} {r['scale']:<6} {r['dataset']:<10} "
            f"{r['bytes_per_node']:>10,} "
            f"{r['backward_multiplier']:>8.2f} "
            f"{r['gamma_us']:>8.1f} {r['alpha_ms']:>8.3f} "
            f"{r['beta_us']:>8.3f} {r['mean_nodes']:>10.1f}"
        )

    # Summary: is α > 0 anywhere?
    alphas = [r["alpha_ms"] for r in results]
    lines.append("")
    if max(alphas) < 0.1:
        lines.append("** α ~ 0 for all models -> throughput ceiling never exists.")
        lines.append("   Budget reduces to: VRAM / bytes_per_node. Consider deleting throughput code.")
    else:
        nonzero = [r for r in results if r["alpha_ms"] >= 0.1]
        lines.append(f"** α > 0 for {len(nonzero)}/{len(results)} combos -> throughput ceiling is real.")

    return "\n".join(lines)


_MATRIX_FIELDS = [
    "dataset", "model_type", "scale", "gpu", "num_workers",
    "budget", "mean_nodes", "graphs_per_batch",
    "mem_budget", "throughput_floor", "binding", "cg_ratio",
]

_WORKER_COUNTS = [2, 4, 6, 8]


def _load_gpu_vram() -> dict[str, int]:
    """Read GPU VRAM specs from clusters.yaml → {name: free_bytes}."""
    clusters = read_yaml(CONFIG_DIR / "resources" / "clusters.yaml")
    return {
        name: int(spec["free_gb"] * 1024**3)
        for name, spec in clusters["clusters"]["gpu_vram"].items()
    }


def _write_matrix(probe_results: list[dict], lake_root: str) -> Path:
    """Compute node_budget across GPU types × worker counts and write CSV.

    Uses measured probe values (bytes_per_node, γ, α, β) from the current run
    and varies free VRAM and worker count to produce the full budget profile.
    """
    from unittest.mock import patch

    from graphids.core.preprocessing.budget import node_budget

    gpu_vram = _load_gpu_vram()
    rows: list[dict] = []

    for probe in probe_results:
        bpn = probe["bytes_per_node"]
        bwd_mult = probe["backward_multiplier"]
        gamma = probe["gamma_us"] * 1e-6   # back to seconds
        alpha = probe["alpha_ms"] * 1e-3
        beta = probe["beta_us"] * 1e-6
        ds = probe["dataset"]
        mean_nodes = probe["mean_nodes"]

        # Write temporary cache_metadata.json for node_budget
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "cache_metadata.json"
            metadata.write_text(json.dumps(
                {"graph_stats": {"node_count": {"mean": mean_nodes}}}
            ))

            def mock_probe(model, dataset, step_fn=None,
                           _bpn=bpn, _bwd=bwd_mult,
                           _g=gamma, _a=alpha, _b=beta):
                return _bpn, _bwd, _g, _a, _b

            for gpu_name, free_bytes in gpu_vram.items():
                for nw in _WORKER_COUNTS:
                    with (
                        patch("graphids.core.preprocessing.budget.cache_dir",
                              return_value=Path(tmp)),
                        patch("graphids.core.preprocessing.budget._probe", mock_probe),
                        patch("torch.cuda.is_available", return_value=True),
                        patch("torch.cuda.mem_get_info",
                              return_value=(free_bytes, free_bytes)),
                    ):
                        result = node_budget(
                            ds, tmp, conv_type="gatv2",
                            model=True, train_dataset=True, num_workers=nw,
                        )
                    rows.append({
                        "dataset": ds,
                        "model_type": probe["model_type"],
                        "scale": probe["scale"],
                        "gpu": gpu_name,
                        "num_workers": nw,
                        "budget": result.budget,
                        "mean_nodes": result.mean_nodes,
                        "graphs_per_batch": round(result.budget / result.mean_nodes, 1),
                        "mem_budget": result.mem_budget,
                        "throughput_floor": (result.throughput_floor
                                            if result.throughput_floor is not None else ""),
                        "binding": result.binding,
                        "cg_ratio": (round(result.cg_ratio, 3)
                                     if result.cg_ratio is not None else ""),
                    })

    out_dir = Path(lake_root) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "budget_matrix.csv"
    rows.sort(key=lambda r: (r["dataset"], r["model_type"], r["scale"],
                             r["gpu"], r["num_workers"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MATRIX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv: list[str]) -> None:
    """CLI entry point -- called from __main__.py."""
    parser = argparse.ArgumentParser(
        description="Profile budget cost model (bytes_per_node, gamma, alpha, beta) on GPU"
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
        "--matrix", action="store_true",
        help="After probing, compute budget across all GPU types × worker counts "
             "and write {lake_root}/reference/budget_matrix.csv",
    )
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("ERROR: profile-budget requires a GPU. Submit via: scripts/submit.sh profile-budget",
              file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")

    # Resolve combos
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
                    log.info("probe_done", label=label,
                             bytes_per_node=result["bytes_per_node"],
                             alpha_ms=result["alpha_ms"],
                             beta_us=result["beta_us"])
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

    if args.matrix and results:
        csv_path = _write_matrix(results, args.lake_root)
        n_gpus = len(_load_gpu_vram())
        n_rows = len(results) * n_gpus * len(_WORKER_COUNTS)
        log.info("budget_matrix_written", path=str(csv_path), rows=n_rows)
        print(f"\nBudget matrix: {csv_path} ({n_rows} rows)")
