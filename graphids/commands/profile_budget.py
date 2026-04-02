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
import json
import sys

import structlog

from graphids.config import CONFIG_DIR, LAKE_ROOT, VALID_MODEL_TYPES, VALID_SCALES, cache_dir

log = structlog.get_logger()


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
    """Instantiate a model from config YAMLs via jsonargparse (same path as LightningCLI)."""
    import pytorch_lightning as pl
    from jsonargparse import ArgumentParser

    base = CONFIG_DIR / "models" / model_type / "base.yaml"
    scale_yaml = CONFIG_DIR / "models" / model_type / "scales" / f"{scale}.yaml"

    parser = ArgumentParser()
    parser.add_subclass_arguments(pl.LightningModule, "model")

    cli_args = [
        f"--config={base}",
        f"--config={scale_yaml}",
        f"--model.init_args.num_ids={num_ids}",
        f"--model.init_args.in_channels={in_channels}",
    ]
    # Disable torch.compile for profiling — not all models have this param
    if model_type != "temporal":
        cli_args.append("--model.init_args.compile_model=false")

    cfg = parser.parse_args(cli_args)
    return parser.instantiate_classes(cfg).model


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
    bytes_per_node, gamma, alpha, beta = _probe(model, ds, step_fn=step_fn)

    # --- Free GPU memory ---
    del model
    torch.cuda.empty_cache()

    return {
        "model_type": model_type,
        "scale": scale,
        "dataset": dataset_name,
        "bytes_per_node": bytes_per_node,
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
        f"{'bytes/node':>10} {'γ(μs)':>8} {'α(ms)':>8} {'β(μs)':>8} "
        f"{'mean_nodes':>10} {'num_ids':>8}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['model_type']:<12} {r['scale']:<6} {r['dataset']:<10} "
            f"{r['bytes_per_node']:>10,} {r['gamma_us']:>8.1f} {r['alpha_ms']:>8.3f} "
            f"{r['beta_us']:>8.3f} {r['mean_nodes']:>10.1f} {r['num_ids']:>8}"
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
