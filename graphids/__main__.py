"""CLI entry point: python -m graphids <subcommand>

Subcommands:
  fit|test|validate|predict  — LightningCLI training/evaluation
  analyze                    — generate analysis artifacts from checkpoints
  analyze-from-spec          — run analyzer from canonical AnalysisSpec
  pipeline-status            — aggregated dagster + SLURM phase status
  job-stats                  — sacct resource profiler
  probe-budget               — hardware cost model measurement
  profile                    — profiled training run (PyTorchProfiler)
  train-from-spec            — run training from canonical TrainingSpec
  rebuild-caches             — rebuild preprocessed graph caches
  stage-data                 — stage data from NFS to scratch/TMPDIR
  submit-profile             — print SLURM resource profile for submit.sh
  test-preprocessing         — validate preprocessing pipeline

Dagster (separate entry point):
  python -m graphids.orchestrate validate  — validate config chains
  dg launch --assets ...                   — materialize assets
"""

from __future__ import annotations

import argparse
import importlib
import sys

from graphids.log import configure_logging

configure_logging()

_LIGHTNING_COMMANDS = ("fit", "test", "validate", "predict")
# Values are either "module" (calls module.main) or "module:function".
_COMMAND_MODULES: dict[str, str] = {
    "analyze": "graphids.commands.analyze",
    "analyze-from-spec": "graphids.commands.from_spec:main_analyze",
    "pipeline-status": "graphids.commands.pipeline_status",
    "job-stats": "graphids.commands.job_stats",
    "probe-budget": "graphids.commands.probe_budget",
    "profile": "graphids.commands.profile",
    "rebuild-caches": "graphids.commands.rebuild_caches",
    "stage-data": "graphids.commands.stage_data",
    "submit-profile": "graphids.commands.submit_profile",
    "extract-fusion-states": "graphids.commands.extract_fusion_states",
    "test-from-spec": "graphids.commands.from_spec:main_test",
    "test-preprocessing": "graphids.commands.test_preprocessing",
    "train-from-spec": "graphids.commands.from_spec:main_train",
    "_finalize-record": "graphids.commands.finalize_record",
    "rebuild-catalog": "graphids.commands.rebuild_catalog",
}


def _run_lightning(command: str, argv: list[str]) -> None:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    from graphids.cli import run_lightning

    run_lightning([command, *argv])


def _run_module(target: str, argv: list[str]) -> None:
    module_name, _, func_name = target.partition(":")
    func_name = func_name or "main"
    try:
        mod = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(f"Failed to load command module '{module_name}': {exc}") from exc
    func = getattr(mod, func_name, None)
    if func is None:
        raise SystemExit(f"Module '{module_name}' does not expose '{func_name}(argv)'")
    func(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m graphids")
    subs = parser.add_subparsers(dest="command")

    for cmd in _LIGHTNING_COMMANDS:
        p = subs.add_parser(cmd, add_help=False)
        p.set_defaults(kind="lightning", command_name=cmd)

    for cmd, module in _COMMAND_MODULES.items():
        p = subs.add_parser(cmd, add_help=False)
        p.set_defaults(kind="module", module_name=module)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    # Preserve legacy behavior: no subcommand defaults to LightningCLI.
    if not args:
        _run_lightning("fit", [])
        return

    parser = _build_parser()
    ns, remaining = parser.parse_known_args(args)

    if ns.command is None:
        parser.print_help()
        raise SystemExit(2)

    if ns.kind == "lightning":
        _run_lightning(ns.command_name, remaining)
        return

    _run_module(ns.module_name, remaining)


if __name__ == "__main__":
    main()
