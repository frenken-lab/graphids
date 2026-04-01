"""Assert naive merge_yaml_chain matches jsonargparse for representative config chains.

Marked slurm because GraphIDSCLI imports torch at instantiation time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from graphids.cli import resolve_configs
from graphids.config.yaml_utils import write_yaml
from graphids.core.contracts import TrainingContract, TrainingSpec

# (label, stage, model_family, scale, fusion_method)
_CHAINS = [
    ("autoencoder_small", "autoencoder", "vgae", "small", None),
    ("normal_small", "normal", "gat", "small", None),
    ("fusion_bandit", "fusion", None, "small", "bandit"),
]


def _build_spec(stage, model_family, scale, fusion_method, run_dir):
    return TrainingSpec(
        stage=stage,
        model_family=model_family or "",
        scale=scale,
        dataset="hcrl_ch",
        seed=42,
        run_dir=run_dir,
        config_files=TrainingContract.resolve_config_files(
            stage, scale, model_family=model_family, fusion_method=fusion_method,
        ),
    )


def _parse_via_jsonargparse(config_files, overrides, snapshot_path):
    """Build a fresh GraphIDSCLI per chain so the parser matches the model class."""
    from graphids._lightning import CLI_KWARGS, GraphIDSCLI

    resolved = resolve_configs(config_files, overrides)
    write_yaml(resolved, snapshot_path)

    # Build CLI args matching this chain (same files + overrides the real path uses)
    args = ["fit", "--config", str(snapshot_path)]

    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        cli = GraphIDSCLI(
            **{**CLI_KWARGS, "run": False, "auto_configure_optimizers": False},
            args=args,
        )
        parsed = yaml.safe_load(
            cli.parser.dump(cli.config["fit"], skip_link_targets=False, skip_none=False)
        )
    finally:
        sys.argv = saved

    return resolved, parsed


@pytest.mark.slurm
@pytest.mark.parametrize("label,stage,model_family,scale,fusion_method", _CHAINS)
def test_naive_merge_matches_jsonargparse(
    label, stage, model_family, scale, fusion_method, tmp_path,
):
    spec = _build_spec(stage, model_family, scale, fusion_method, str(tmp_path))
    overrides = TrainingContract.to_override_dict(spec)

    naive, jp = _parse_via_jsonargparse(
        spec.config_files, overrides, tmp_path / f"{label}.yaml",
    )

    for ns in ("model", "data"):
        naive_args = naive.get(ns, {}).get("init_args", {})
        jp_args = jp.get(ns, {}).get("init_args", {})
        for key, naive_val in naive_args.items():
            assert key in jp_args, (
                f"[{label}] {ns}.init_args.{key} missing from jsonargparse output"
            )
            # String comparison normalizes type coercion differences (int vs str)
            assert str(jp_args[key]) == str(naive_val), (
                f"[{label}] {ns}.init_args.{key}: naive={naive_val!r} jp={jp_args[key]!r}"
            )
