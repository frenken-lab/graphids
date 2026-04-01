"""Layer 0: Pure Python tests for orchestration helpers.

All tests run on the login node — no SLURM, no dagster, no torch.
Tests cover: config path helpers, identity hashing, asset enumeration,
CLI value formatting, SLURM script generation, resource specs, and
cluster-agnostic resource resolution.
"""

from __future__ import annotations

from unittest import mock

import pytest

from graphids.config import PROJECT_ROOT, compute_identity_hash, run_dir
from graphids.orchestrate.component import (
    StageConfig,
    _cli_val,
    _identity_value,
    build_cli_args,
    enumerate_assets,
)
from graphids.orchestrate.resources import (
    ResourceSpec,
    _detect_cluster,
    get_resources,
    scale_resources,
)
from graphids.orchestrate.slurm import generate_script


# ---------------------------------------------------------------------------
# run_dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args, expected", [
    (("/lake", "alice", "set_01", "vgae", "small", "autoencoder", "_abc12345", "", 42),
     "/lake/dev/alice/set_01/vgae_small_autoencoder_abc12345/seed_42"),
    (("/lake", "alice", "set_01", "gat", "large", "curriculum", "_def67890", "_kd", 0),
     "/lake/dev/alice/set_01/gat_large_curriculum_def67890_kd/seed_0"),
    (("/lake", "bob", "hcrl_sa", "preprocess", "small", "preprocess", "", "", 42),
     "/lake/dev/bob/hcrl_sa/preprocess_small_preprocess/seed_42"),
], ids=["basic", "kd_tag", "empty_identity"])
def test_run_dir(args, expected):
    assert run_dir(*args) == expected


# ---------------------------------------------------------------------------
# compute_identity_hash
# ---------------------------------------------------------------------------


def test_identity_hash_deterministic():
    cfg = {"scale": "small", "conv_type": "gatv2", "variational": True}
    h1 = compute_identity_hash("autoencoder", cfg)
    h2 = compute_identity_hash("autoencoder", cfg)
    assert h1 == h2
    assert h1.startswith("_")
    assert len(h1) == 9  # "_" + 8 hex chars


def test_identity_hash_differs_on_value_change():
    cfg_a = {"scale": "small", "conv_type": "gatv2", "variational": True}
    cfg_b = {"scale": "large", "conv_type": "gatv2", "variational": True}
    assert compute_identity_hash("autoencoder", cfg_a) != compute_identity_hash("autoencoder", cfg_b)


def test_identity_hash_empty_for_no_keys():
    assert compute_identity_hash("preprocess", {}) == ""


def test_identity_hash_raises_on_missing_key():
    with pytest.raises(KeyError, match="Identity keys.*not found"):
        compute_identity_hash("autoencoder", {"scale": "small"})


# ---------------------------------------------------------------------------
# _cli_val
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (True, "true"),
    (False, "false"),
    ("gatv2", "gatv2"),
    (42, "42"),
    (0.001, "0.001"),
], ids=["bool_true", "bool_false", "str", "int", "float"])
def test_cli_val(value, expected):
    assert _cli_val(value) == expected


# ---------------------------------------------------------------------------
# build_cli_args
# ---------------------------------------------------------------------------


def test_build_cli_args_base_args():
    cfg = StageConfig(
        asset_name="ae_abc", stage="autoencoder", model_type="vgae",
        scale="small", config_files=("stages/ae.yaml",), identity="_abc",
    )
    args = build_cli_args(cfg, "set_01", 42, "/lake/run")

    assert args == [
        "--data.init_args.dataset=set_01",
        "--seed_everything=42",
        "--trainer.default_root_dir=/lake/run",
    ]


def test_build_cli_args_model_overrides():
    cfg = StageConfig(
        asset_name="cur_xyz", stage="curriculum", model_type="gat",
        scale="small", config_files=("stages/cur.yaml",), identity="_xyz",
        model_overrides={"conv_type": "gatv1", "loss_fn": "ce"},
    )
    args = build_cli_args(cfg, "set_01", 42, "/lake/run")

    assert "--model.init_args.conv_type=gatv1" in args
    assert "--model.init_args.loss_fn=ce" in args


def test_build_cli_args_upstream_ckpts():
    cfg = StageConfig(
        asset_name="fus_xyz", stage="fusion", model_type="dqn",
        scale="small", config_files=("stages/fus.yaml",), identity="_xyz",
        upstream_asset_names=("ae_abc", "cur_def"),
        upstream_ckpt_flags={
            "ae_abc": "--data.init_args.vgae_ckpt_path",
            "cur_def": "--data.init_args.gat_ckpt_path",
        },
    )
    args = build_cli_args(cfg, "set_01", 42, "/lake/run", upstream_ckpts={
        "ae_abc": "/lake/ae/best.ckpt",
        "cur_def": "/lake/cur/best.ckpt",
    })

    assert "--data.init_args.vgae_ckpt_path=/lake/ae/best.ckpt" in args
    assert "--data.init_args.gat_ckpt_path=/lake/cur/best.ckpt" in args


def test_build_cli_args_unknown_upstream_ignored():
    """Upstream ckpt with no matching flag is silently skipped."""
    cfg = StageConfig(
        asset_name="fus_xyz", stage="fusion", model_type="dqn",
        scale="small", config_files=("stages/fus.yaml",), identity="_xyz",
        upstream_ckpt_flags={},
    )
    args = build_cli_args(cfg, "set_01", 42, "/lake/run",
                          upstream_ckpts={"some_asset": "/path/ckpt"})

    assert not any("ckpt_path" in a for a in args)


def test_build_cli_args_no_upstream_produces_base_only():
    cfg = StageConfig(
        asset_name="ae_abc", stage="autoencoder", model_type="vgae",
        scale="small", config_files=("stages/ae.yaml",), identity="_abc",
    )
    args = build_cli_args(cfg, "set_02", 0, "/lake/run")
    assert len(args) == 3


# ---------------------------------------------------------------------------
# _identity_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key, merged, stages, expected", [
    ("gat_stage", {}, ["autoencoder", "curriculum", "fusion"], "curriculum"),
    ("gat_stage", {}, ["autoencoder", "normal", "fusion"], "normal"),
    ("method", {"fusion_method": "dqn"}, ["fusion"], "dqn"),
    ("conv_type", {"conv_type": "gatv2"}, ["autoencoder"], "gatv2"),
    ("nonexistent", {}, [], None),
], ids=["gat_stage_curriculum", "gat_stage_normal", "recipe_rename", "direct", "missing"])
def test_identity_value(key, merged, stages, expected):
    assert _identity_value(key, merged, stages) == expected


# ---------------------------------------------------------------------------
# generate_script
# ---------------------------------------------------------------------------


GPU_SPEC = ResourceSpec(
    partition="gpu", time="02:00:00", mem="36G",
    cpus_per_task=4, num_workers=3, gres="gpu:1",
)


def test_generate_script_structure():
    script = generate_script(["graphids/config/stages/autoencoder.yaml"], GPU_SPEC)

    assert script.startswith("#!/bin/bash\n")
    assert f"source {PROJECT_ROOT}/scripts/slurm/_preamble.sh" in script
    assert "python -m graphids fit" in script
    assert "--config" in script
    assert "autoencoder.yaml" in script
    assert f"source {PROJECT_ROOT}/scripts/slurm/_epilog.sh" in script


def test_generate_script_includes_existing_ckpt(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()

    script = generate_script(["stages/normal.yaml"], GPU_SPEC, ckpt_path=ckpt)
    assert f"--ckpt_path {ckpt}" in script


def test_generate_script_skips_missing_ckpt(tmp_path):
    script = generate_script(
        ["stages/normal.yaml"], GPU_SPEC,
        ckpt_path=tmp_path / "nonexistent.ckpt",
    )
    assert "--ckpt_path" not in script


def test_generate_script_appends_overrides():
    script = generate_script(
        ["stages/normal.yaml"], GPU_SPEC,
        cli_overrides=["--model.init_args.conv_type=gatv1", "--seed_everything=0"],
    )
    assert "--model.init_args.conv_type=gatv1" in script
    assert "--seed_everything=0" in script


def test_generate_script_multi_config():
    script = generate_script(
        ["stages/autoencoder.yaml", "models/vgae/small.yaml"], GPU_SPEC,
    )
    assert script.count("--config") == 2


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mem, expected_mb", [
    ("36G", 36 * 1024),
    ("512M", 512),
], ids=["gigabytes", "megabytes"])
def test_resource_spec_mem_mb(mem, expected_mb):
    spec = ResourceSpec(partition="gpu", time="01:00:00", mem=mem,
                        cpus_per_task=4, num_workers=3)
    assert spec.mem_mb == expected_mb


def test_resource_spec_time_minutes():
    spec = ResourceSpec(partition="gpu", time="04:30:00", mem="36G",
                        cpus_per_task=4, num_workers=3)
    assert spec.time_minutes == 270


# ---------------------------------------------------------------------------
# _detect_cluster
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env, hostname, expected", [
    ({"KD_GAT_CLUSTER": "cardinal"}, "anything", "cardinal"),
    ({}, "pitzer-login02.hpc.osc.edu", "pitzer"),
    ({}, "ascend-login01.hpc.osc.edu", "ascend"),
    ({}, "unknown-host.example.com", "pitzer"),
], ids=["env_override", "pitzer_hostname", "ascend_hostname", "unknown_defaults_pitzer"])
def test_detect_cluster(env, hostname, expected):
    with mock.patch.dict("os.environ", env, clear=("KD_GAT_CLUSTER" not in env)):
        with mock.patch("socket.gethostname", return_value=hostname):
            assert _detect_cluster() == expected


# ---------------------------------------------------------------------------
# get_resources / scale_resources
# ---------------------------------------------------------------------------


def test_get_resources_known_profile():
    spec = get_resources("vgae", "small", "autoencoder")
    assert spec.partition
    assert spec.mem
    assert spec.cpus_per_task > 0


def test_get_resources_raises_on_missing_profile():
    with pytest.raises(KeyError, match="No resource profile"):
        get_resources("nonexistent_model", "small", "autoencoder")


@pytest.mark.parametrize("reason, check", [
    ("OUT_OF_MEMORY", lambda orig, scaled: scaled.mem_mb > orig.mem_mb),
    ("TIMEOUT", lambda orig, scaled: scaled.time_minutes > orig.time_minutes),
    ("SOMETHING_ELSE", lambda orig, scaled: scaled.mem == orig.mem and scaled.time == orig.time),
], ids=["oom_scales_mem", "timeout_scales_time", "unknown_is_noop"])
def test_scale_resources(reason, check):
    spec = ResourceSpec(partition="gpu", time="02:00:00", mem="36G",
                        cpus_per_task=4, num_workers=3, gres="gpu:1")
    scaled = scale_resources(spec, reason)
    assert check(spec, scaled)


# ---------------------------------------------------------------------------
# enumerate_assets
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_pipeline():
    """Minimal pipeline topology for testing enumerate_assets."""
    return {
        "stages": {
            "autoencoder": {
                "learning_type": "unsupervised", "model": "vgae",
                "mode": "gpu_train", "depends_on": [],
                "identity_keys": ["scale", "conv_type", "variational"],
            },
            "curriculum": {
                "learning_type": "supervised", "model": "gat",
                "mode": "gpu_train",
                "depends_on": [{"model": "vgae", "stage": "autoencoder"}],
                "identity_keys": ["scale", "conv_type", "loss_fn", "variational"],
            },
            "fusion": {
                "learning_type": "rl_fusion", "model": "dqn",
                "mode": "gpu_train",
                "depends_on": [
                    {"model": "vgae", "stage": "autoencoder"},
                    {"model": "gat", "stage": "curriculum"},
                    {"model": "gat", "stage": "normal"},
                ],
                "identity_keys": ["scale", "gat_stage", "loss_fn", "method",
                                  "conv_type", "variational"],
                "model_keys": [],
            },
        },
    }


@pytest.fixture()
def mini_recipe():
    """Two configs that share an autoencoder (tests dedup)."""
    return {
        "defaults": {
            "stages": ["autoencoder", "curriculum", "fusion"],
            "scale": "small", "conv_type": "gatv2", "loss_fn": "focal",
            "fusion_method": "bandit", "variational": True,
        },
        "configs": {
            "focal_curriculum": {"fusion_method": "weighted_avg"},
            "ce_curriculum": {"loss_fn": "ce", "fusion_method": "weighted_avg"},
        },
    }


def test_enumerate_deduplicates_shared_autoencoder(mini_pipeline, mini_recipe):
    assets = enumerate_assets(mini_pipeline, mini_recipe)
    stages = [a.stage for a in assets]

    assert stages.count("autoencoder") == 1  # deduped
    assert stages.count("curriculum") == 2
    assert stages.count("fusion") == 2


def test_enumerate_fusion_refs_shared_autoencoder(mini_pipeline, mini_recipe):
    assets = enumerate_assets(mini_pipeline, mini_recipe)
    ae_name = next(a.asset_name for a in assets if a.stage == "autoencoder")

    for fa in (a for a in assets if a.stage == "fusion"):
        assert ae_name in fa.upstream_asset_names


def test_enumerate_identity_hashes_present(mini_pipeline, mini_recipe):
    for a in enumerate_assets(mini_pipeline, mini_recipe):
        assert a.identity.startswith("_"), f"{a.asset_name} missing identity hash"
        assert len(a.identity) == 9


def test_enumerate_upstream_ckpt_flags_set(mini_pipeline, mini_recipe):
    for ca in (a for a in enumerate_assets(mini_pipeline, mini_recipe)
               if a.stage == "curriculum"):
        assert len(ca.upstream_asset_names) >= 1
        for up_name in ca.upstream_asset_names:
            if up_name in ca.upstream_ckpt_flags:
                assert "vgae_ckpt_path" in ca.upstream_ckpt_flags[up_name]


def test_enumerate_different_fusion_methods_differ(mini_pipeline):
    recipe = {
        "defaults": {
            "stages": ["autoencoder", "curriculum", "fusion"],
            "scale": "small", "conv_type": "gatv2", "loss_fn": "focal",
            "fusion_method": "bandit", "variational": True,
        },
        "configs": {"bandit": {}, "dqn": {"fusion_method": "dqn"}},
    }
    fusion_ids = {a.identity for a in enumerate_assets(mini_pipeline, recipe)
                  if a.stage == "fusion"}
    assert len(fusion_ids) == 2


def test_enumerate_normal_vs_curriculum_gat_stage_differ(mini_pipeline):
    mini_pipeline["stages"]["normal"] = {
        "learning_type": "supervised", "model": "gat", "mode": "gpu_train",
        "depends_on": [], "identity_keys": ["scale", "conv_type", "loss_fn"],
    }
    recipe = {
        "defaults": {
            "scale": "small", "conv_type": "gatv2", "loss_fn": "focal",
            "fusion_method": "bandit", "variational": True,
        },
        "configs": {
            "with_curriculum": {"stages": ["autoencoder", "curriculum", "fusion"]},
            "with_normal": {"stages": ["autoencoder", "normal", "fusion"]},
        },
    }
    assets = enumerate_assets(mini_pipeline, recipe)
    fusion_ids = {a.identity for a in assets if a.stage == "fusion"}
    assert len(fusion_ids) == 2

    # Normal-only fusion must have GAT checkpoint wired (not silently empty)
    normal_fusion = next(a for a in assets if a.stage == "fusion"
                         and any("normal" in n for n in a.upstream_asset_names))
    assert any("gat_ckpt_path" in f for f in normal_fusion.upstream_ckpt_flags.values())


def test_enumerate_config_files_populated(mini_pipeline, mini_recipe):
    for a in enumerate_assets(mini_pipeline, mini_recipe):
        assert len(a.config_files) >= 1
        assert a.stage in a.config_files[0]


def test_enumerate_stage_config_types(mini_pipeline, mini_recipe):
    for a in enumerate_assets(mini_pipeline, mini_recipe):
        assert a.asset_name and a.stage and a.model_type
        assert a.scale == "small"
        assert isinstance(a.config_files, tuple)
        assert isinstance(a.upstream_asset_names, tuple)
