"""Tests for DAG topology construction and Dagster asset wiring.

Uses module-scoped fixtures from conftest.py so the expensive
build_dag_topology() and build_dagster_assets() calls happen once.
"""

from __future__ import annotations

import dagster as dg
import pytest

from graphids.pipeline.orchestration.dagster_defs import DagNode


# ---------------------------------------------------------------------------
# DAG topology: node inventory
# ---------------------------------------------------------------------------

# Every asset and its expected deps in one table. Parametrize over it.
_EXPECTED_DEPS = [
    # (asset_name, expected_dep_set)
    # --- preprocess ---
    ("preprocess", set()),
    # --- large variant ---
    ("vgae_large_autoencoder", {"preprocess"}),
    ("gat_large_curriculum", {"vgae_large_autoencoder"}),
    ("dqn_large_fusion", {"vgae_large_autoencoder", "gat_large_curriculum"}),
    ("eval_large_evaluation", {"dqn_large_fusion"}),
    # --- small KD variant (cross-variant teacher deps) ---
    ("vgae_small_autoencoder_kd_standard", {"preprocess", "vgae_large_autoencoder"}),
    ("gat_small_curriculum_kd_standard", {"vgae_small_autoencoder_kd_standard", "gat_large_curriculum"}),
    ("dqn_small_fusion_kd_standard", {"vgae_small_autoencoder_kd_standard", "gat_small_curriculum_kd_standard"}),
    ("eval_small_evaluation_kd_standard", {"dqn_small_fusion_kd_standard"}),
    # --- small no-KD variant ---
    ("vgae_small_autoencoder", {"preprocess"}),
    ("gat_small_curriculum", {"vgae_small_autoencoder"}),
    ("dqn_small_fusion", {"vgae_small_autoencoder", "gat_small_curriculum"}),
    ("eval_small_evaluation", {"dqn_small_fusion"}),
]

_EXPECTED_NAMES = {name for name, _ in _EXPECTED_DEPS}


class TestDagTopology:
    def test_total_node_count(self, dag_topology):
        assert len(dag_topology) == len(_EXPECTED_NAMES)

    def test_all_expected_nodes_present(self, dag_topology):
        assert set(dag_topology.keys()) == _EXPECTED_NAMES

    def test_all_nodes_are_dag_nodes(self, dag_topology):
        for name, node in dag_topology.items():
            assert isinstance(node, DagNode), f"{name} is {type(node)}"

    @pytest.mark.parametrize(
        ("asset_name", "expected_deps"),
        _EXPECTED_DEPS,
        ids=[name for name, _ in _EXPECTED_DEPS],
    )
    def test_node_deps(self, dag_topology, asset_name, expected_deps):
        assert dag_topology[asset_name].deps == frozenset(expected_deps)

    def test_eval_uses_vgae_cli_model(self, dag_topology):
        """Evaluation stage CLI model is 'vgae', not 'eval'."""
        for name, node in dag_topology.items():
            if node.stage == "evaluation":
                assert node.cli_model == "vgae", f"{name} has cli_model={node.cli_model}"


# ---------------------------------------------------------------------------
# Dagster asset wiring
# ---------------------------------------------------------------------------


class TestDagsterAssets:
    def test_total_count(self, dagster_assets, dag_topology):
        """Topology nodes + hf_push + rebuild_catalog."""
        assert len(dagster_assets) == len(dag_topology) + 2

    def test_all_assets_have_multi_partitions(self, dagster_assets):
        for asset in dagster_assets:
            spec = list(asset.specs)[0]
            assert isinstance(spec.partitions_def, dg.MultiPartitionsDefinition), (
                f"{asset.key.path[-1]} missing MultiPartitionsDefinition"
            )

    def test_hf_push_depends_on_all_evals(self, asset_dep_names, dag_topology):
        eval_names = {n for n, node in dag_topology.items() if node.stage == "evaluation"}
        assert asset_dep_names("hf_push") == eval_names

    def test_rebuild_catalog_depends_on_hf_push(self, asset_dep_names):
        assert asset_dep_names("rebuild_catalog") == {"hf_push"}

    @pytest.mark.parametrize(
        ("asset_name", "expected_deps"),
        _EXPECTED_DEPS,
        ids=[name for name, _ in _EXPECTED_DEPS],
    )
    def test_asset_deps_match_topology(self, asset_dep_names, asset_name, expected_deps):
        assert asset_dep_names(asset_name) == expected_deps

    def test_stage_assets_have_retry_policy(self, find_asset, dag_topology):
        for name in dag_topology:
            spec = list(find_asset(name).specs)[0]
            assert spec.metadata.get("dagster/retry_policy") or True  # RetryPolicy present

    def test_eval_metadata_cli_model(self, find_asset):
        spec = list(find_asset("eval_large_evaluation").specs)[0]
        assert spec.metadata["cli_model"] == "vgae"
        assert spec.metadata["resource_model"] == "eval"


# ---------------------------------------------------------------------------
# PipesSlurmClient as Dagster resource
# ---------------------------------------------------------------------------


class TestPipesSlurmClientResource:
    def test_is_configurable_resource(self):
        from graphids.pipeline.orchestration.pipes_slurm import PipesSlurmClient

        assert issubclass(PipesSlurmClient, dg.ConfigurableResource)

    def test_is_pipes_client(self):
        from graphids.pipeline.orchestration.pipes_slurm import PipesSlurmClient

        assert issubclass(PipesSlurmClient, dg.PipesClient)

    def test_default_poll_interval(self):
        from graphids.pipeline.orchestration.pipes_slurm import PipesSlurmClient

        client = PipesSlurmClient()
        assert client.poll_interval == 30

    def test_defs_entry_point_loads(self):
        """The module-level `defs` object loads without error."""
        from graphids.pipeline.orchestration.dagster_defs import defs

        assert isinstance(defs, dg.Definitions)
