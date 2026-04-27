"""Tests for ``graphids/slurm/status.py`` — MLflow query + formatters.

Pure tests — pandas DataFrames built in-process, no MLflow server.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from graphids.slurm.dag import Node
from graphids.slurm.status import (
    NodeStatus,
    format_json,
    format_table,
    query_all,
    query_node_status,
)


def _df(**row) -> pd.DataFrame:
    return pd.DataFrame([row])


_FIT = Node(
    name="vgae", preset_path="unsupervised/vgae.jsonnet", group="unsupervised", variant="vgae"
)
_TEST = Node(
    name="vgae-test",
    preset_path="unsupervised/vgae.jsonnet",
    group="unsupervised",
    variant="vgae",
    action="test",
    deps=("vgae",),
)
_CMD = Node(name="extract-states", command="echo hi", deps=("vgae",))


# CONTRACT: command nodes short-circuit to NA; no MLflow call.
def test_command_node_is_na() -> None:
    with patch("mlflow.search_runs") as m:
        s = query_node_status(_CMD, dataset="ds", seed=42)
    assert s.status == "NA"
    assert m.call_count == 0


# CONTRACT: empty MLflow result → PENDING.
def test_empty_returns_pending() -> None:
    with patch("mlflow.search_runs", return_value=pd.DataFrame()):
        s = query_node_status(_FIT, dataset="ds", seed=42)
    assert s.status == "PENDING"
    assert s.run_id is None


# CONTRACT: non-empty result passes status verbatim.
@pytest.mark.parametrize("mlflow_status", ["FINISHED", "RUNNING", "FAILED", "KILLED"])
def test_passes_status_through(mlflow_status: str) -> None:
    with patch(
        "mlflow.search_runs",
        return_value=_df(status=mlflow_status, run_id="abc", end_time=None),
    ):
        s = query_node_status(_FIT, dataset="ds", seed=42)
    assert s.status == mlflow_status
    assert s.run_id == "abc"


# CONTRACT: test-action nodes query MLflow with phase='test'. Catches a
# regression where action was forgotten and test/fit rows collided.
def test_test_node_queries_with_phase_test() -> None:
    with patch("mlflow.search_runs") as m:
        m.return_value = pd.DataFrame()
        query_node_status(_TEST, dataset="ds", seed=42)
    fs = m.call_args.kwargs["filter_string"]
    assert "graphids.phase` = 'test'" in fs


# DIFFERENTIAL: query_all hits MLflow once per non-command node.
def test_query_all_hits_every_non_command_node() -> None:
    nodes = (_FIT, _TEST, _CMD)
    with patch("mlflow.search_runs", return_value=pd.DataFrame()) as m:
        out = query_all(nodes, dataset="ds", seed=42)
    assert len(out) == 3
    assert m.call_count == 2  # only fit + test query MLflow


# CONTRACT: table contains every node name + status + a summary line.
def test_table_lists_every_node() -> None:
    statuses = [
        NodeStatus(node=_FIT, status="FINISHED", run_id="r1"),
        NodeStatus(node=_TEST, status="PENDING"),
        NodeStatus(node=_CMD, status="NA"),
    ]
    out = format_table(statuses, dataset="ds", seed=42)
    for s in statuses:
        assert s.node.name in out
    assert "Summary:" in out
    assert "ds" in out


def test_json_is_valid_with_expected_schema() -> None:
    statuses = [
        NodeStatus(node=_FIT, status="FINISHED", run_id="r1"),
        NodeStatus(node=_CMD, status="NA"),
    ]
    payload = json.loads(format_json(statuses, dataset="ds", seed=42))
    assert payload["plan"]["dataset"] == "ds"
    assert {n["name"] for n in payload["nodes"]} == {"vgae", "extract-states"}
    assert payload["summary"]["FINISHED"] == 1
    assert payload["summary"]["NA"] == 1
