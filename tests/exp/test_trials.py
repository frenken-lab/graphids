import json
from pathlib import Path

import pytest
import yaml

from graphids.exp import temporal_cuda_graph, temporal_pipeline_ab, trials


def test_trial_writes_standalone_config_and_checks_result(monkeypatch, tmp_path, trial):
    config = {"experiment_name": "probe", "config": {"model": {"scan": {"cuda_graph": True}}}}
    calls = []

    def run(command, *, check):
        calls.append(command)
        assert check
        assert yaml.safe_load(Path(command[3]).read_text()) == config
        output = Path(command[5])
        output.mkdir(parents=True)
        (output / "comparison.json").write_text(json.dumps(trial()))

    monkeypatch.setattr(trials.subprocess, "run", run)
    result = trials.run_trial(tmp_path, "probe", config=config, section_limit=2048, profile=True, reference=trial())
    assert result == trial()
    assert calls[0][1:3] == ["-m", "graphids.exp.temporal_comparison"]
    assert calls[0][-3:] == ["--section-limit", "2048", "--profile"]
    wrong = trial()
    wrong["section_queue_sha256"] = "different"
    with pytest.raises(ValueError, match="section_queue"):
        trials.run_trial(tmp_path, "wrong", config=config, reference=wrong)


def test_graph_pairs_keep_alternating_order_and_equal_work(monkeypatch, tmp_path, trial):
    calls = []

    def run(root, name, *, config, section_limit, reference):
        calls.append((name, config, section_limit))
        assert config["config"]["data"]["num_workers"] == 2
        assert config["config"]["model"]["scan"]["ssm_kernel"] == "torch"
        return trial(110 if config["config"]["model"]["scan"]["cuda_graph"] else 100)

    monkeypatch.setattr(temporal_cuda_graph, "run_trial", run)
    report = temporal_cuda_graph.run_pairs(tmp_path)
    assert [name for name, _, _ in calls] == ["pair1_eager", "pair1_graph", "pair2_graph", "pair2_eager", "pair3_eager", "pair3_graph"]
    assert all(limit == 2048 for _, _, limit in calls)
    assert report["selection"]["accepted"]
    assert len(json.loads((tmp_path / "pairs.json").read_text())["pairs"]) == 3


@pytest.mark.parametrize("fusion,names", [
    (False, ["workers0", "workers2", "workers4", "workers0_repeat"]),
    (True, ["torch", "fused", "torch_repeat", "fused_repeat"]),
])
def test_pipeline_keeps_treatments_and_stops_on_failure(monkeypatch, tmp_path, fusion, names, trial):
    calls = []

    def run(root, name, *, config, section_limit, reference):
        calls.append(name)
        assert section_limit == 1024
        if fusion:
            assert config["config"]["model"]["scan"].get("ssm_kernel", "torch") == ("triton" if name.startswith("fused") else "torch")
        if len(calls) == 2:
            raise RuntimeError("trial failed")
        return trial()

    monkeypatch.setattr(temporal_pipeline_ab, "run_trial", run)
    with pytest.raises(RuntimeError, match="trial failed"):
        temporal_pipeline_ab.run_trials(tmp_path, fusion=fusion)
    assert calls == names[:2]
    assert list(json.loads((tmp_path / "pipeline.json").read_text())) == names[:1]
