from graphids.core.data.preprocessing.representations import representation_kind
from graphids.exp.config import ExperimentConfig


def test_defaults_to_event_representation():
    cfg = ExperimentConfig(experiment_name="demo", dataset="toy")
    run = cfg.build_run(name="demo-run", stage="fit")
    assert representation_kind(cfg.representation_cfg) == "can_event"
    assert representation_kind(run.representation_cfg) == "can_event"
    assert run.mlflow_tags()["graphids.phase"] == "fit"


def test_from_yaml_resolves_git_sha(monkeypatch, tmp_path):
    import graphids.exp.config as config_mod

    monkeypatch.setattr(config_mod, "_current_git_sha", lambda: "abc123")
    path = tmp_path / "demo.yml"
    path.write_text("experiment_name: demo\ndataset: toy\n")
    assert ExperimentConfig.from_yaml(path).git_sha == "abc123"
    path.write_text("experiment_name: demo\ndataset: toy\ngit_sha: explicit456\n")
    assert ExperimentConfig.from_yaml(path).git_sha == "explicit456"
