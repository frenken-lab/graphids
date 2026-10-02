from pathlib import Path

import pytest
import yaml

from graphids.exp.config import ExperimentConfig, RunConfig, load_experiment_yaml


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))
    return path


def test_relative_inheritance_preserves_placeholders_and_replaces_values(tmp_path):
    write(tmp_path / "base.yml", {
        "experiment_name": "base", "dataset": "set_01",
        "config": {"trainer": {"max_epochs": 3, "plugins": ["old"]},
                   "compile": {"backend": "inductor"},
                   "callbacks": {"last": {"class_path": "module.Callback", "init_args": {"dirpath": "{run_dir}/checkpoints"}}}},
    })
    write(tmp_path / "nested/parent.yml", {"extends": "../base.yml", "config": {"trainer": {"max_epochs": 2}}})
    path = write(tmp_path / "leaf.yml", {
        "extends": "nested/parent.yml", "experiment_name": "leaf",
        "config": {"trainer": {"plugins": []}, "compile": None},
    })
    expanded = load_experiment_yaml(path)
    assert "extends" not in expanded
    assert expanded["config"]["trainer"] == {"max_epochs": 2, "plugins": []}
    assert expanded["config"]["compile"] is None
    assert expanded["config"]["callbacks"]["last"]["init_args"]["dirpath"] == "{run_dir}/checkpoints"
    experiment = ExperimentConfig.from_yaml(path)
    run = experiment.build_run(name="leaf", stage="fit", run_dir=tmp_path / "output")
    assert run.payload.callbacks["last"]["init_args"]["dirpath"] == str(tmp_path / "output/checkpoints")
    assert RunConfig.model_validate(yaml.safe_load(run.write_config().read_text())) == run
    expanded_path = write(tmp_path / "standalone.yml", expanded)
    assert ExperimentConfig.from_yaml(expanded_path) == experiment


@pytest.mark.parametrize("key", ["type", "class_path"])
def test_component_change_discards_inherited_arguments(tmp_path, key):
    write(tmp_path / "base.yml", {"config": {"model": {key: "old", "obsolete": 1}}})
    path = write(tmp_path / "leaf.yml", {"extends": "base.yml", "config": {"model": {key: "new", "hidden": 8}}})
    assert load_experiment_yaml(path)["config"]["model"] == {key: "new", "hidden": 8}


def test_inheritance_errors_identify_the_source_files(tmp_path):
    a = write(tmp_path / "a.yml", {"extends": "b.yml"})
    b = write(tmp_path / "b.yml", {"extends": "a.yml"})
    with pytest.raises(ValueError, match="inheritance cycle") as exc:
        load_experiment_yaml(a)
    assert str(a) in str(exc.value) and str(b) in str(exc.value)
    b.unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_experiment_yaml(a)
    assert str(a) in str(exc.value) and str(b) in str(exc.value)


@pytest.mark.parametrize("value", [[], "scalar", None])
def test_non_mapping_yaml_is_rejected(tmp_path, value):
    path = write(tmp_path / "bad.yml", value)
    with pytest.raises(ValueError, match="must be a mapping"):
        load_experiment_yaml(path)


@pytest.mark.parametrize("base", [[], 1, "", None])
def test_extends_accepts_only_one_nonempty_path(tmp_path, base):
    path = write(tmp_path / "bad.yml", {"extends": base})
    with pytest.raises(ValueError, match="extends must be"):
        load_experiment_yaml(path)


def test_all_experiment_leaves_build_runs():
    paths = [p for p in Path("configs/experiments").rglob("*.yml") if "_base" not in p.parts]
    assert paths
    for path in paths:
        experiment = ExperimentConfig.from_yaml(path)
        experiment.build_run(name=experiment.experiment_name, stage=experiment.stage)
