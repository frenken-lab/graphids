"""The expensive fit comparisons must differ only in their declared controls."""

from copy import deepcopy
from pathlib import Path

import yaml

from graphids.exp.config import ExperimentConfig
from scripts.event_ablation import experiments


def test_matched_training_controls_and_classification_only_arm():
    base = yaml.safe_load(
        Path("configs/experiments/graph_mamba_events_source.yml").read_text()
    )
    original = deepcopy(base)
    arms = list(experiments(base, 46.660762196485265, classification_only=True))
    assert base == original
    assert len(arms) == len({a["experiment_name"] for a in arms}) == 5
    for arm in arms:
        ExperimentConfig.model_validate(arm)
        normalized = deepcopy(arm)
        normalized["experiment_name"] = arms[0]["experiment_name"]
        normalized["config"]["model"]["positive_weight"] = 1.0
        normalized["config"]["model"]["payload_weight"] = 1.0
        normalized["config"]["data"]["stream_batch_size"] = 1
        assert normalized == arms[0]
    assert [a["config"]["data"]["stream_batch_size"] for a in arms] == [1, 1, 4, 4, 1]
    assert [a["config"]["model"]["positive_weight"] for a in arms] == [
        1.0,
        46.660762196485265,
        1.0,
        46.660762196485265,
        1.0,
    ]
    assert arms[-1]["config"]["model"]["payload_weight"] == 0
    assert len(list(experiments(base, 46.660762196485265))) == 4
