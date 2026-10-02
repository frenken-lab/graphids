import pytest


def _trial(rate=100.0, replays=10):
    return {
        "coverage": {
            "duplicate": 0,
            "unexpected": 0,
            "incorrectly_scored_warmup_or_padding": 0,
            "duplicate_successor_targets": 0,
            "unexpected_successor_targets": 0,
            "unique": 100,
            "observed_successor_targets": 100,
            "warmup_events": 10,
            "scored_events_per_second": rate,
        },
        "planned_scored_events": 100,
        "planned_successor_targets": 100,
        "planned_warmup_events": 10,
        "initial_weight_sha256": "same",
        "section_queue_sha256": "same",
        "cuda_device": "V100",
        "cuda_graph": {"replays": replays, "disabled": False},
        "optimizer_steps": 10,
        "end_to_end_scored_events_per_second": rate,
    }


@pytest.fixture(name="trial")
def trial_factory():
    return _trial
