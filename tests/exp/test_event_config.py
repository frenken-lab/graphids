"""Project-specific event configuration contracts."""

import pytest

from graphids.exp.config import ExperimentConfig, FitRunPayload
from graphids.primitives_data import CANEventDMCfg


def test_microbatch_limits_agree_for_config_and_direct_construction():
    """Both construction paths enforce the same two coupled event budgets."""
    from graphids.core.data.events.data import CANEventDataModule

    for streams, cap, effective, accepted in (
        (1, 16384, 16384, True),
        (16, 262144, 262144, True),
        (2, 16384, 32768, False),
        (2, 32768, 16384, False),
    ):
        kwargs = dict(
            manifest_path="unused",
            datasets=["set_01"],
            cache_root="unused",
            chunk_events=16384,
            stream_batch_size=streams,
            max_microbatch_events=cap,
            effective_batch_events=effective,
        )
        if accepted:
            configured = CANEventDMCfg(**kwargs).build()
            direct = CANEventDataModule(**kwargs)
            assert configured.stream_batch_size == direct.stream_batch_size == streams
            assert (
                configured.max_microbatch_events == direct.max_microbatch_events == cap
            )
        else:
            for constructor in (CANEventDMCfg, CANEventDataModule):
                with pytest.raises(ValueError, match="microbatch"):
                    constructor(**kwargs)


def test_retired_loss_override_is_rejected_at_config_boundaries(tmp_path):
    """A historical loss setting must never silently change or bypass objectives."""
    from graphids.primitives_models import EventGraphMambaCfg

    legacy = {"loss_fn": {"class_path": "torch.nn.MSELoss"}}
    path = tmp_path / "legacy.yml"
    path.write_text("experiment_name: legacy\ndataset: toy\nconfig:\n  loss_fn: {}\n")
    with pytest.raises(ValueError, match="loss_fn is retired"):
        ExperimentConfig.from_yaml(path)
    with pytest.raises(ValueError, match="loss_fn is retired"):
        FitRunPayload(**legacy)
    experiment = ExperimentConfig(experiment_name="run", dataset="toy")
    with pytest.raises(ValueError, match="loss_fn is retired"):
        experiment.build_run(name="run", stage="fit", config=legacy)
    # Older resolved payloads emitted this unused field as null.
    assert "loss_fn" not in FitRunPayload(loss_fn=None).model_dump()
    model = EventGraphMambaCfg(width=8, backend="reference").build()
    assert "loss_fn" not in model.hparams
