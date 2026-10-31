"""Selected per-ID timing-gap causality, readout, and checkpoint contracts."""

from copy import copy, deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from graphids.data.context import fit_timing_statistics, previous_rows
from graphids.data.events import EventRecording, materialize_events
from graphids.data.representation import CANEventRepresentationConfig
from graphids.models.event_graph_mamba import EventGraphMamba
from tests.event_fixtures import source_dm, write_record
from tests.event_reference import serial_loss, serial_scan


def context_model(dm, **options):
    torch.manual_seed(42)
    kwargs = dict(
        width=8,
        layers=2,
        d_state=3,
        d_conv=3,
        expand=1,
        dropout=0,
        backend="reference",
        schema_version=2,
        use_id_gap=True,
        timing_min_support=2,
    )
    kwargs.update(options)
    net = EventGraphMamba(**kwargs)
    net.prepare_from_datamodule(dm)
    return net


def test_context_matches_physical_predecessors_across_slices_and_payload_lengths(
    tmp_path,
):
    rec = write_record(
        tmp_path,
        rows=[
            (1e9, "A", "0102", 0),
            (1e9, "B", "FF", 0),
            (1e9, "A", "03", 0),
            (1e9 + 0.001, "B", "0011", 0),
            (1e9 + 0.002, "A", "040506", 0),
            (1e9 + 0.003, "A", "", 0),
        ],
    )
    events = EventRecording(
        materialize_events(rec, tmp_path / "cache"), local_context=True
    )
    full = events.batch(0, events.count)
    expected, latest = [], {}
    for row, key in enumerate(full.routing_ids):
        prior = latest.get(key)
        expected.append(prior)
        assert bool(full.context.seen[row]) == (prior is not None)
        if prior is not None:
            assert (
                full.context.gaps[row] == full.timestamps[row] - full.timestamps[prior]
            )
        latest[key] = row
    assert events.previous_rows.tolist() == [
        -1 if row is None else row for row in expected
    ]
    for lo, hi in ((0, 2), (2, 5), (5, 6)):
        part, sliced = events.batch(lo, hi), full.slice(lo, hi)
        for name, values in vars(part.context).items():
            torch.testing.assert_close(values, getattr(full.context, name)[lo:hi])
            torch.testing.assert_close(values, getattr(sliced.context, name))
    # Future IDs and bytes cannot change any preceding observation feature.
    changed = events.arrays["ids"].copy()
    changed[4:] = 0xF
    np.testing.assert_array_equal(previous_rows(changed)[:4], events.previous_rows[:4])


def test_fitted_timing_ignores_nontraining_and_untrusted_recordings(tmp_path):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    records = list(dm.event_recordings.values())
    keys, statistics = fit_timing_statistics(records, min_support=2)
    assert keys.tolist() == [0xC9]
    # These arrays must not even be accessed while fitting.
    untrusted = copy(records[0])
    untrusted.recording = replace(
        untrusted.recording, key="untrusted", verified_normal=False
    )
    untrusted.arrays = None
    validation = copy(records[1])
    validation.arrays = None
    actual_keys, actual = fit_timing_statistics(
        [untrusted, validation, records[0]], min_support=2
    )
    np.testing.assert_array_equal(actual_keys, keys)
    np.testing.assert_array_equal(actual, statistics)
    with pytest.raises(ValueError, match="verified-normal training"):
        fit_timing_statistics([untrusted, validation])
    net = context_model(dm)
    batch = records[0].batch(0, 6)
    features = net._id_gap_features(batch)
    assert features[:, 1].tolist() == [0, 1, 1, 0, 1, 0]
    assert features[:, 2].tolist() == [1, 0, 0, 1, 0, 0]
    assert features[0, 0] == features[3, 0] == 0
    unknown = replace(batch, ids=torch.full_like(batch.ids, 0xFFFF))
    assert not net._id_gap_features(unknown)[:, 1].any()
    # Unseen IDs use the frozen pooled distribution, never evaluation-fitted scales.
    assert torch.isfinite(net._id_gap_features(unknown)).all()
    huge = torch.tensor([1e8, 1e12], dtype=torch.float64)
    clipped = net._normalize_gap(huge, net.timing_statistics[0])
    assert clipped[0] == clipped[1]


@pytest.mark.parametrize(
    "execution",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.slurm])],
)
def test_context_stream_outputs_states_and_gradients_match_serial_oracle(
    tmp_path, execution
):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    net = (
        context_model(dm, backend="official" if execution == "cuda" else "reference")
        .to(execution)
        .eval()
    )
    oracle = deepcopy(net)
    records = list(dm.event_recordings.values())
    actual_states = expected_states = None
    actual_loss = expected_loss = 0
    for lo in (0, 3):
        batches = [record.batch(lo, lo + 3).to(execution) for record in records]
        outputs, actual_states = net.scan(batches, actual_states)
        expected, expected_states = serial_scan(oracle, batches, expected_states)
        for output, reference, batch, state, serial_state in zip(
            outputs, expected, batches, actual_states, expected_states, strict=True
        ):
            for name in output:
                torch.testing.assert_close(
                    output[name], reference[name], rtol=5e-4, atol=5e-5
                )
            torch.testing.assert_close(
                state.nodes, serial_state.nodes, rtol=5e-4, atol=5e-5
            )
            actual_loss = actual_loss + serial_loss(net, output, batch)
            expected_loss = expected_loss + serial_loss(oracle, reference, batch)
    actual_loss.backward()
    expected_loss.backward()
    for (name, p), (_, q) in zip(
        net.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=3e-4, msg=name)
    # Independent stream state and output also agree with a complete chronological scan.
    full = [record.batch(0, 6).to(execution) for record in records]
    whole, whole_states = net.scan(full)
    for stream, record in enumerate(records):
        state, parts = None, []
        for lo in (0, 2, 4):
            output, state = net.stream(record.batch(lo, lo + 2).to(execution), state)
            parts.append(output)
        for name in whole[stream]:
            torch.testing.assert_close(
                torch.cat([part[name] for part in parts]),
                whole[stream][name],
                rtol=5e-4,
                atol=5e-5,
            )
        torch.testing.assert_close(state.nodes, whole_states[stream].nodes)
        assert state.recording == record.recording.key


def test_current_payload_does_not_leak_into_forecasts(tmp_path):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    net = context_model(dm).eval()
    events = list(dm.event_recordings.values())[0]
    batch = events.batch(0, 6)
    _, header = net.fields(batch)
    # Perturb raw traffic and rebuild all derived context to avoid stale metadata.
    altered = copy(events)
    altered.arrays = {k: np.array(v) for k, v in events.arrays.items()}
    altered.arrays["payload"][2:] ^= 255
    modified = altered.batch(0, 6)
    original = net(batch)
    actual = net(modified)
    torch.testing.assert_close(net.fields(modified)[1], header, rtol=0, atol=0)
    torch.testing.assert_close(
        original["classification_logits"][:2], actual["classification_logits"][:2]
    )
    torch.testing.assert_close(
        original["payload_logits"][:3], actual["payload_logits"][:3]
    )
    assert not torch.equal(
        original["classification_logits"][2], actual["classification_logits"][2]
    )
    assert not torch.equal(original["payload_logits"][3:], actual["payload_logits"][3:])


@pytest.mark.slurm
def test_context_checkpoint_preserves_normalizers_and_exact_optimizer_rng_resume(
    tmp_path, monkeypatch
):
    import lightning.pytorch as pl

    import graphids.data.context as context

    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )

    def trainer(name, epochs):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path / name,
        )

    full = context_model(dm, dropout=0.2)
    full_trainer = trainer("full", 2)
    full_trainer.fit(full, datamodule=dm)
    first = context_model(dm, dropout=0.2)
    first_trainer = trainer("first", 1)
    first_trainer.fit(first, datamodule=dm)
    path = tmp_path / "context.ckpt"
    first_trainer.save_checkpoint(path)
    loaded = EventGraphMamba.load_from_checkpoint(path)

    def no_refit(*args, **kwargs):
        raise AssertionError("loaded checkpoint tried to refit normalizers")

    monkeypatch.setattr(context, "fit_timing_statistics", no_refit)
    loaded.prepare_from_datamodule(dm)
    torch.testing.assert_close(
        loaded.timing_statistics, first.timing_statistics, rtol=0, atol=0
    )
    loaded.eval(), first.eval()
    batch = next(iter(dm.val_dataloader()))
    for name, output in first(batch).items():
        torch.testing.assert_close(loaded(batch)[name], output, rtol=0, atol=0)
    loaded.train()
    resumed_trainer = trainer("resumed", 2)
    resumed_trainer.fit(loaded, datamodule=dm, ckpt_path=path)
    assert resumed_trainer.global_step == full_trainer.global_step
    for name, value in full.state_dict().items():
        torch.testing.assert_close(
            value, loaded.state_dict()[name], rtol=0, atol=0, msg=name
        )
    for a, b in zip(
        full_trainer.optimizers[0].state.values(),
        resumed_trainer.optimizers[0].state.values(),
        strict=True,
    ):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_selected_preset_and_defaults_require_representation_v2(tmp_path):
    from pathlib import Path

    import yaml

    from graphids.exp.config import ExperimentConfig, RunPayload
    from graphids.models.config import EventGraphMambaConfig

    base = yaml.safe_load(
        Path("configs/experiments/graph_mamba_events_source.yml").read_text()
    )
    ExperimentConfig.model_validate(base)
    assert EventGraphMambaConfig().use_id_gap
    assert EventGraphMambaConfig().schema_version == 2
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    net = context_model(dm)
    batch = next(iter(dm.val_dataloader()))
    fields, header = net.fields(batch)
    assert fields.shape == (len(batch.ids), 51)
    assert header.shape == (len(batch.ids), 35)
    invalid = deepcopy(base["config"])
    invalid["data"]["representation_cfg"]["schema_version"] = 1
    with pytest.raises(ValueError, match="representation schema_version=2"):
        RunPayload.model_validate(invalid)
    # Omitting the model's flags must still validate the actual selected defaults.
    invalid["model"] = {"type": "event_graph_mamba"}
    with pytest.raises(ValueError, match="representation schema_version=2"):
        RunPayload.model_validate(invalid)


@pytest.mark.parametrize(
    "options",
    [
        dict(use_elapsed_time=False),
        dict(timing_mode="normalized"),
        dict(payload_context="delta"),
        dict(payload_context="xor"),
        dict(payload_context="both"),
        dict(graph_age=True),
    ],
)
def test_removed_variants_fail_explicitly(options):
    from graphids.models.config import EventGraphMambaConfig

    with pytest.raises(ValueError, match="variant is retired"):
        EventGraphMambaConfig(**options)
    with pytest.raises(ValueError, match="variant is retired"):
        EventGraphMamba(backend="reference", **options)


@pytest.mark.parametrize("schema", [1, 2])
def test_historical_neutral_flags_preserve_checkpoint_identity(tmp_path, schema):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    net = context_model(dm, schema_version=schema, use_id_gap=schema == 2)
    checkpoint = {}
    net.on_save_checkpoint(checkpoint)
    old_flags = dict(
        use_elapsed_time=True,
        timing_mode="raw",
        payload_context="none",
        graph_age=False,
        graph_age_scale=1.0,
    )
    checkpoint["event_input_contract"]["model_config"].update(old_flags)
    import lightning.pytorch as pl

    path = tmp_path / f"historical-schema{schema}.ckpt"
    checkpoint.update(
        state_dict=net.state_dict(),
        hyper_parameters=dict(net.hparams) | old_flags,
        **{"pytorch-lightning_version": pl.__version__},
    )
    torch.save(checkpoint, path)
    loaded = EventGraphMamba.load_from_checkpoint(path)
    loaded.prepare_from_datamodule(dm)
    assert not set(old_flags) & set(loaded.hparams)
    batch = next(iter(dm.val_dataloader()))
    for key, value in net(batch).items():
        torch.testing.assert_close(loaded(batch)[key], value, rtol=0, atol=0)


def test_invalid_fitted_statistics_and_incompatible_feature_checkpoints_are_rejected(
    tmp_path,
):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    net = context_model(dm)
    checkpoint = {}
    net.on_save_checkpoint(checkpoint)
    old = EventGraphMamba(
        width=8, backend="reference", schema_version=1, use_id_gap=False
    )
    with pytest.raises(ValueError, match="incompatible"):
        old.on_load_checkpoint(checkpoint)
    changed = context_model(dm, timing_min_support=3)
    with pytest.raises(ValueError, match="configuration differs"):
        changed.on_load_checkpoint(checkpoint)
    corrupted = {name: value.clone() for name, value in net.state_dict().items()}
    corrupted["timing_statistics"][:, 2] = 0
    with pytest.raises(ValueError, match="invalid fitted timing"):
        net.load_state_dict(corrupted)
