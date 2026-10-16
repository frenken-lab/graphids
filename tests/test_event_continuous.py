"""Continuous state, stream scheduling, and objective accumulation gates."""

from dataclasses import replace

import pytest
import torch

from graphids.core.data.events.data import EventStreamWindows
from graphids.core.models.event_graph_mamba import EventGraphMambaState
from tests.test_event_graph_mamba import event_record, model

pytestmark = pytest.mark.slurm


def state_close(a, b):
    assert (a.next_row, a.ids) == (b.next_row, b.ids)
    torch.testing.assert_close(a.nodes, b.nodes, rtol=4e-4, atol=4e-5)
    torch.testing.assert_close(a.previous_bus, b.previous_bus, rtol=4e-4, atol=4e-5)
    for ca, cb in zip(a.caches, b.caches, strict=True):
        for x, y in zip(ca, cb, strict=True):
            torch.testing.assert_close(x, y, rtol=4e-4, atol=4e-5)


@pytest.mark.parametrize("backend", ["reference", "official"])
def test_differentiable_scan_incoming_state_and_truncated_gradients(tmp_path, backend):
    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(0, 6).to(device)
    actual, reference = model(backend).to(device), model().to(device)
    reference.load_state_dict(actual.state_dict())
    # An incoming nonzero cache exercises the decay correction's gradients.
    for net in (actual, reference):
        _, incoming = net.scan([batch.slice(0, 2)], detach=True)
        incoming[0].nodes.requires_grad_()
        incoming[0].previous_bus.requires_grad_()
        for c, m in incoming[0].caches:
            c.requires_grad_(), m.requires_grad_()
        outputs, following = net.scan([batch.slice(2, 6)], incoming, detach=True)
        loss, _ = net.losses(outputs[0], batch.slice(2, 6))
        loss.backward()
        assert not following[0].nodes.requires_grad
        assert all(not x.requires_grad for cache in following[0].caches for x in cache)
        assert incoming[0].nodes.grad is not None
        assert all(x.grad is not None for c in incoming[0].caches for x in c)
        net.checked_output, net.checked_state = outputs[0], following[0]
    for name in actual.checked_output:
        torch.testing.assert_close(
            actual.checked_output[name],
            reference.checked_output[name],
            rtol=4e-4,
            atol=4e-5,
        )
    state_close(actual.checked_state, reference.checked_state)
    for (name, p), (_, q) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("backend", ["reference", "official"])
def test_untruncated_scan_gradients_match_functional_full_sequence(tmp_path, backend):
    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(0, 6).to(device)
    chunked, full = model(backend).to(device), model().to(device)
    full.load_state_dict(chunked.state_dict())
    a, state = chunked.scan([batch.slice(0, 2)])
    b, _ = chunked.scan([batch.slice(2, 6)], state)
    output = {key: torch.cat((a[0][key], b[0][key])) for key in a[0]}
    chunked.losses(output, batch)[0].backward()
    full.losses(full(batch), batch)[0].backward()
    for (name, p), (_, q) in zip(
        chunked.named_parameters(), full.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-2, atol=1e-4, msg=name)


@pytest.mark.parametrize("backend", ["reference", "official"])
def test_stream_batching_matches_serial_with_distinct_ids_and_replacement(
    tmp_path, backend
):
    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    _, a = event_record(tmp_path, key="a", normal=True)
    _, b = event_record(tmp_path, key="b")
    _, c = event_record(tmp_path, key="c", normal=True)
    # Truncate b to create a different tail; map IDs without changing payloads.
    b.count = 5
    original = b.batch

    def different_ids(start, stop):
        batch = original(start, stop)
        return replace(
            batch,
            ids=batch.ids + 123,
            routing_ids=tuple(i + 123 for i in batch.routing_ids),
        )

    b.batch = different_ids
    dataset = EventStreamWindows([a, b, c], 2, streams=2, effective_events=8)
    net = model(backend).to(device).eval()
    states, coverage = {}, {}
    serial = {r.recording.key: net(r.batch(0, r.count).to(device)) for r in (a, b, c)}
    for window in dataset:
        window = window.to(device)
        for batches in window.groups:
            outputs, following = net.scan(
                batches,
                [states.get(x.recording, EventGraphMambaState()) for x in batches],
                detach=True,
            )
            for batch, out, state in zip(batches, outputs, following, strict=True):
                rows = coverage.setdefault(batch.recording, [])
                rows.extend(range(batch.row_start, batch.row_start + len(batch.ids)))
                for name in out:
                    torch.testing.assert_close(
                        out[name],
                        serial[batch.recording][name][batch.row_start : state.next_row],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                if batch.recording_end:
                    states.pop(batch.recording, None)
                else:
                    states[batch.recording] = state
    assert coverage == {"a": list(range(6)), "b": list(range(5)), "c": list(range(6))}
    assert not states


def test_accumulation_uses_separate_complete_window_denominators(tmp_path):
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(0, 6)
    batch.labels[1:4] = -1
    batch.scored[0] = False
    one, accumulated = model(), model()
    one.losses(one(batch), batch)[0].backward()
    nc = int((batch.scored & (batch.labels >= 0)).sum())
    npayload = int((batch.lengths * batch.scored).sum())
    state = None
    for lo, hi in ((0, 1), (1, 4), (4, 6)):
        part = batch.slice(lo, hi)
        output, state = accumulated.scan([part], state)
        cs, _, ps, _ = accumulated.loss_sums(output[0], part)
        # Retain recurrence graph here: compare combined, untruncated objective.
        ((cs / nc) + ps / npayload).backward(retain_graph=True)
    for p, q in zip(one.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
    masked = replace(
        batch, labels=torch.full_like(batch.labels, -1), verified_normal=False
    )
    loss, _ = one.losses(one(masked), masked)
    assert loss.item() == 0
    loss.backward()


def test_mid_epoch_checkpoint_rejected():
    net = model()
    net._epoch_complete = False
    with pytest.raises(ValueError, match="mid-epoch"):
        net.on_save_checkpoint({})


def test_duplicate_physical_rows_are_preserved(tmp_path):
    from graphids.core.data.events.data import EventRecording, materialize_events
    from tests.event_fixtures import write_record

    rec = write_record(
        tmp_path, rows=[(0.0, "1", "AA", 0), (0.0, "1", "AA", 0), (0.01, "1", "BB", 1)]
    )
    events = EventRecording(materialize_events(rec, tmp_path / "cache"))
    assert events.count == 3
    batch = events.batch(0, 3)
    assert batch.timestamps.tolist() == [0.0, 0.0, 0.01]
    assert batch.payload[:, 0].tolist() == [170, 170, 187]
    windows = EventStreamWindows([events], 2, 1, 4)
    assert [b.row_start for w in windows for g in w.groups for b in g] == [0, 2]


@pytest.mark.parametrize("checkpoint_mode", ["manual", "validation_callback"])
def test_epoch_boundary_resume_matches_uninterrupted_optimizer_and_rng(
    tmp_path, checkpoint_mode
):
    import lightning.pytorch as pl

    from graphids.core.models.event_graph_mamba import EventGraphMamba
    from tests.test_event_graph_mamba import source_dm

    dm = source_dm(tmp_path)

    def new_model():
        torch.manual_seed(42)
        net = EventGraphMamba(
            width=8, layers=1, d_state=3, expand=1, backend="reference", dropout=0.2
        )
        net.prepare_from_datamodule(dm)
        return net

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

    full = new_model()
    full_trainer = trainer("full", 2)
    full_trainer.fit(full, datamodule=dm)
    first = new_model()
    first_trainer = trainer("first", 1)
    if checkpoint_mode == "validation_callback":
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=tmp_path / "selected",
            monitor="val/classification_frame_ap",
            mode="max",
            save_last=True,
            save_on_train_epoch_end=False,
        )
        first_trainer.callbacks.append(checkpoint)
    first_trainer.fit(first, datamodule=dm)
    if checkpoint_mode == "manual":
        path = tmp_path / "epoch.ckpt"
        first_trainer.save_checkpoint(path)
    else:
        path = checkpoint.last_model_path
    saved = torch.load(path, weights_only=True)
    assert saved["event_input_contract"]["epoch_boundary"]
    resumed = new_model()
    resumed_trainer = trainer("resumed", 2)
    resumed_trainer.fit(resumed, datamodule=dm, ckpt_path=path)
    assert resumed_trainer.global_step == full_trainer.global_step == 2
    for name, p in full.state_dict().items():
        torch.testing.assert_close(
            p, resumed.state_dict()[name], rtol=0, atol=0, msg=name
        )
    for a, b in zip(
        full_trainer.optimizers[0].state.values(),
        resumed_trainer.optimizers[0].state.values(),
        strict=True,
    ):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_incomplete_epoch_checkpoint_is_rejected_even_at_last_limited_batch(tmp_path):
    import lightning.pytorch as pl

    from tests.test_event_graph_mamba import source_dm

    dm = source_dm(tmp_path)
    dm.effective_batch_events = 2
    net = model()
    net.prepare_from_datamodule(dm)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path / "limited",
    )
    with pytest.raises(ValueError, match="incomplete event training epoch"):
        trainer.fit(net, datamodule=dm)
    with pytest.raises(ValueError, match="mid-epoch"):
        net.on_save_checkpoint({})


@pytest.mark.parametrize("backend", ["reference", "official"])
def test_optimized_graph_gathers_match_archived_outputs_and_gradients(
    tmp_path, backend
):
    import types

    from scripts.event_model_gate import baseline_readout

    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(0, 6).to(device)
    baseline, optimized = model(backend).to(device), model(backend).to(device)
    baseline._readout = types.MethodType(baseline_readout(), baseline)
    results = []
    for net in (baseline, optimized):
        _, state = net.scan([batch.slice(0, 2)], detach=True)
        outputs, state = net.scan([batch.slice(2, 6)], state, detach=True)
        net.losses(outputs[0], batch.slice(2, 6))[0].backward()
        results.append((outputs[0], state[0]))
    for name in results[0][0]:
        torch.testing.assert_close(
            results[0][0][name], results[1][0][name], rtol=4e-4, atol=4e-5
        )
    state_close(results[0][1], results[1][1])
    for (name, p), (_, q) in zip(
        baseline.named_parameters(), optimized.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-2, atol=1e-4, msg=name)


def test_retired_configuration_errors_reference_verified_archive():
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.runtime import build_component

    for spec in (
        {"type": "can_snapshot_dm"},
        {"type": "vehicle_graph_mamba_joint"},
        {"class_path": "graphids.primitives_losses.CELossCfg"},
    ):
        with pytest.raises(ValueError, match="archives/snapshot-2026-10-16.tar.gz"):
            build_component(spec)
    with pytest.raises(ValueError, match="archives/snapshot-2026-10-16.tar.gz"):
        ExperimentConfig.from_yaml("configs/experiments/graph_mamba_joint.yml")
    with pytest.raises(ValueError, match="archives/snapshot-2026-10-16.tar.gz"):
        ExperimentConfig(
            experiment_name="retired",
            dataset="toy",
            representation_cfg={"kind": "can_snapshot"},
        )


def test_microbatch_capacity_is_validated_before_launch():
    from graphids.primitives_data import CANEventDMCfg

    with pytest.raises(ValueError, match="microbatch"):
        CANEventDMCfg(
            manifest_path="unused",
            datasets=["set_01"],
            cache_root="unused",
            chunk_events=2048,
            stream_batch_size=8,
        )


def test_event_verification_cli_backend_publishes_complete_and_failed_reports(tmp_path):
    import json

    import numpy as np
    import yaml

    from graphids.core.data.events.verify import verify_events
    from tests.test_event_graph_mamba import source_dm

    dm = source_dm(tmp_path)
    config = tmp_path / "audit.yml"
    config.write_text(
        yaml.safe_dump(
            dict(
                experiment_name="audit",
                dataset="fixture",
                config=dict(
                    data=dict(
                        type="can_event_dm",
                        manifest_path=dm.manifest_path,
                        datasets=dm.datasets,
                        cache_root=str(dm.cache_root),
                    )
                ),
            )
        )
    )
    report = tmp_path / "audit.json"
    assert verify_events(config, report)["frames"] == 12
    assert json.loads(report.read_text())["status"] == "complete"
    first = next(iter(dm.event_recordings.values()))
    values = np.load(first.path / "payload.npy", mmap_mode="r+")
    values[0, 0] ^= 1
    values.flush()
    with pytest.raises(AssertionError):
        verify_events(config, report)
    assert json.loads(report.read_text())["status"] == "incomplete"


def test_exact_frame_ap_preserves_raw_score_ties_and_unknown_labels():
    from graphids.core.metrics import raw_average_precision

    assert raw_average_precision([2.0, 2.0, 1.0, 20.0], [1, 0, 1, -1]) == pytest.approx(
        (0.5 + 2 / 3) / 2
    )
    assert raw_average_precision([1.0, 1.0], [0, 1]) == 0.5
    assert raw_average_precision([1.0, 2.0], [0, -1]) is None
