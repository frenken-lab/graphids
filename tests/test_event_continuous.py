"""Continuous state, stream scheduling, and objective accumulation gates."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from graphids.core.data.events.data import EventStreamWindows
from graphids.core.models.event_graph_mamba import EventGraphMambaState
from tests.event_reference import serial_loss, serial_loss_sums, serial_scan
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
    serial_loss(full, serial_scan(full, [batch])[0][0], batch).backward()
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
            routing=None,
            ids=batch.ids + 123,
            routing_ids=tuple(i + 123 for i in batch.routing_ids),
        )

    b.batch = different_ids
    dataset = EventStreamWindows([a, b, c], 2, streams=2, effective_events=8)
    net = model(backend).to(device).eval()
    states, coverage = {}, {}
    serial = {
        r.recording.key: serial_scan(net, [r.batch(0, r.count).to(device)])[0][0]
        for r in (a, b, c)
    }
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
        cs, _, ps, _ = accumulated.batched_loss_sums(output, [part])
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


@pytest.mark.parametrize(
    "execution", ["cpu", "cuda_sync", "cuda_workers", "cuda_prefetch"]
)
@pytest.mark.parametrize("checkpoint_mode", ["manual", "validation_callback"])
def test_epoch_boundary_resume_matches_uninterrupted_optimizer_and_rng(
    tmp_path, checkpoint_mode, execution
):
    import lightning.pytorch as pl

    from graphids.core.models.event_graph_mamba import EventGraphMamba
    from tests.test_event_graph_mamba import source_dm

    if execution != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dm = source_dm(tmp_path)
    dm.cuda_prefetch = execution == "cuda_prefetch"
    dm.pin_memory = execution != "cpu"
    dm.num_workers = 2 if execution in ("cuda_workers", "cuda_prefetch") else 0
    dm.persistent_workers = dm.num_workers > 0

    def new_model():
        torch.manual_seed(42)
        net = EventGraphMamba(
            width=8, layers=1, d_state=3, expand=1, backend="reference", dropout=0.2
        )
        net.prepare_from_datamodule(dm)
        return net

    def trainer(name, epochs):
        return pl.Trainer(
            accelerator="cpu" if execution == "cpu" else "gpu",
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
def test_optimized_graph_matches_independent_serial_outputs_and_gradients(
    tmp_path, backend
):
    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(0, 6).to(device)
    baseline, optimized = model(backend).to(device), model(backend).to(device)
    results = []
    for net in (baseline, optimized):
        scan = partial(serial_scan, net) if net is baseline else net.scan
        _, state = scan([batch.slice(0, 2)], detach=True)
        outputs, state = scan([batch.slice(2, 6)], state, detach=True)
        serial_loss(net, outputs[0], batch.slice(2, 6)).backward()
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


def test_unsupported_components_and_missing_experiments_have_generic_errors():
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.runtime import build_component

    with pytest.raises(ValueError, match="unknown primitive: 'unknown_detector'"):
        build_component({"type": "unknown_detector"})
    with pytest.raises(AttributeError):
        build_component({"class_path": "graphids.primitives_models.UnknownDetector"})
    with pytest.raises(FileNotFoundError, match="experiment YAML not found"):
        ExperimentConfig.from_yaml("configs/experiments/missing.yml")
    with pytest.raises(ValueError, match="can_event"):
        ExperimentConfig(
            experiment_name="unsupported",
            dataset="toy",
            representation_cfg={"kind": "unsupported"},
        )


def test_microbatch_capacity_is_validated_before_launch():
    from graphids.primitives_data import CANEventDMCfg

    with pytest.raises(ValueError, match="microbatch"):
        CANEventDMCfg(
            manifest_path="unused",
            datasets=["set_01"],
            cache_root="unused",
            chunk_events=4096,
            stream_batch_size=8,
            effective_batch_events=32768,
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


@pytest.mark.parametrize(
    "chunk, streams", [(2048, 8), (4096, 4), (8192, 2), (16384, 1), (1024, 16)]
)
def test_doubled_microbatch_cap_accepts_larger_combinations(chunk, streams):
    from graphids.primitives_data import CANEventDMCfg

    cfg = CANEventDMCfg(
        manifest_path="unused",
        datasets=["set_01"],
        cache_root="unused",
        chunk_events=chunk,
        stream_batch_size=streams,
        effective_batch_events=16384,
    )
    assert cfg.build().chunk_events * cfg.stream_batch_size == 16384


def test_ten_stream_schedule_fills_effective_windows_without_replay(tmp_path):
    recordings = [
        event_record(tmp_path, key=f"independent-{i}", normal=True)[1]
        for i in range(10)
    ]
    windows = EventStreamWindows(
        recordings, chunk_events=2, streams=10, effective_events=16
    )
    coverage = {}
    counts = []
    for window in windows:
        counts.append(sum(len(b.ids) for g in window.groups for b in g))
        assert window.classification_targets == counts[-1]
        for group in window.groups:
            assert len(group) <= 10 and len({len(b.ids) for b in group}) == 1
            for batch in group:
                rows = coverage.setdefault(batch.recording, [])
                assert batch.row_start == len(rows)
                rows.extend(range(batch.row_start, batch.row_start + len(batch.ids)))
    assert counts == [16, 16, 16, 12]
    assert coverage == {r.recording.key: list(range(6)) for r in recordings}


@pytest.mark.parametrize("streams", [1, 2, 4, 8, 10])
def test_fixed_16384_chunk_allows_explicit_larger_stream_cap(streams):
    from graphids.primitives_data import CANEventDMCfg

    cfg = CANEventDMCfg(
        manifest_path="unused",
        datasets=["set_01"],
        cache_root="unused",
        chunk_events=16384,
        stream_batch_size=streams,
        max_microbatch_events=16384 * streams,
        effective_batch_events=163840,
    )
    dm = cfg.build()
    assert dm.chunk_events == 16384
    assert dm.max_microbatch_events == 16384 * streams


@pytest.mark.parametrize("workspace", [None, 2304])
@pytest.mark.parametrize("backend", ["reference", "official"])
def test_prepared_batched_graph_losses_and_input_gradients(
    tmp_path, backend, workspace, monkeypatch
):
    if backend == "official" and not torch.cuda.is_available():
        pytest.skip("official CUDA required")
    device = "cuda" if backend == "official" else "cpu"
    if workspace is not None:
        import importlib

        module = importlib.import_module("graphids.core.models.event_graph_mamba")
        monkeypatch.setattr(module, "_GRAPH_WORKSPACE_BYTES", workspace)
    _, a = event_record(tmp_path, key="prepared-a", normal=True)
    _, b = event_record(tmp_path, key="prepared-b")
    from graphids.core.data.events.data import prepare_routing

    def batches(lo, hi):
        first, second = a.batch(lo, hi), b.batch(lo, hi)
        # Different ID sets/counts, including IDs first seen inside the chunk.
        second.ids = torch.arange(lo + 1000, hi + 1000)
        second.routing_ids = tuple(second.ids.tolist())
        second.routing = prepare_routing(
            second.recording,
            second.source_digest,
            lo,
            second.routing_ids,
            tuple(range(1000, lo + 1000)),
        )
        first.labels[::2] = -1
        first.scored[0] = False
        return [first.to(device), second.to(device)]

    optimized, serial = model(backend).to(device), model(backend).to(device)
    results, inputs = [], []
    for net in (optimized, serial):
        prefix = batches(0, 2)
        scan = partial(serial_scan, net) if net is serial else net.scan
        _, incoming = scan(prefix, detach=True)
        for state in incoming:
            state.nodes.requires_grad_()
            state.previous_bus.requires_grad_()
            for cache in state.caches:
                for value in cache:
                    value.requires_grad_()
        current = batches(2, 6)
        for batch in current:
            batch.times.requires_grad_()
        outputs, following = scan(current, incoming)
        if net is optimized:
            c, nc, p, npayload = net.batched_loss_sums(outputs, current)
        else:
            sums = [
                serial_loss_sums(o, b) for o, b in zip(outputs, current, strict=True)
            ]
            c, nc, p, npayload = (sum(x[i] for x in sums) for i in range(4))
        loss = c / nc.clamp_min(1) + p / npayload.clamp_min(1)
        loss.backward()
        inputs.append(
            [b.times.grad for b in current]
            + [
                x.grad
                for state in incoming
                for x in (
                    state.nodes,
                    state.previous_bus,
                    *(v for c in state.caches for v in c),
                )
            ]
        )
        results.append((outputs, following, loss))
    torch.testing.assert_close(results[0][2], results[1][2])
    for actual, expected in zip(results[0][0], results[1][0], strict=True):
        for name in actual:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=5e-4, atol=5e-5
            )
    for actual, expected in zip(results[0][1], results[1][1], strict=True):
        state_close(actual, expected)
    for actual, expected in zip(inputs[0], inputs[1], strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-4)
    for (name, p), (_, q) in zip(
        optimized.named_parameters(), serial.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-2, atol=1e-4, msg=name)
    masked = [
        replace(b, scored=torch.zeros_like(b.scored), verified_normal=False)
        for b in current
    ]
    optimized.zero_grad()
    outputs, _ = optimized.scan(masked, incoming)
    c, nc, p, npayload = optimized.batched_loss_sums(outputs, masked)
    assert (c + p).item() == nc.item() == npayload.item() == 0
    (c + p).backward()
    assert all(
        p.grad is not None and not torch.count_nonzero(p.grad)
        for p in optimized.parameters()
    )


def test_prepared_routing_arbitrary_boundaries_and_provenance(tmp_path):
    _, events = event_record(tmp_path)
    net = model().eval()
    for start in range(1, 6):
        _, incoming = net.scan([events.batch(0, start)], detach=True)
        batch = events.batch(start, 6)
        prepared = net._routing(batch, incoming[0])
        reference = net._routing(replace(batch, routing=None), incoming[0])
        assert prepared[0] == reference[0]
        for a, b in zip(prepared[1:], reference[1:], strict=True):
            torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b))
        for changed in (
            replace(batch, recording="wrong"),
            replace(batch, row_start=start + 1),
        ):
            with pytest.raises(ValueError, match="provenance"):
                net._routing(changed, incoming[0])
        with pytest.raises(ValueError, match="provenance"):
            net._routing(batch, replace(incoming[0], ids=()))


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("prefetch", [False, True])
def test_training_loader_order_rng_and_early_exit(tmp_path, workers, prefetch):
    from graphids.core.data.events.data import EventTrainingLoader

    _, a = event_record(tmp_path, key="loader-a", normal=True)
    _, b = event_record(tmp_path, key="loader-b")
    dataset = EventStreamWindows([a, b], 2, 2, 4)
    loader = EventTrainingLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        cuda_prefetch=prefetch,
        generator=torch.Generator().manual_seed(42),
        **({"prefetch_factor": 2} if workers else {}),
    )
    import multiprocessing

    initial_children = {p.pid for p in multiprocessing.active_children()}
    rng = torch.get_rng_state().clone()
    actual = []
    for window in loader:
        actual.extend(
            (batch.recording, batch.row_start, batch.routing_ids)
            for group in window.groups
            for batch in group
        )
    expected = [
        (b.recording, b.row_start, b.routing_ids)
        for w in dataset
        for g in w.groups
        for b in g
    ]
    assert actual == expected
    assert torch.equal(rng, torch.get_rng_state())
    iterator = iter(loader)
    next(iterator)
    iterator.close()
    # A new epoch still starts at row zero after an abandoned iterator.
    assert next(iter(loader)).groups[0][0].row_start == 0
    assert {p.pid for p in multiprocessing.active_children()} <= initial_children


@pytest.mark.parametrize("prefetch", [False, True])
def test_loader_propagates_preparation_errors(tmp_path, prefetch):
    from graphids.core.data.events.data import EventTrainingLoader

    _, events = event_record(tmp_path)
    dataset = EventStreamWindows([events], 2, 1, 2)
    events.count = 1  # stale scheduled window is an explicit preparation failure
    loader = EventTrainingLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        cuda_prefetch=prefetch,
        generator=torch.Generator().manual_seed(42),
    )
    with pytest.raises(ValueError, match="invalid event window"):
        list(loader)


def test_cuda_prefetch_preserves_scores_gradients_and_dropout_rng(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from graphids.core.data.events.data import EventTrainingLoader
    from graphids.core.models.event_graph_mamba import EventGraphMamba

    _, a = event_record(tmp_path, key="parity-a", normal=True)
    _, b = event_record(tmp_path, key="parity-b")
    dataset = EventStreamWindows([a, b], 2, 2, 4)
    results = []
    for prefetch in (False, True):
        torch.manual_seed(42)
        net = (
            EventGraphMamba(
                width=8, layers=2, d_state=3, expand=1, backend="official", dropout=0.2
            )
            .cuda()
            .train()
        )
        loader = EventTrainingLoader(
            dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            cuda_prefetch=prefetch,
            generator=torch.Generator().manual_seed(42),
        )
        states, scores = {}, []
        for window in loader:
            window = window.to("cuda", non_blocking=True)
            for batches in window.groups:
                outputs, following = net.scan(
                    batches,
                    [states.get(b.recording, EventGraphMambaState()) for b in batches],
                    detach=True,
                )
                c, _, p, _ = net.batched_loss_sums(outputs, batches)
                (
                    c / max(1, window.classification_targets)
                    + p / max(1, window.payload_targets)
                ).backward()
                scores.extend(
                    o["classification_logits"].detach().clone() for o in outputs
                )
                states.update(
                    (b.recording, s) for b, s in zip(batches, following, strict=True)
                )
        results.append(
            (
                scores,
                [p.grad.detach().clone() for p in net.parameters()],
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
            )
        )
    for i, (actual, expected) in enumerate(
        zip(results[0][:2], results[1][:2], strict=True)
    ):
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                a, b, rtol=0 if i == 0 else 1e-2, atol=0 if i == 0 else 1e-4
            )
    assert torch.equal(results[0][2], results[1][2])
    assert torch.equal(results[0][3], results[1][3])


def test_gate_defaults_follow_public_configuration_and_explicit_overrides():
    from argparse import Namespace

    from scripts.event_model_gate import benchmark_options

    args = Namespace(
        config="configs/experiments/graph_mamba_events_source.yml",
        max_microbatch_events=None,
        effective_events=None,
        chunks=None,
        streams=None,
        workers=None,
        cuda_prefetch=None,
        repeats=3,
        no_profiles=True,
        events_per_recording=65536,
    )
    options = benchmark_options(args)
    assert options["chunks"] == [16384]
    assert options["stream_counts"] == [1]
    assert options["effective_events"] == options["max_microbatch_events"] == 16384
    assert options["workers"] == 2 and options["persistent_workers"]
    assert options["pin_memory"] and not options["cuda_prefetch"]
    args.workers, args.cuda_prefetch, args.chunks = 0, True, [32768]
    options = benchmark_options(args)
    assert options["workers"] == 0 and options["cuda_prefetch"]
    assert options["chunks"] == [32768]


def test_independent_oracle_detects_optimized_readout_perturbation(
    tmp_path, monkeypatch
):
    _, events = event_record(tmp_path)
    batch = events.batch(0, 6)
    net = model().eval()
    expected = serial_scan(net, [batch])[0][0]
    original = net._readout_batched

    def perturbed(*args, **kwargs):
        outputs, nodes = original(*args, **kwargs)
        outputs[0]["classification_logits"] = outputs[0]["classification_logits"] + 1
        return outputs, nodes

    monkeypatch.setattr(net, "_readout_batched", perturbed)
    actual = net.scan([batch])[0][0]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            actual["classification_logits"],
            expected["classification_logits"],
            rtol=5e-4,
            atol=5e-5,
        )

    # The oracle still works if all optimized execution/routing/loss entrypoints fail.
    def unavailable(*args, **kwargs):
        raise AssertionError("production helper reached by independent oracle")

    for name in (
        "_routing",
        "_readout_batched",
        "scan",
        "forward",
        "batched_loss_sums",
        "loss_sums",
    ):
        monkeypatch.setattr(net, name, unavailable)
    serial_loss(net, serial_scan(net, [batch])[0][0], batch).backward()


def test_stateless_forward_on_arbitrary_slice_matches_serial_oracle(tmp_path):
    _, events = event_record(tmp_path, normal=True)
    batch = events.batch(2, 6)
    actual, expected = model(), model()
    outputs = actual(batch)
    reference = serial_scan(expected, [batch])[0][0]
    for name in outputs:
        torch.testing.assert_close(outputs[name], reference[name], rtol=5e-4, atol=5e-5)
    actual.losses(outputs, batch)[0].backward()
    serial_loss(expected, reference, batch).backward()
    for p, q in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-2, atol=1e-4)


def test_constructor_typed_and_production_defaults_agree():
    import inspect

    from graphids.core.data.events.data import CANEventDataModule
    from graphids.core.models.event_graph_mamba import EventGraphMamba
    from graphids.exp.config import ExperimentConfig
    from graphids.primitives_data import CANEventDMCfg
    from graphids.primitives_models import EventGraphMambaCfg

    config = ExperimentConfig.from_yaml(
        "configs/experiments/graph_mamba_events_source.yml"
    ).config
    data = CANEventDMCfg(
        manifest_path="unused", datasets=["set_01"], cache_root="unused"
    )
    for name in (
        "chunk_events",
        "stream_batch_size",
        "max_microbatch_events",
        "effective_batch_events",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "cuda_prefetch",
    ):
        assert (
            getattr(data, name)
            == inspect.signature(CANEventDataModule).parameters[name].default
            == config["data"][name]
        )
    assert (
        EventGraphMambaCfg().max_graph_tokens
        == inspect.signature(EventGraphMamba).parameters["max_graph_tokens"].default
        == config["model"]["max_graph_tokens"]
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        CANEventDMCfg(
            manifest_path="unused", datasets=[], cache_root="unused", warmup_events=0
        )
