"""Message preservation, causal graph/Mamba state, and runnable training contracts."""

import json
from dataclasses import asdict, replace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from graphids.core.data.events.data import (
    CANEventDataModule,
    EventChunks,
    EventRecording,
    materialize_events,
)
from graphids.core.data.recordings import (
    CANRecordingManifest,
    iter_rows,
    read_rows,
)
from graphids.core.metrics import raw_average_precision
from graphids.core.models.event_graph_mamba import EventGraphMamba
from graphids.exp.config import ExperimentConfig
from graphids.exp.runtime import build_component
from tests.event_fixtures import write_record

pytestmark = pytest.mark.slurm


def event_record(tmp_path, *, key="frames", normal=False, role="train_query"):
    rec = write_record(
        tmp_path,
        key=key,
        normal=normal,
        role=role,
        rows=[
            (100.0, "0C9", "00010203040506", 0),
            (100.001, "0C9", "FFFFFFFFFFFFFF", 0 if normal else 1),
            (100.001, "0C9", "00010203040506", 0),
            (100.009, "1F5", "00", 0),
            (100.01, "0C9", "00010203040506", 0),
            (100.02, "1F5", "", 0),
        ],
    )
    return rec, EventRecording(materialize_events(rec, tmp_path / "cache"))


def model(backend="reference", **kwargs):
    torch.manual_seed(42)
    return EventGraphMamba(
        width=8,
        layers=2,
        d_state=3,
        d_conv=3,
        expand=1,
        dropout=0,
        backend=backend,
        **kwargs,
    )


def test_all_frames_payload_lengths_tied_timestamps_and_order_survive(tmp_path):
    rec, events = event_record(tmp_path)
    original = read_rows(rec)
    assert list(iter_rows(rec)) == original
    batch = events.batch(0, events.count)
    assert len(batch.ids) == len(original) == 6
    for i, row in enumerate(original):
        length = int(batch.lengths[i])
        assert bytes(batch.payload[i, :length].tolist()) == row["data"]
        assert int(batch.ids[i]) == int(row["id"], 16)
        assert float(batch.timestamps[i]) == row["time"]
        assert int(batch.labels[i]) == row["label"]
    assert batch.labels.tolist() == [0, 1, 0, 0, 0, 0]
    assert batch.timestamps[1] == batch.timestamps[2]
    assert batch.payload[1, 0] == 255  # would be overwritten in a snapshot
    assert batch.payload[2, 0] == 0
    assert batch.lengths.tolist() == [7, 7, 7, 1, 7, 0]


def test_chunks_have_exact_scored_coverage_and_real_preceding_time(tmp_path):
    _, events = event_record(tmp_path)
    data = EventChunks([events], chunk_events=2)
    rows = []
    for i in range(len(data)):
        batch = data[i]
        indices = torch.arange(batch.row_start, batch.row_start + len(batch.ids))
        rows.extend(indices[batch.scored].tolist())
        full = events.batch(0, events.count)
        torch.testing.assert_close(
            batch.times, full.times[batch.row_start : batch.row_start + len(batch.ids)]
        )
    assert rows == list(range(events.count))


def test_unknown_labels_and_unverified_payload_provenance_not_fabricated(tmp_path):
    rec, _ = event_record(tmp_path)
    rec = replace(rec, labels_verified=False, payload_length_verified=False)
    events = EventRecording(materialize_events(rec, tmp_path / "cache"))
    batch = events.batch(0, events.count)
    assert batch.labels.tolist() == [-1] * 6
    net = model()
    output = net(batch)
    loss, parts = net.losses(output, batch)
    assert float(loss) == 0
    assert float(parts["payload"]) == 0
    assert batch.payload[1, 0] == 255  # raw observed bytes still available


def test_fields_are_reversible_and_contain_no_label_or_recording_features(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    net = model()
    fields, _ = net.fields(batch)
    assert fields.shape == (6, 48)
    assert torch.equal((fields[:, :8] * 255).round().byte(), batch.payload)
    bits = fields[:, 16:45].long()
    assert torch.equal((bits * (1 << torch.arange(29))).sum(-1), batch.ids)
    changed = replace(
        batch,
        labels=1 - batch.labels,
        recording="different",
        verified_normal=True,
        source_digest="other",
    )
    torch.testing.assert_close(net.fields(changed)[0], fields)


def test_future_frames_and_new_ids_do_not_change_earlier_scores(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    altered = replace(
        batch,
        payload=batch.payload.clone(),
        ids=batch.ids.clone(),
        times=batch.times.clone(),
        labels=1 - batch.labels,
    )
    altered.payload[3:] = 255
    altered.ids[3:] = 2**29 - 1
    altered.times[3:] += 10
    altered.routing_ids = tuple(int(x) for x in altered.ids)
    altered.routing = None
    net = model().eval()
    a, b = net.scan([batch])[0][0], net.scan([altered])[0][0]
    for name in a:
        torch.testing.assert_close(a[name][:3], b[name][:3])
    # The injected frame actually reaches the model: changing it changes its score.
    changed = replace(batch, payload=batch.payload.clone())
    changed.payload[1] = 0
    assert not torch.allclose(
        net(changed)["classification_logits"][1], a["classification_logits"][1]
    )


def test_payload_prediction_cannot_see_current_payload(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    altered = replace(batch, payload=batch.payload.clone())
    altered.payload[1] = 0
    net = model().eval()
    a, b = net.scan([batch])[0][0], net.scan([altered])[0][0]
    torch.testing.assert_close(a["payload_logits"][:2], b["payload_logits"][:2])
    assert not torch.allclose(a["payload_logits"][2], b["payload_logits"][2])


def test_stream_matches_full_forward_across_arbitrary_chunks(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    net = model().eval()
    expected = net(batch)
    state, outputs = None, []
    for start, stop in ((0, 1), (1, 3), (3, 4), (4, 6)):
        output, state = net.stream(batch.slice(start, stop), state)
        outputs.append(output)
    for name in expected:
        torch.testing.assert_close(
            expected[name], torch.cat([o[name] for o in outputs]), rtol=1e-5, atol=1e-6
        )
    assert state.next_row == 6 and state.ids == (0xC9, 0x1F5)
    # State is serializable, including both Mamba caches and per-ID memory.
    path = tmp_path / "state.pt"
    torch.save(state, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(restored.nodes, state.nodes)
    with pytest.raises(ValueError, match="contiguous"):
        net.stream(batch.slice(4, 6), state)
    with pytest.raises(ValueError, match="another recording"):
        net.stream(replace(batch, recording="other"), state)
    with pytest.raises(ValueError, match="row zero"):
        net.stream(batch.slice(1, 3))


def test_capacity_limits_fail_instead_of_dropping_messages(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    with pytest.raises(ValueError, match="node capacity"):
        model(node_capacity=1)(batch)
    with pytest.raises(ValueError, match="graph capacity"):
        model(max_graph_tokens=1)(batch)


def test_objectives_use_only_scored_labels_and_verified_normal_payloads(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count)
    batch.scored[:2] = False
    net = model()
    a, parts = net.losses(net(batch), batch)
    assert float(parts["payload"]) == 0
    altered = replace(batch, labels=batch.labels.clone())
    altered.labels[:2] = 1 - altered.labels[:2]
    b, _ = net.losses(net(altered), altered)
    torch.testing.assert_close(a, b)
    normal = replace(batch, verified_normal=True)
    loss, parts = net.losses(net(normal), normal)
    assert parts["payload"] > 0
    loss.backward()
    for group in (
        net.input_proj,
        net.temporal,
        net.query,
        net.key,
        net.value,
        net.classifier,
        net.payload_head,
    ):
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in group.parameters()
        )


def source_dm(tmp_path):
    train, _ = event_record(tmp_path, key="train", normal=True)
    val, _ = event_record(tmp_path, key="validation", role="val_query")
    # Distinct content/provenance, independent source sessions.
    path = tmp_path / "manifest.json"
    # A reserved target path must never be opened by the event path.
    target = replace(
        val,
        key="reserved",
        vehicle="target",
        session="reserved",
        role="target_attack_test",
        path="/nonexistent/reserved.csv",
        sha256="f" * 64,
    )
    manifest = CANRecordingManifest((train, val, target), provenance_reviewed=True)
    path.write_text(json.dumps(asdict(manifest)))
    dm = CANEventDataModule(
        manifest_path=str(path),
        datasets=["set_01"],
        cache_root=str(tmp_path / "cache"),
        chunk_events=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    dm.setup()
    assert len(dm.event_recordings) == 2
    assert all(not r.role.startswith("target") for r in dm.manifest.records)
    return dm


def test_lightning_fit_evaluation_coverage_and_checkpoint_round_trip(tmp_path):
    dm = source_dm(tmp_path)
    net = model()
    net.prepare_from_datamodule(dm)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=1,
        default_root_dir=tmp_path / "run",
    )
    trainer.fit(net, datamodule=dm)
    index = json.loads(
        (tmp_path / "run/artifacts/val-events-epoch0000/index.json").read_text()
    )
    assert index["coverage"] == {"validation": 6}
    assert sum(c["count"] for c in index["chunks"]) == 6
    # CONTRACT: Lightning publishes the same exact AP as the saved frame scores.
    shards = sorted((tmp_path / "run/artifacts/val-events-epoch0000").glob("*.npz"))
    blocks = []
    for path in shards:
        with np.load(path) as shard:
            blocks.append(shard["values"])
    values = np.concatenate(blocks)
    for head, column in (("classification", 3), ("payload", 4)):
        selected = (
            np.ones(len(values), dtype=bool)
            if head == "classification"
            else values[:, 5].astype(bool)
        )
        expected = raw_average_precision(values[selected, column], values[selected, 2])
        assert index["metrics"][f"{head}_frame_ap"] == expected
        assert float(trainer.callback_metrics[f"val/{head}_frame_ap"]) == pytest.approx(
            expected
        )
    assert (
        np.load(tmp_path / "run/artifacts/val-events-epoch0000/00000000.npz")[
            "row_start"
        ]
        == 0
    )
    checkpoint = tmp_path / "events.ckpt"
    trainer.save_checkpoint(checkpoint)
    loaded = EventGraphMamba.load_from_checkpoint(checkpoint)
    loaded.prepare_from_datamodule(dm)
    net.eval(), loaded.eval()
    batch = next(iter(dm.val_dataloader()))
    torch.testing.assert_close(
        net(batch)["classification_logits"], loaded(batch)["classification_logits"]
    )
    loaded._source_manifest_digest = "other"
    with pytest.raises(ValueError, match="manifest differs"):
        loaded.prepare_from_datamodule(dm)
    with pytest.raises(ValueError, match="snapshot weights"):
        net.on_load_checkpoint({})


def test_typed_configuration_resolves_and_rejects_mixed_contracts():
    cfg = ExperimentConfig.from_yaml(
        "configs/experiments/graph_mamba_events_source.yml"
    )
    run = cfg.build_run(name="event-check", stage="fit")
    assert run.representation_cfg.kind == "can_event"
    assert run.payload.model["backend"] == "official"
    assert (
        build_component(
            {"type": "event_graph_mamba", "backend": "reference", "width": 8}
        ).hparams.width
        == 8
    )
    changed = dict(cfg.config["data"], representation_cfg={"kind": "can_snapshot"})
    with pytest.raises(ValueError, match="can_event"):
        cfg.build_run(name="bad", stage="fit", config={"data": changed})
    with pytest.raises(ValueError, match="one device"):
        cfg.build_run(name="bad", stage="fit", config={"trainer": {"devices": 2}})
    changed = dict(cfg.config["model"], type="vehicle_graph_mamba_joint")
    with pytest.raises(ValueError):
        cfg.build_run(name="bad", stage="fit", config={"model": changed})


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires pinned official CUDA backend"
)
def test_official_cuda_forward_gradients_and_carried_state(tmp_path):
    _, events = event_record(tmp_path)
    batch = events.batch(0, events.count).to("cuda")
    batch.verified_normal = True
    official, reference = model("official").cuda(), model().cuda()
    reference.load_state_dict(official.state_dict())
    actual, expected = official(batch), reference(batch)
    loss, _ = official.losses(actual, batch)
    ref_loss, _ = reference.losses(expected, batch)
    loss.backward(), ref_loss.backward()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-4, atol=3e-5)
    for (name, p), (_, q) in zip(
        official.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=8e-3, atol=8e-5)
    official.eval()
    with torch.no_grad():
        full = official(batch)
        first, state = official.stream(batch.slice(0, 3))
        second, state = official.stream(batch.slice(3, 6), state)
    for name in full:
        torch.testing.assert_close(
            full[name], torch.cat((first[name], second[name])), rtol=3e-4, atol=3e-5
        )


def test_checkpoint_restore_cannot_replace_prepared_source_identity():
    net = model()
    net._source_manifest_digest = "current"
    with pytest.raises(ValueError, match="manifest differs"):
        net.on_load_checkpoint(
            {
                "event_input_contract": {
                    "schema_version": 1,
                    "source_manifest_digest": "different",
                }
            }
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires official CUDA backend"
)
def test_official_lightning_gpu_fit_and_complete_validation(tmp_path):
    dm = source_dm(tmp_path)
    net = model("official")
    net.prepare_from_datamodule(dm)

    class DeviceCheck(pl.Callback):
        seen = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            assert next(pl_module.parameters()).is_cuda
            assert all(
                b.payload.is_cuda and b.ids.is_cuda for g in batch.groups for b in g
            )
            self.seen += 1

    device_check = DeviceCheck()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=1,
        callbacks=[device_check],
        default_root_dir=tmp_path / "cuda-run",
    )
    trainer.fit(net, datamodule=dm)
    assert device_check.seen == 1
    index = json.loads(
        (tmp_path / "cuda-run/artifacts/val-events-epoch0000/index.json").read_text()
    )
    assert index["coverage"] == {"validation": 6}
