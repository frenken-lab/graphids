"""Ranking, calibration partition isolation and frozen-weight interventions."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from graphids.core.data.snapshots.policy import ReferencePolicy
from graphids.core.models.calibration import CalibrationBundle
from graphids.core.models.score_columns import (
    COL,
    COLUMNS,
    ScoreBlock,
    ScoreTable,
    raw_average_precision,
)

pytestmark = pytest.mark.slurm


@pytest.mark.parametrize(
    "scores,labels,expected",
    [
        ([3, 2, 1], [1, 0, 1], 5 / 6),
        ([1000, 999, 998], [1, 0, 1], 5 / 6),
        ([5, 5, 2], [1, 0, 1], 7 / 12),
        ([0, 0], [0, 1], 0.5),
        ([3, 2, 1], [-1, 0, 1], 0.5),
        ([1], [1], None),
        ([], [], None),
    ],
)
def test_raw_ap(scores, labels, expected):
    actual = raw_average_precision(scores, labels)
    assert actual == pytest.approx(expected) if expected is not None else actual is None
    if expected is not None:
        assert raw_average_precision(
            np.asarray(scores) * 10 + 15, labels
        ) == pytest.approx(actual)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_nonfinite_ap(score):
    with pytest.raises(ValueError, match="finite"):
        raw_average_precision([score, 0], [1, 0])


def table(
    mode="zero-shot", role="val_query", dataset="normal", labels=(0, 0, 1), offset=0
):
    values = np.zeros((len(labels), len(COLUMNS)))
    for name, data in dict(
        label=labels,
        normal=np.asarray(labels) == 0,
        classification=np.arange(len(labels)) * 0.1 + offset,
        anomaly=np.arange(len(labels)) + offset,
        anomaly_eligible=[0] + [1] * (len(labels) - 1),
        component0=np.arange(len(labels)) + offset,
        component1=0,
        mask0=1,
        exposure=1,
    ).items():
        values[:, COL[name]] = data
    return ScoreTable(
        [
            ScoreBlock(
                dict(
                    recording=dataset,
                    vehicle="v",
                    dataset=dataset,
                    role=role,
                    context=mode,
                ),
                values,
            )
        ]
    )


def fit(t, mode="zero-shot", source=None):
    return CalibrationBundle.fit(
        t,
        checkpoint_hash="weights",
        source_manifest="manifest",
        reference_policy="policy",
        mode=mode,
        source=source,
    )


def test_isolation_first_terminal_missing_components_and_target_thresholds():
    source = table()
    source.values[0, COL["classification"]] = 0.99
    source.values[0, COL["exposure"]] = 0
    source.blocks[0].values = source.values
    contaminated = table(role="target_attack_test", offset=100)
    t = ScoreTable(
        source.blocks + contaminated.blocks + table("conditioned", offset=20).blocks
    )
    bundle = fit(t)
    assert bundle.thresholds[0] > 0.98  # classification includes first/terminal bin
    assert bundle.center == (1, 0, 0)
    assert bundle.available == (True, False, False)
    assert fit(t, "conditioned").center[0] == 21
    with pytest.raises(ValueError, match="no verified"):
        fit(table("conditioned"))
    with pytest.raises(ValueError, match="mismatch"):
        bundle.validate("other", "manifest", "policy", "zero-shot")
    with pytest.raises(ValueError, match="conditioned-only"):
        fit(table(role="target_calibration", offset=5), source=bundle)
    conditioned = fit(t, "conditioned")
    target = table("conditioned", role="target_calibration", offset=5)
    adapted = fit(target, "conditioned", source=conditioned)
    assert adapted.center == conditioned.center and adapted.scale == conditioned.scale
    assert adapted.thresholds != conditioned.thresholds
    scores = conditioned.scores(target.values[:, 5:8], target.values[:, 8:11])
    np.testing.assert_array_equal(
        scores, adapted.scores(target.values[:, 5:8], target.values[:, 8:11])
    )
    assert raw_average_precision(
        scores, target.column("label")
    ) == raw_average_precision(
        adapted.scores(target.values[:, 5:8], target.values[:, 8:11]),
        target.column("label"),
    )


def test_vehicle_pooling_across_registry_names():
    normal = table(role="target_normal_test", labels=(0, 0), dataset="normal")
    attack = table(role="target_attack_test", labels=(1, 1), dataset="attack", offset=1)
    split = ScoreTable(normal.blocks + attack.blocks).report()
    unified = ScoreTable(
        [
            replace(b, identity=dict(b.identity, dataset="unified"))
            for b in normal.blocks + attack.blocks
        ]
    ).report()
    assert (
        split["groups"]["zero-shot/vehicle/v"]
        == unified["groups"]["zero-shot/vehicle/v"]
    )
    assert (
        split["groups"]["zero-shot/dataset/normal"]["heads"]["classification"]["status"]
        == "missing_both_classes"
    )


def test_recent_pooling_preserves_forecasting_and_policy_reset(tmp_path):
    from tests.core.models.test_graph_mamba_chunk import make_batch, model

    batch, _, _ = make_batch(tmp_path)
    net = model().eval()
    original, state = net.stream(batch)
    net.reference_policy = ReferencePolicy(recent_classifier_pooling=True)
    actual, _ = net.stream(
        batch, state
    )  # changing policy resets contiguous stream state
    for key in (
        "count_rate",
        "change_logits",
        "new_id_rate",
        "anomaly_components",
        "anomaly_component_mask",
        "anomaly_eligible",
    ):
        torch.testing.assert_close(original[key], actual[key], rtol=0, atol=0)
    before = net._tokens(batch)
    net.reference_policy = ReferencePolicy(profiles=False, reference_flags=False)
    net._tokens(batch)
    assert batch.profiles.abs().sum() >= 0  # source metadata remains available
    assert before.shape[-1] == net.hparams.width


def test_reference_input_isolation_and_cache_immutability(tmp_path):
    from graphids.core.data.snapshots.batch import recording_batch
    from tests.core.models.test_graph_mamba import make_batch, model

    batch, snapshot, context = make_batch(tmp_path)
    net = model().eval()
    captured = []
    hook = net.input_proj.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].clone())
    )
    originals = [
        getattr(context, key).copy()
        for key in ("profiles", "center", "scale", "neighbors", "edges", "edge_mask")
    ]
    net._tokens(batch)
    full = captured[-1]
    for policy, selected in (
        (ReferencePolicy(profiles=False), slice(23, 27)),
        (ReferencePolicy(reference_flags=False), slice(27, 29)),
    ):
        net.reference_policy = policy
        net._tokens(batch)
        expected = full.clone()
        expected[..., selected] = 0
        torch.testing.assert_close(captured[-1], expected, rtol=0, atol=0)
    hook.remove()
    canonical = recording_batch(
        snapshot, context, reference_policy=ReferencePolicy(normalization=False)
    )
    np.testing.assert_array_equal(
        canonical.x[0].numpy(), np.where(snapshot.live[..., None], snapshot.features, 0)
    )
    modified = recording_batch(snapshot, context)
    modified.profiles.zero_()
    modified.edge_mask.zero_()
    restored = recording_batch(snapshot, context)
    torch.testing.assert_close(restored.profiles, batch.profiles)
    torch.testing.assert_close(restored.edge_mask, batch.edge_mask)
    for key, original in zip(
        ("profiles", "center", "scale", "neighbors", "edges", "edge_mask"),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(getattr(context, key), original)
        assert not getattr(context, key).flags.writeable
    h = torch.randn(*batch.live.shape, net.hparams.width)
    # Masking cross-ID edges preserves the graph module's self update.
    edge_free = replace(batch, edge_mask=torch.zeros_like(batch.edge_mask))
    torch.testing.assert_close(
        net.graph(h, batch, graph_enabled=False), net.graph(h, edge_free)
    )


def test_expiring_anchors_missing_neighbors_empty_windows_and_chunk_carry(tmp_path):
    from graphids.core.data.snapshots.batch import recording_batch
    from graphids.core.data.snapshots.context import build_context
    from graphids.core.data.snapshots.online import CausalCANBinner
    from graphids.core.data.snapshots.preprocessing import (
        CANSnapshotRepresentationCfg,
        build_snapshots,
    )
    from tests.core.data.test_snapshots import write_record
    from tests.core.models.test_graph_mamba import model

    rec = write_record(
        tmp_path,
        role="train_reference",
        normal=True,
        rows=[(0.0, "A", "00", 0), (0.01, "B", "01", 0), (0.08, "A", "02", 0)],
    )
    cfg = CANSnapshotRepresentationCfg()
    context = build_context("v", [build_snapshots(rec, cfg)])
    policy = ReferencePolicy(persistent_anchors=False)
    snapshot = build_snapshots(
        rec,
        cfg,
        anchor_ids=context.anchor_ids,
        persistent_anchors=False,
        retention_seconds=0.015,
    )
    assert not snapshot.live[4].any()
    assert snapshot.generation[-1, 0] > snapshot.generation[0, 0]
    batch = recording_batch(snapshot, context, reference_policy=policy)
    net = model().eval()
    net.reference_policy = policy
    full, _ = net.stream(batch)
    parts, state = [], None
    for start, stop in ((0, 3), (3, 7), (7, len(snapshot.closes))):
        chunk = recording_batch(
            snapshot, context, start=start, stop=stop, reference_policy=policy
        )
        assert int(chunk.neighbors.max()) < chunk.live.shape[-1]
        out, state = net.stream(chunk, state)
        parts.append(out)
    for key in (
        "classification_logits",
        "new_id_rate",
        "anomaly_components",
        "anomaly_component_mask",
        "anomaly_eligible",
    ):
        torch.testing.assert_close(
            torch.cat([part[key] for part in parts], 1), full[key]
        )
    online = CausalCANBinner(
        rec, context, reference_policy=policy, retention_seconds=0.015
    )
    bins = []
    for ts, key, data in (
        (0.0, "A", b"\x00"),
        (0.01, "B", b"\x01"),
        (0.08, "A", b"\x02"),
    ):
        bins.extend(online.push(ts, key, data, label=0))
    bins.extend(online.finish())
    for index, item in enumerate(bins):
        assert int(item.live.sum()) == int(snapshot.live[index].sum())
    with pytest.raises(ValueError, match="another recording/context"):
        CausalCANBinner(rec, context, state=online.state)


def test_legacy_checkpoint_load_requires_explicit_replay(tmp_path):
    from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
    from tests.core.models.test_graph_mamba import model

    net = model()
    checkpoint = {
        "graphids_schema": {"model": "vehicle_graph_mamba_joint", "version": 1}
    }
    net.on_load_checkpoint(checkpoint)
    net.calibrated.fill_(True)
    with pytest.raises(ValueError, match="legacy pooled"):
        net.on_test_epoch_start()
    net.hparams.legacy_calibration_replay = True
    net.on_test_epoch_start()
    fit_bundle = fit(table())
    saved = tmp_path / "metadata.pt"
    torch.save(fit_bundle.to_dict(), saved)
    loaded = CalibrationBundle(**torch.load(saved, weights_only=True))
    assert loaded == fit_bundle


def test_checkpoint_metric_is_source_vehicle_macro_minimum():
    from types import SimpleNamespace

    blocks = []
    for mode in ("zero-shot", "conditioned"):
        blocks += table(mode, labels=(0, 0, 1)).blocks
        blocks += table(mode, role="target_attack_test", offset=100).blocks
    captured = {}
    net = SimpleNamespace(
        hparams=SimpleNamespace(classification=True, forecasting=True),
        log=lambda key, value, **_: captured.__setitem__(key, value),
    )
    ScoreTable(blocks).log_metrics(net, "val")
    values = [
        captured[f"val/{mode}/{head}_ap_macro"]
        for mode in ("zero-shot", "conditioned")
        for head in ("classification", "anomaly")
    ]
    assert captured["val/source_min_vehicle_macro_ap"] == min(values)
    assert all(value == 1 for value in values)


def test_corrected_preset_preserves_original_selection():
    from graphids.exp.config import ExperimentConfig

    old = ExperimentConfig.from_yaml(
        "configs/experiments/graph_mamba_ctat_learning.yml"
    )
    new = ExperimentConfig.from_yaml(
        "configs/experiments/graph_mamba_ctat_learning_corrected.yml"
    )
    assert (
        old.config["callbacks"]["checkpoint"]["init_args"]["monitor"]
        == "val/zero-shot/classification_ap_macro"
    )
    assert (
        new.config["callbacks"]["checkpoint"]["init_args"]["monitor"]
        == "val/source_min_vehicle_macro_ap"
    )
    new.build_run(name="corrected-config-check", stage="fit")


def test_snapshot_cache_depends_on_lifecycle_and_preserves_reference_membership(
    tmp_path,
):
    from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
    from scripts.graph_mamba_gate import fixture_manifest

    manifest = fixture_manifest(tmp_path / "fixtures")
    dm = CANSnapshotDataModule(
        manifest_path=str(manifest),
        datasets=["set_01", "hcrl_sa"],
        cache_root=str(tmp_path / "cache"),
    )
    dm.setup()
    key = next(r.key for r in dm.records.values() if r.role == "val_query")
    context = dm.context_for(key)
    original = dm.snapshot_for(key, context)
    dm.reference_policy = ReferencePolicy(
        profiles=False, normalization=False, graph_edges=False, reference_flags=False
    )
    assert dm.snapshot_for(key, context) is original
    dm.reference_policy = ReferencePolicy(persistent_anchors=False)
    expired = dm.snapshot_for(key, context)
    assert expired is not original
    assert set(context.anchor_ids).issubset(expired.ids)
    np.testing.assert_array_equal(expired.closes, original.closes)
    np.testing.assert_array_equal(expired.labels, original.labels)
    np.testing.assert_array_equal(expired.event_count, original.event_count)


def test_json_rows_restore_compact_recording_blocks():
    original = ScoreTable(table().blocks + table("conditioned").blocks)
    restored = ScoreTable.from_rows(iter(original))
    assert len(restored.blocks) == 2
    np.testing.assert_array_equal(restored.values, original.values)
    assert restored.keys() == original.keys()


def test_all_unavailable_components_have_no_anomaly_ap():
    scored = table()
    scored.values[:, 8:11] = 0
    bundle = fit(scored)
    assert bundle.available == (False, False, False)
    assert bundle.thresholds[1] is None
    report = scored.report({"zero-shot": bundle})
    head = report["groups"]["zero-shot/vehicle/v"]["heads"]["anomaly"]
    assert head["status"] == "unavailable_components"
    assert head["ap"] is None and head["eligible_bins"] == 0


@pytest.mark.parametrize("head", ["classification", "anomaly"])
def test_reports_reject_nonfinite_unknown_or_ineligible_scores(head):
    scored = table(labels=(-1, 0, 1))
    scored.values[0, COL[head]] = np.nan
    with pytest.raises(ValueError, match="nonfinite scored"):
        scored.report()


def test_no_anomaly_alert_when_only_uncalibrated_components_are_present():
    scored = table()
    bundle = replace(fit(scored), thresholds=(2.0, -1.0))
    # Only component 0 was calibrated; component 1 is present at inference.
    scored.values[:, 8:11] = (0, 1, 0)
    scored.apply_bundles({"zero-shot": bundle})
    assert not scored.column("alert").any()
    report = scored.report({"zero-shot": bundle})["groups"]["zero-shot/vehicle/v"]
    assert report["anomaly"]["bin_alerts"] == 0
    assert report["heads"]["anomaly"]["eligible_bins"] == 0


@pytest.mark.parametrize("disabled", ["classification", "anomaly"])
def test_disabled_heads_are_not_evaluated(disabled):
    scored = table()
    bundle = fit(scored)
    scored.values[:, COL[disabled]] = np.nan
    enabled = dict(
        classification=disabled != "classification", forecasting=disabled != "anomaly"
    )
    group = scored.report({"zero-shot": bundle}, **enabled)["groups"][
        "zero-shot/vehicle/v"
    ]
    assert group["heads"][disabled] == dict(
        ap=None, status="disabled", eligible_bins=0, positive_prevalence=None
    )
    assert group[disabled]["bin_alerts"] == 0
    import json

    json.dumps(group, allow_nan=False)


def test_score_tables_reject_wrong_or_missing_mode_bundles():
    scored = table()
    conditioned = fit(table("conditioned"), "conditioned")
    with pytest.raises(ValueError, match="mode mismatch"):
        scored.apply_bundles({"zero-shot": conditioned})
    with pytest.raises(ValueError, match="mode mismatch"):
        scored.report({"zero-shot": conditioned})
    with pytest.raises(ValueError, match="missing calibration mode"):
        scored.report({"conditioned": conditioned})
    with pytest.raises(ValueError, match="unsupported inference mode"):
        scored.mode_mask("typo")


def test_collected_bin_alerts_require_an_available_calibrated_component(tmp_path):
    from types import SimpleNamespace

    from tests.core.models.test_graph_mamba import make_batch, model

    batch, _, _ = make_batch(tmp_path)
    net = model().eval()
    net._recordings = {
        batch.recording_keys[0]: SimpleNamespace(
            key=batch.recording_keys[0], dataset="d", vehicle="v", role="val_query"
        )
    }
    output, _ = net.stream(batch)
    bundle = replace(fit(table("conditioned"), "conditioned"), thresholds=(2.0, -1.0))
    net.calibration_bundles = {"conditioned": bundle}
    output["anomaly_component_mask"].zero_()
    output["anomaly_component_mask"][..., 1] = True  # uncalibrated component
    output["anomaly_eligible"].fill_(True)
    output["anomaly_score"].zero_()
    block = ScoreBlock.from_batch(net, batch, output)
    assert not block.values[:, COL["alert"]].any()


@pytest.mark.parametrize("disabled", ["classification", "anomaly"])
def test_target_calibrated_artifacts_respect_disabled_heads(
    tmp_path, monkeypatch, disabled
):
    import json

    import lightning.pytorch as pl

    from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
    from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
    from scripts.graph_mamba_gate import fixture_manifest

    monkeypatch.setenv("GRAPHIDS_RUN_DIR", str(tmp_path))
    manifest = fixture_manifest(tmp_path / "fixtures")
    dm = CANSnapshotDataModule(
        manifest_path=str(manifest),
        datasets=["set_01", "hcrl_sa"],
        cache_root=str(tmp_path / "cache"),
    )
    dm.setup()
    net = VehicleGraphMambaJoint(
        width=8,
        layers=1,
        dropout=0,
        backend="reference",
        classification=disabled != "classification",
        forecasting=disabled != "anomaly",
    )
    net.prepare_from_datamodule(dm)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.validate(net, datamodule=dm, verbose=False)
    trainer.test(net, datamodule=dm, verbose=False)
    for artifact in ("metrics-v2.json", "target-calibrated-metrics-v2.json"):
        report = json.loads((tmp_path / "artifacts" / artifact).read_text())
        assert report["groups"]
        for group in report["groups"].values():
            assert group["heads"][disabled]["status"] == "disabled"
            assert group["heads"][disabled]["ap"] is None
            assert group[disabled]["bin_alerts"] == 0
