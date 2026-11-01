"""Classifier-only objective, causal features and paired exposure contracts."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graphids.data.representation import CANEventRepresentationConfig
from graphids.evaluation import iter_score_chunks
from graphids.models.event_graph_mamba import EventGraphMambaState
from scripts.event_classifier_pilot import (
    FrozenReadout,
    evaluate,
    fingerprint,
    fit_heads,
    head_loss,
    positive_weights,
)
from tests.event_fixtures import source_dm, write_record
from tests.test_event_context import context_model


def counts_record(key, labels, role="train_query"):
    return SimpleNamespace(
        recording=SimpleNamespace(key=key, role=role),
        arrays={"labels": np.array(labels)},
        count=len(labels),
    )


def test_weights_have_final_cap_preserve_positive_mass_and_ignore_validation():
    records = [
        counts_record("large", [1] * 1000 + [0, -1]),
        counts_record("small", [1, 0]),
        counts_record("normal", [0] * 10),
    ]
    validation = SimpleNamespace(recording=SimpleNamespace(role="val_query"))
    groups = positive_weights([*records, validation], cap=10)
    assert groups["small"]["positive_weight"] == 10
    assert groups["normal"]["positive_weight"] == 1
    assert groups["large"]["unknown"] == 1
    assert sum(
        g["positives"] * g["positive_weight"] for g in groups.values()
    ) == pytest.approx(1001)
    assert all(0 < g["positive_weight"] <= 10 for g in groups.values())
    assert all(
        g["positive_weight"] == 1 for g in positive_weights(records, cap=1).values()
    )


@pytest.mark.parametrize("cap", [0.5, float("inf"), float("nan")])
def test_invalid_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="cap"):
        positive_weights([], cap)


def test_loss_uses_window_count_and_keeps_negatives_and_unknown_mask():
    batch = SimpleNamespace(
        labels=torch.tensor([1, 0, -1, 1]),
        scored=torch.tensor([True, True, True, False]),
    )
    logits = torch.tensor([0.3, -0.4, 10.0, -10.0], requires_grad=True)
    loss = head_loss(logits, batch, positive_weight=7, denominator=8)
    # Independent scalar logistic calculation; do not cancel single-group weight.
    expected = (7 * np.logaddexp(0, -0.3) + np.logaddexp(0, -0.4)) / 8
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert logits.grad[0] == pytest.approx(
        7 * (torch.sigmoid(logits[0]).item() - 1) / 8
    )
    assert logits.grad[1] == pytest.approx(torch.sigmoid(logits[1]).item() / 8)
    assert logits.grad[2:].tolist() == [0, 0]


def pilot_fixture(tmp_path):
    dm = source_dm(
        tmp_path, representation_cfg=CANEventRepresentationConfig(schema_version=2)
    )
    path = Path(dm.manifest_path)
    manifest = json.loads(path.read_text())
    for key, positives in (("rare", 1), ("common", 4)):
        rec = write_record(
            tmp_path,
            key=key,
            rows=[(0.01 * i, "0C9", f"{i:02X}", int(i < positives)) for i in range(5)],
        )
        manifest["records"].append(asdict(rec))
    path.write_text(json.dumps(manifest))
    dm.event_recordings.clear()
    dm.effective_batch_events = 4  # Includes optimizer-window and recording tails.
    dm.setup()
    net = context_model(dm, dropout=0.5)
    return dm, net


def test_frozen_features_match_stream_at_chunk_boundaries(tmp_path):
    dm, net = pilot_fixture(tmp_path)
    net.eval().requires_grad_(False)
    rec = dm.event_recordings["validation"]
    state = EventGraphMambaState()
    with FrozenReadout(net) as readout:
        for lo, hi in ((0, 2), (2, 6)):
            batch = rec.batch(lo, hi)
            expected, expected_state = net.stream(batch, state)
            outputs, states, features = readout.scan([batch], [state])
            torch.testing.assert_close(
                outputs[0]["classification_logits"],
                expected["classification_logits"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                net.classifier(features[0]).squeeze(-1),
                expected["classification_logits"],
                rtol=0,
                atol=0,
            )
            assert not features.requires_grad
            assert states[0].next_row == expected_state.next_row == hi
            state = states[0]
    assert not net.classifier._forward_pre_hooks


def test_paired_fit_freezes_model_preserves_coverage_and_common_cap_control(tmp_path):
    dm, net = pilot_fixture(tmp_path)
    before = fingerprint(net)
    initial = deepcopy(net.classifier.state_dict())
    heads, training = fit_heads(net, dm, epochs=2, cap=1)
    assert fingerprint(net) == before
    assert not net.training
    assert not any(p.requires_grad or p.grad is not None for p in net.parameters())
    assert [e["steps"] for e in training["epochs"]] == [5, 5]
    assert all(
        e["coverage"] == dict(train=6, rare=5, common=5) for e in training["epochs"]
    )
    for key, value in heads["ordinary"].state_dict().items():
        torch.testing.assert_close(
            value, heads["balanced"].state_dict()[key], rtol=0, atol=0
        )
    assert not torch.equal(initial["weight"], heads["ordinary"].weight)
    # Both trained arms must receive exactly the same evaluation cohort/payload.
    results = evaluate(net, heads, dm, tmp_path / "scores")
    assert fingerprint(net) == before
    assert all(r["coverage"] == dict(validation=6) for r in results.values())
    original = np.concatenate(
        [v for _, v in iter_score_chunks(tmp_path / "scores/original")]
    )
    for name in heads:
        values = np.concatenate(
            [v for _, v in iter_score_chunks(tmp_path / "scores" / name)]
        )
        np.testing.assert_array_equal(
            values[:, [0, 1, 2, 4, 5]], original[:, [0, 1, 2, 4, 5]]
        )
        assert (
            results[name]["recording_macro_ap"]
            == results[name]["metrics"]["classification_frame_ap"]
        )


def test_balancing_changes_only_head_and_hooks_are_removed_on_error(tmp_path):
    dm, net = pilot_fixture(tmp_path)
    before = fingerprint(net)
    heads, _ = fit_heads(net, dm, epochs=1, cap=10)
    assert fingerprint(net) == before
    assert not torch.equal(heads["ordinary"].weight, heads["balanced"].weight)
    rec = dm.event_recordings["train"]
    with (
        pytest.raises(ValueError, match="physical row zero"),
        FrozenReadout(net) as readout,
    ):
        readout.scan([replace(rec.batch(0, 2), row_start=1)], [EventGraphMambaState()])
    assert not net.classifier._forward_pre_hooks
