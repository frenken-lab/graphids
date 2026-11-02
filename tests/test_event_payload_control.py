"""Independent categorical-count oracle, causal backoff and cohort checks."""

from collections import Counter, defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from graphids.data.context import previous_rows
from scripts.event_causal_controls import evaluate, markdown
from scripts.event_payload_control import (
    byte_probabilities,
    fit_payload_profiles,
    payload_scores,
    save_profiles,
)
from tests.test_event_failure_report import events, report_fixture


def serial_counts(records):
    marginal, pairs = defaultdict(Counter), defaultdict(Counter)
    for record in records:
        latest = {}
        for key, payload, length in zip(
            record.arrays["ids"],
            record.arrays["payload"],
            record.arrays["lengths"],
            strict=True,
        ):
            for position in range(length):
                marginal[(record.recording.vehicle, int(key), position)][
                    int(payload[position])
                ] += 1
                if key in latest and position < len(latest[key]):
                    pairs[
                        (
                            record.recording.vehicle,
                            int(key),
                            position,
                            latest[key][position],
                        )
                    ][int(payload[position])] += 1
            latest[key] = list(map(int, payload[:length]))
    return marginal, pairs


def training():
    a = events(
        "a",
        [10, 11, 10, 10, 10, 10],
        [0, 9, 1, 0, 2, 0],
        normal=True,
        lengths=[2, 1, 1, 2, 2, 1],
    )
    a.arrays["payload"][:, 1] = [4, 0, 0, 5, 4, 0]
    b = events("b", [10, 10, 10], [255, 255, 254], normal=True, vehicle="b")
    # A second capture never inherits a predecessor from the first.
    a2 = events("a2", [10, 10], [7, 1], normal=True)
    return [a, b, a2]


def test_predictive_distributions_match_serial_counts_and_normalize():
    records = training()
    marginal, pairs = serial_counts(records)
    profiles = fit_payload_profiles(records, block=2)
    current = np.arange(256)
    for vehicle, key, position in marginal:
        for prior in (-1, 0, 1, 7, 128):
            probabilities = byte_probabilities(
                profiles[vehicle],
                np.full(256, key * 8 + position),
                np.full(256, prior),
                current,
            )
            mcounts, pcounts = (
                marginal[(vehicle, key, position)],
                pairs[(vehicle, key, position, prior)],
            )
            expected_m = np.array(
                [(mcounts[b] + 1 / 256) / (sum(mcounts.values()) + 1) for b in current]
            )
            expected_t = np.array(
                [
                    (pcounts[b] + 32 * expected_m[b]) / (sum(pcounts.values()) + 32)
                    for b in current
                ]
            )
            np.testing.assert_allclose(probabilities[0], expected_m, rtol=1e-14)
            np.testing.assert_allclose(probabilities[1], expected_t, rtol=1e-14)
            assert probabilities[0].sum() == pytest.approx(1)
            assert probabilities[1].sum() == pytest.approx(1)
            assert (probabilities[0] > 0).all() and (probabilities[1] > 0).all()
    # Numeric ID 10 has unrelated byte distributions on the two vehicles.
    assert (
        byte_probabilities(
            profiles["v"], np.array([80]), np.array([-1]), np.array([255])
        )[0][0]
        < byte_probabilities(
            profiles["b"], np.array([80]), np.array([-1]), np.array([255])
        )[0][0]
    )
    assert profiles["v"]["transition_counts"].sum() == 6


def test_frame_scores_match_independent_serial_oracle_with_variable_lengths():
    records = training()
    marginal, pairs = serial_counts(records)
    profiles = fit_payload_profiles(records)
    query = events(
        "q",
        [10, 11, 10, 10, 10, 99, 10, 10],
        [1, 9, 0, 2, 1, 0, 0, 1],
        role="val_query",
        labels=[0, 0, -1, 1, 0, 1, 1, 0],
        lengths=[1, 2, 1, 2, 0, 1, 3, 1],
    )
    result = payload_scores(query, profiles, block=2)
    latest = {}
    for row, (key, payload, length) in enumerate(
        zip(
            query.arrays["ids"],
            query.arrays["payload"],
            query.arrays["lengths"],
            strict=True,
        )
    ):
        supported = length > 0 and all(("v", key, p) in marginal for p in range(length))
        assert result["byte_supported"][row] == supported
        mscore, tscore, contexts = 0.0, 0.0, 0
        if supported:
            for position in range(length):
                m = marginal[("v", key, position)]
                value = int(payload[position])
                pm = (m[value] + 1 / 256) / (sum(m.values()) + 1)
                prior = (
                    latest[key][position]
                    if key in latest and position < len(latest[key])
                    else -1
                )
                t = pairs[("v", key, position, prior)]
                pt = (t[value] + 32 * pm) / (sum(t.values()) + 32)
                mscore -= np.log(pm) / length
                tscore -= np.log(pt) / length
                contexts += bool(t)
        assert result["byte_marginal"][row] == pytest.approx(mscore)
        assert result["byte_transition"][row] == pytest.approx(tscore)
        assert result["byte_scored_bytes"][row] == (length if supported else 0)
        assert result["byte_context_bytes"][row] == contexts
        latest[key] = list(map(int, payload[:length]))
    assert result["byte_supported"].tolist() == [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
    ]


def test_scoring_is_causal_label_independent_and_block_invariant():
    records = training()
    profiles = fit_payload_profiles(records, block=1)
    unchanged = deepcopy(profiles)
    query = events(
        "q", [10] * 6, [7, 1, 0, 2, 0, 1], role="val_query", labels=[0, -1, 1, 0, 1, 0]
    )
    whole = payload_scores(query, profiles, block=99)
    for name, values in whole.items():
        np.testing.assert_array_equal(
            payload_scores(query, profiles, block=1)[name], values
        )
    # Unknown-label row 1 supplies the previous value for the next observation.
    query.arrays["labels"][:] = 1
    for name, values in whole.items():
        np.testing.assert_array_equal(payload_scores(query, profiles)[name], values)
    query.arrays["payload"][4:] = 255
    query.arrays["ids"][4:] = 99
    query.previous_rows = previous_rows(query.arrays["ids"])
    for name, values in whole.items():
        np.testing.assert_array_equal(
            payload_scores(query, profiles)[name][:4], values[:4]
        )
    assert whole["byte_marginal"][0] == whole["byte_transition"][0]
    assert whole["byte_backoff"][0]
    for vehicle in profiles:
        for name in profiles[vehicle]:
            np.testing.assert_equal(profiles[vehicle][name], unchanged[vehicle][name])


def test_fitting_skips_untrusted_arrays_and_unverified_payloads():
    forbidden = [
        SimpleNamespace(
            recording=SimpleNamespace(
                role=role, verified_normal=normal, payload_length_verified=verified
            )
        )
        for role, normal, verified in (
            ("val_query", True, True),
            ("target_reference", True, True),
            ("train_query", False, True),
            ("train_query", True, False),
        )
    ]
    trusted = training()[0]
    result = fit_payload_profiles([*forbidden, trusted])
    assert set(result) == {"v"}
    assert result["v"]["recordings"] == ["a"]
    with pytest.raises(ValueError, match="payload training"):
        fit_payload_profiles(forbidden)
    trusted.arrays["labels"][0] = -1
    with pytest.raises(ValueError, match="nonnormal"):
        fit_payload_profiles([trusted])
    for vehicle, verified in (("v", False), ("unprofiled", True)):
        query = SimpleNamespace(
            count=4,
            recording=SimpleNamespace(
                vehicle=vehicle, payload_length_verified=verified
            ),
        )
        assert all(not value.any() for value in payload_scores(query, result).values())


def test_profiles_preserve_29_bit_ids_and_reload_exactly(tmp_path):
    trusted = events(
        "wide", [2**29 - 1] * 3, [255, 254, 255], normal=True, lengths=[8] * 3
    )
    trusted.arrays["payload"][:, 7] = [254, 255, 255]
    profiles = fit_payload_profiles([trusted])
    summaries = save_profiles(profiles, tmp_path)
    assert summaries["v"]["training_bytes"] == 24
    assert summaries["v"]["training_pairs"] == 16
    assert profiles["v"]["transition_keys"].max() > 2**32
    with np.load(tmp_path / summaries["v"]["file"]) as restored:
        for name in restored.files:
            np.testing.assert_array_equal(restored[name], profiles["v"][name])
        query = events("q", [2**29 - 1] * 2, [0, 255], role="val_query", lengths=[8, 8])
        for name, values in payload_scores(query, profiles).items():
            np.testing.assert_array_equal(
                payload_scores(query, {"v": restored})[name], values
            )


def test_report_compares_identical_byte_support_and_preserves_denominators(tmp_path):
    dm = report_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    report = evaluate(
        dm, tmp_path, min_support=2, output=output, payload_transition=True
    )
    assert report["coverage"] == dict(normal=4, weak=6)
    assert report["payload_control"]["profiles"]["v"]["recordings"] == ["train"]
    weak = report["by_recording"]["weak"]
    for name in ("byte_marginal", "byte_transition"):
        assert weak["detectors"][name]["supported_metrics"]["frames"] == 4
        assert weak["detectors"][name]["supported_metrics"]["positives"] == 3
        assert weak["detectors"][name]["at_normal_budgets"][0]["positives"] == 4
        assert (
            weak["detectors"][name]["at_normal_budgets"][0]["unsupported_positives"]
            == 1
        )
        assert (
            weak["strata"]["byte_supported"]["detectors"]["original"][
                "supported_metrics"
            ]["frames"]
            == 4
        )
    assert weak["payload_control_bytes"]["scored"] == 5  # includes the unknown row
    assert report["saved_metric_parity"]["passed"]
    assert "Mean NLL" in markdown(report)
    artifact = report["control_artifacts"]["weak"]
    with np.load(output / artifact["file"]) as values:
        assert values["byte_supported"].sum() == 5
