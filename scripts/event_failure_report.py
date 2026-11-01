"""Source-only diagnosis from a completed classifier pilot; no model execution.

python -m scripts.event_failure_report --pilot PILOT_ROOT --focus FOCUS.json \
    --output NEW_DIRECTORY

FOCUS maps explicit validation recording keys to lists of matched training keys.
These are reviewed reporting annotations, never inferred model/family features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from graphids.evaluation import iter_score_chunks
from graphids.metrics import frame_metrics

ARMS = ("original", "ordinary", "balanced")
BUDGETS = (0.0, 1e-5, 1e-4, 1e-3)
BLOCK = 65536


def quantiles(values):
    return (
        dict(
            zip(
                ("min", "p10", "median", "p90", "p99", "max"),
                map(float, np.quantile(values, [0, 0.1, 0.5, 0.9, 0.99, 1])),
                strict=True,
            )
        )
        if len(values)
        else None
    )


def budget_point(scores, supported, rate):
    """Conservative tied-score threshold on a common normal-frame denominator.

    Unsupported rows never trigger; they remain in total normal exposure.
    The threshold is diagnostic, fitted on selection-validation negatives.
    """
    scores, supported = np.asarray(scores), np.asarray(supported, dtype=bool)
    if not len(scores) or supported.shape != scores.shape or not 0 <= rate < 1:
        raise ValueError("invalid normal score population/budget")
    values = scores[supported]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("budget needs finite supported normal scores")
    allowed = math.floor(rate * len(scores))
    if allowed >= len(values):
        threshold = float(values.min())
    else:
        index = len(values) - allowed - 1
        threshold = float(np.nextafter(np.partition(values, index)[index], np.inf))
    achieved = int((values >= threshold).sum())
    return dict(
        requested_fpr=rate,
        normal_frames=len(scores),
        supported_normal_frames=len(values),
        allowed_fp=allowed,
        achieved_fp=achieved,
        achieved_fpr=achieved / len(scores),
        threshold=threshold,
    )


def decisions(scores, supported, labels, point):
    positive, negative = labels == 1, labels == 0
    predicted = supported & (scores >= point["threshold"])
    tp, fp = int((predicted & positive).sum()), int((predicted & negative).sum())
    return dict(
        tp=tp,
        fp=fp,
        positives=int(positive.sum()),
        negatives=int(negative.sum()),
        unsupported_positives=int((positive & ~supported).sum()),
        recall=tp / positive.sum() if positive.any() else None,
        fpr=fp / negative.sum() if negative.any() else None,
    )


def byte_keys(events, lo, hi, *, transition=False):
    arrays = events.arrays
    ids, payload = arrays["ids"][lo:hi], arrays["payload"][lo:hi]
    positions = np.arange(8)[None]
    valid = positions < arrays["lengths"][lo:hi, None]
    valid &= events.recording.payload_length_verified
    keys = (ids[:, None] * 8 + positions) * 256 + payload
    if transition:
        previous = events.previous_rows[lo:hi]
        prior = previous.clip(0)
        valid &= (previous >= 0)[:, None] & (positions < arrays["lengths"][prior, None])
        keys = (
            (ids[:, None] * 8 + positions) * 256 + arrays["payload"][prior]
        ) * 256 + payload
    return keys, valid


def fit_profiles(recordings):
    """Marginal bytes and preceding same-ID byte pairs, normal TRAIN QUERY only.

    Keep vehicle-scoped profiles primary; pooled numeric IDs are a sensitivity
    check because identical IDs across vehicles need not have identical meanings.
    """
    profiles = {}
    for events in recordings:
        rec = events.recording
        if rec.role != "train_query" or not rec.verified_normal:
            continue
        profile = profiles.setdefault(
            rec.vehicle, dict(ids=set(), values=set(), transitions=set())
        )
        profile["ids"].update(map(int, np.unique(events.arrays["ids"])))
        for lo in range(0, events.count, BLOCK):
            hi = min(events.count, lo + BLOCK)
            for name, transition in (("values", False), ("transitions", True)):
                keys, valid = byte_keys(events, lo, hi, transition=transition)
                profile[name].update(map(int, np.unique(keys[valid])))
    if not profiles:
        raise ValueError("normal training recordings required")
    profiles["pooled"] = {
        name: set().union(*(p[name] for p in profiles.values()))
        for name in ("ids", "values", "transitions")
    }
    return {
        scope: {
            name: np.array(sorted(values), dtype=np.int64)
            for name, values in profile.items()
        }
        for scope, profile in profiles.items()
    }


def payload_features(events, profile):
    result = {
        name: np.zeros(events.count, dtype=bool)
        for name in (
            "normal_id_seen",
            "payload_available",
            "unseen_position",
            "novel_value",
            "transition_available",
            "novel_transition",
            "novel_transition_known_values",
        )
    }
    positions = np.unique(profile["values"] // 256)
    for lo in range(0, events.count, BLOCK):
        hi = min(events.count, lo + BLOCK)
        keys, valid = byte_keys(events, lo, hi)
        known = np.isin(keys, profile["values"])
        supported = np.isin(keys // 256, positions)
        pairs, pair_valid = byte_keys(events, lo, hi, transition=True)
        new_pair = ~np.isin(pairs, profile["transitions"])
        prior_known = np.isin(pairs // 256, profile["values"])
        values = dict(
            normal_id_seen=np.isin(events.arrays["ids"][lo:hi], profile["ids"]),
            payload_available=valid.any(-1),
            unseen_position=(valid & ~supported).any(-1),
            novel_value=(valid & supported & ~known).any(-1),
            transition_available=pair_valid.any(-1),
            novel_transition=(pair_valid & new_pair).any(-1),
            novel_transition_known_values=(
                pair_valid & new_pair & known & prior_known
            ).any(-1),
        )
        for name, value in values.items():
            result[name][lo:hi] = value
    return result


def timing_features(events, ids, statistics):
    previous = events.previous_rows
    seen = previous >= 0
    gaps = np.where(
        seen, events.arrays["times"] - events.arrays["times"][previous.clip(0)], 0
    )
    position = np.searchsorted(ids, events.arrays["ids"]).clip(0, max(0, len(ids) - 1))
    supported = (
        seen & (ids[position] == events.arrays["ids"])
        if len(ids)
        else np.zeros_like(seen)
    )
    index = np.where(supported, position + 2, 1)
    scale, center, spread, clip = statistics[index].T
    normalized = np.where(
        seen, (np.log1p(np.minimum(gaps, clip) / scale) - center) / spread, 0
    )
    return dict(
        first_seen=~seen,
        timing_supported=supported,
        timing_fallback=seen & ~supported,
        timing_clipped=seen & (gaps > clip),
        gap_seconds=gaps,
        normalized_gap=normalized,
    )


def read_scores(directory, events, keep):
    """Check every physical validation row and paired arm before reporting."""
    validation = {k: r for k, r in events.items() if r.recording.role == "val_query"}
    result = {
        key: {name: np.empty(validation[key].count) for name in (*ARMS, "payload")}
        | dict(supported=np.empty(validation[key].count, dtype=bool))
        for key in keep
    }
    coverage = {}
    iterators = [iter_score_chunks(directory / name) for name in ARMS]
    for paired in zip(*iterators, strict=True):
        chunk, rows = paired[0]
        key, lo, count = chunk["recording"], chunk["row_start"], chunk["count"]
        if key not in validation or lo != coverage.get(key, 0):
            raise ValueError("duplicate/missing/unexpected validation rows")
        record, hi = validation[key], lo + count
        if hi > record.count or chunk["source_digest"] != record.digest:
            raise ValueError("score source/count differs from reviewed recording")
        expected = np.column_stack(
            [record.arrays[k][lo:hi] for k in ("times", "ids", "labels")]
        )
        support = (
            record.arrays["lengths"][lo:hi] > 0
        ) & record.recording.payload_length_verified
        if not np.array_equal(rows[:, :3], expected) or not np.array_equal(
            rows[:, 5], support
        ):
            raise ValueError("score timestamps/IDs/labels/payload support differ")
        for name, (meta, values) in zip(ARMS, paired, strict=True):
            if meta != chunk or not np.array_equal(
                values[:, [0, 1, 2, 4, 5]], rows[:, [0, 1, 2, 4, 5]]
            ):
                raise ValueError("paired scores are not aligned or payload changed")
            if (
                not np.isfinite(values).all()
                or ((values[:, 3] < 0) | (values[:, 3] > 1)).any()
            ):
                raise ValueError("nonfinite/invalid saved scores")
            if key in result:
                result[key][name][lo:hi] = values[:, 3]
        if key in result:
            result[key]["payload"][lo:hi], result[key]["supported"][lo:hi] = (
                rows[:, 4],
                support,
            )
        coverage[key] = hi
    if coverage != {key: r.count for key, r in validation.items()}:
        raise ValueError("incomplete validation coverage")
    return result, coverage


def stratum(mask, features, timing, scores):
    count = int(mask.sum())
    payload = mask & scores["supported"]
    return dict(
        rows=count,
        counts={
            key: int((mask & value).sum())
            for key, value in (features | timing).items()
            if value.dtype == bool
        },
        gap_seconds=quantiles(timing["gap_seconds"][mask & ~timing["first_seen"]]),
        normalized_gap=quantiles(
            timing["normalized_gap"][mask & ~timing["first_seen"]]
        ),
        classifier=quantiles(scores["original"][mask]),
        payload_nll=quantiles(scores["payload"][payload]),
        payload_supported=int(payload.sum()),
    )


def diagnose(dm, score_directory, focus, timing_ids, timing_statistics):
    events = dm.event_recordings
    for key, matches in focus.items():
        if key not in events or events[key].recording.role != "val_query":
            raise ValueError("focus must identify source validation queries")
        if any(
            k not in events or events[k].recording.role != "train_query"
            for k in matches
        ):
            raise ValueError("matches must identify source training queries")
    normals = [
        key
        for key, r in events.items()
        if r.recording.role == "val_query" and r.recording.verified_normal
    ]
    if not normals:
        raise ValueError("verified-normal validation exposure required")
    scores, coverage = read_scores(score_directory, events, set(focus) | set(normals))
    profiles = fit_profiles(events.values())
    training = {}
    audit = {}
    for key, record in events.items():
        arrays, rec = record.arrays, record.recording
        labels, times = arrays["labels"], arrays["times"]
        if (
            not np.isin(labels, [-1, 0, 1]).all()
            or not np.isfinite(times).all()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("invalid reviewed labels/clock")
        entry = dict(
            source_digest=record.digest,
            source_sha256=rec.sha256,
            role=rec.role,
            vehicle=rec.vehicle,
            rows=record.count,
            positives=int((labels == 1).sum()),
            negatives=int((labels == 0).sum()),
            unknown=int((labels < 0).sum()),
            labels_verified=rec.labels_verified,
            payload_length_verified=rec.payload_length_verified,
            start_time=float(times[0]),
            end_time=float(times[-1]),
            clock_ties=int((np.diff(times) == 0).sum()),
        )
        audit[key] = entry
        if rec.role == "train_query":
            training[key] = entry | dict(
                positive_ids={
                    f"{int(i):X}": int(((labels == 1) & (arrays["ids"] == i)).sum())
                    for i in np.unique(arrays["ids"][labels == 1])
                }
            )
    points = {}
    for name in (*ARMS, "payload"):
        normal_scores = np.concatenate(
            [scores[k][name][events[k].arrays["labels"] == 0] for k in normals]
        )
        support = np.concatenate(
            [
                scores[k]["supported"][events[k].arrays["labels"] == 0]
                if name == "payload"
                else np.ones((events[k].arrays["labels"] == 0).sum(), dtype=bool)
                for k in normals
            ]
        )
        points[name] = [budget_point(normal_scores, support, rate) for rate in BUDGETS]
        for point in points[name]:
            point["by_recording"] = {
                key: decisions(
                    scores[key][name],
                    scores[key]["supported"]
                    if name == "payload"
                    else np.ones(events[key].count, dtype=bool),
                    events[key].arrays["labels"],
                    point,
                )
                for key in normals
            }
    groups = {}
    for key, matches in focus.items():
        record, current = events[key], scores[key]
        labels = record.arrays["labels"]
        timing = timing_features(record, timing_ids, timing_statistics)
        primary = payload_features(
            record,
            profiles.get(
                record.recording.vehicle,
                {
                    name: np.empty(0, dtype=np.int64)
                    for name in ("ids", "values", "transitions")
                },
            ),
        )
        pooled = payload_features(record, profiles["pooled"])
        masks = dict(
            positive=labels == 1,
            negative=labels == 0,
            unknown=labels < 0,
            missed_positive=(labels == 1) & (current["original"] < 0.5),
        )
        metrics, budgeted = {}, {}
        for name in (*ARMS, "payload"):
            support = (
                current["supported"]
                if name == "payload"
                else np.ones(record.count, dtype=bool)
            )
            metrics[name] = frame_metrics(
                current[name][support],
                labels[support],
                threshold=0.5 if name != "payload" else None,
            )
            budgeted[name] = [
                decisions(current[name], support, labels, point)
                for point in points[name]
            ]
        common = current["supported"]
        metrics["original_on_payload_support"] = frame_metrics(
            current["original"][common], labels[common]
        )
        disagreement = {}
        for j, rate in enumerate(BUDGETS):
            classifier = current["original"] >= points["original"][j]["threshold"]
            payload = current["supported"] & (
                current["payload"] >= points["payload"][j]["threshold"]
            )
            disagreement[str(rate)] = {
                name: dict(
                    rows=int(mask.sum()),
                    both=int((mask & classifier & payload).sum()),
                    classifier_only=int((mask & classifier & ~payload).sum()),
                    payload_only=int((mask & ~classifier & payload).sum()),
                    neither=int((mask & ~classifier & ~payload).sum()),
                )
                for name, mask in masks.items()
                if name != "missed_positive"
            }
        by_id = {
            f"{int(i):X}": dict(
                positives=int(((labels == 1) & (record.arrays["ids"] == i)).sum()),
                negatives=int(((labels == 0) & (record.arrays["ids"] == i)).sum()),
            )
            for i in np.unique(record.arrays["ids"][labels == 1])
        }
        examples = []
        for row in np.flatnonzero(masks["missed_positive"])[:5]:
            length = record.arrays["lengths"][row]
            examples.append(
                dict(
                    row=int(row),
                    derived_csv_line=int(row + 2),
                    timestamp=float(record.arrays["times"][row]),
                    can_id=f"{int(record.arrays['ids'][row]):X}",
                    payload=record.arrays["payload"][row, :length].tobytes().hex(),
                    classifier=float(current["original"][row]),
                    payload_nll=float(current["payload"][row]) if common[row] else None,
                    **{name: bool(values[row]) for name, values in primary.items()},
                    timing_supported=bool(timing["timing_supported"][row]),
                    normalized_gap=float(timing["normalized_gap"][row]),
                )
            )
        groups[key] = dict(
            audit=audit[key],
            matched_training=matches,
            matched_training_positives=sum(training[k]["positives"] for k in matches),
            positive_ids=by_id,
            metrics=metrics,
            at_normal_budgets=budgeted,
            disagreement=disagreement,
            examples=examples,
            strata={
                name: stratum(mask, primary, timing, current)
                for name, mask in masks.items()
            },
            pooled_profile_counts={
                name: {k: int((mask & v).sum()) for k, v in pooled.items()}
                for name, mask in masks.items()
            },
        )
    return dict(
        status="finished",
        created_at=datetime.now(UTC).isoformat(),
        interpretation="Exploratory source selection data; empirical normal-frame budgets, not calibrated alerts/hour or independent test results.",
        budget_protocol="Thresholds use pooled verified-normal val_query negatives only, >= comparison, conservative ties. All known normal rows remain in denominators; unsupported payload rows never alarm.",
        novelty_protocol="Verified-normal train_query only; marginal bytes and preceding same-ID byte transitions. Vehicle-scoped primary, pooled-ID sensitivity. Novelty does not establish attacker origin or whole-model distinguishability.",
        coverage=coverage,
        source_audit=audit,
        training_exposure=training,
        normal_budget_recordings=normals,
        normal_budget_points=points,
        focus=groups,
    )


def markdown(report):
    lines = [
        "# Weak-capture failure report",
        "",
        report["interpretation"],
        "",
        report["budget_protocol"],
        "",
        report["novelty_protocol"],
        "",
        f"All {sum(report['coverage'].values()):,} validation rows verified against reviewed source caches and paired score artifacts.",
        "",
        "| Capture | Matched training positives | Validation positives | Classifier AP | Payload AP | Classifier AP on payload support |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, g in report["focus"].items():
        m = g["metrics"]
        lines.append(
            f"| {key} | {g['matched_training_positives']} | {g['audit']['positives']} | {m['original'].get('frame_ap')} | {m['payload'].get('frame_ap')} | {m['original_on_payload_support'].get('frame_ap')} |"
        )
    lines += [
        "",
        "## Empirical normal-budget thresholds",
        "",
        "| Detector | Requested normal FPR | Threshold | Achieved FP / normal frames |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, points in report["normal_budget_points"].items():
        for point in points:
            lines.append(
                f"| {name} | {point['requested_fpr']} | {point['threshold']:.8g} | {point['achieved_fp']:,} / {point['normal_frames']:,} |"
            )
    for key, g in report["focus"].items():
        lines += [
            "",
            f"## {key}",
            "",
            "### Signal support (vehicle-scoped normal training)",
            "",
            "| Stratum | Rows | Timing supported | First seen | Payload supported | Novel byte | Novel transition with known byte values |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, s in g["strata"].items():
            c = s["counts"]
            lines.append(
                f"| {name} | {s['rows']} | {c['timing_supported']} | {c['first_seen']} | {s['payload_supported']} | {c['novel_value']} | {c['novel_transition_known_values']} |"
            )
        lines += [
            "",
            "### Recall at empirical normal-frame budgets",
            "",
            "Each cell gives recall; parenthesized counts are false-positive frames within this attack capture. Payload recall includes unsupported positives in its denominator.",
            "",
            "| Requested normal FPR | Original | Ordinary | Balanced | Payload |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for j, rate in enumerate(BUDGETS):
            cells = [
                f"{g['at_normal_budgets'][name][j]['recall']:.4f} ({g['at_normal_budgets'][name][j]['fp']:,})"
                for name in (*ARMS, "payload")
            ]
            lines.append("| " + " | ".join([str(rate), *cells]) + " |")
        lines += [
            "",
            "Detailed score/gap quantiles, support differences, per-ID counts, matched training captures, head disagreement and five missed-frame examples are in `report.json`.",
        ]
    lines += [
        "",
        "These are checks of reviewed labels, source identities and clock integrity, not an independent relabeling audit. Current-byte overlap does not establish indistinguishability with history. Only two normal training captures supply the novelty profiles.",
        "",
    ]
    return "\n".join(lines)


def main():
    import torch

    from graphids.exp.config import ExperimentConfig
    from graphids.exp.run import build_component
    from graphids.exp.slurm import dependency_versions, file_hash
    from scripts.event_model_gate import source_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--focus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run full source analysis inside a CPU Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = json.loads((args.pilot / "pilot/pilot.json").read_text())
    if (
        pilot["status"] != "finished"
        or file_hash(Path(pilot["checkpoint"])) != pilot["checkpoint_sha256"]
    ):
        raise ValueError("pilot incomplete or base checkpoint changed")
    focus = json.loads(args.focus.read_text())
    cfg = ExperimentConfig.from_yaml(args.pilot / "pilot/configuration.yml")
    dm = build_component(cfg.config["data"])
    dm.setup()
    saved = torch.load(pilot["checkpoint"], map_location="cpu", weights_only=False)
    if (
        dm.manifest.digest != pilot["source_manifest_digest"]
        or dm.manifest.digest != saved["event_input_contract"]["source_manifest_digest"]
    ):
        raise ValueError("source manifest identity differs")
    arrays = saved["state_dict"]
    report = diagnose(
        dm,
        args.pilot / "pilot/scores",
        focus,
        arrays["timing_ids"].numpy(),
        arrays["timing_statistics"].numpy(),
    )
    git_head, diff, hashes = source_provenance(Path(__file__).resolve().parents[1])
    files = [
        args.focus,
        args.pilot / "pilot/pilot.json",
        args.pilot / "pilot/configuration.yml",
        Path(pilot["checkpoint"]),
    ]
    files += [
        args.pilot / "pilot/scores" / arm / name
        for arm in ARMS
        for name in ("index.json", "scores.bin")
    ]
    report["provenance"] = dict(
        git_head=git_head,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        checkpoint_sha256=pilot["checkpoint_sha256"],
        source_manifest_digest=dm.manifest.digest,
        inputs={str(p): file_hash(p) for p in files},
        source_hashes=hashes,
        dependency_versions=dependency_versions(),
    )
    for name, value in (
        ("report.json", report),
        ("source-manifest.json", asdict(dm.manifest)),
    ):
        (args.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "working-tree.diff").write_bytes(diff)
    (args.output / "report.md").write_text(markdown(report))
    print(f"Complete: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
