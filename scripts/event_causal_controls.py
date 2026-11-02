"""Source-only ID-novelty and same-ID-gap controls against saved neural scores.

Run inside a CPU Slurm allocation; no neural execution or target data access.
python -m scripts.event_causal_controls --pilot PILOT_ROOT --output NEW_DIRECTORY
Add --payload-transition for smoothed marginal and same-ID byte-pair controls.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from graphids.data.context import fit_timing_statistics
from graphids.metrics import frame_metrics
from scripts.event_failure_report import (
    BUDGETS,
    budget_point,
    decisions,
    read_scores,
    timing_features,
)


def fit_controls(recordings, *, min_support=32):
    """Freeze vehicle-local ID sets and robust gap tables on trusted training only."""
    if min_support < 1:
        raise ValueError("min_support must be positive")
    vehicles = {}
    for events in recordings:
        rec = events.recording
        if rec.role == "train_query" and rec.verified_normal:
            if not np.all(events.arrays["labels"] == 0):
                raise ValueError("verified-normal training contains nonnormal labels")
            vehicles.setdefault(rec.vehicle, []).append(events)
    if not vehicles:
        raise ValueError("verified-normal training required")
    return {
        vehicle: dict(
            recordings=[r.recording.key for r in records],
            rows=sum(r.count for r in records),
            ids=np.unique(np.concatenate([r.arrays["ids"] for r in records])).tolist(),
            timing_ids=keys.tolist(),
            timing_statistics=statistics.tolist(),
        )
        for vehicle, records in vehicles.items()
        for keys, statistics in [
            fit_timing_statistics(records, min_support=min_support)
        ]
    }


def control_scores(events, profiles):
    """Read preceding same-ID times; never update normal profiles from query rows.

    Unprofiled vehicles abstain. Gap control also abstains on first occurrences
    and IDs with fewer than min_support sampled training pairs. No pooled-ID
    fallback is used. Unsupported scores are zero and cannot raise an alarm.
    """
    profile = profiles.get(events.recording.vehicle)
    first = events.previous_rows < 0
    known = (
        np.isin(events.arrays["ids"], profile["ids"])
        if profile
        else np.zeros(events.count, bool)
    )
    novelty_support = np.full(events.count, profile is not None, dtype=bool)
    gap_support = np.zeros(events.count, bool)
    gap = np.zeros(events.count)
    if profile:
        timing = timing_features(
            events,
            np.asarray(profile["timing_ids"], dtype=np.int64),
            np.asarray(profile["timing_statistics"]),
        )
        gap_support = timing["timing_supported"]
        gap[gap_support] = np.abs(timing["normalized_gap"][gap_support])
    return dict(
        id_novelty=(~known & novelty_support).astype(float),
        id_novelty_supported=novelty_support,
        gap=gap,
        gap_supported=gap_support,
        known_id=known,
        first_seen=first,
    )


def summarize(values, supports, labels, points):
    """Ranking on each detector's support plus all-row diagnostic decisions."""
    result = {}
    for name in values:
        score, support = values[name], supports[name]
        result[name] = dict(
            supported_metrics=frame_metrics(score[support], labels[support]),
            at_normal_budgets=[
                decisions(score, support, labels, p) for p in points[name]
            ],
        )
    return result


def evaluate(
    dm, score_directory, *, min_support=32, output=None, payload_transition=False
):
    from scripts.event_payload_control import (
        MARGINAL_MASS,
        TRANSITION_MASS,
        fit_payload_profiles,
        payload_scores,
        save_profiles,
    )

    events = dm.event_recordings
    if any(
        r.recording.role not in ("train_query", "val_query") for r in events.values()
    ):
        raise ValueError("controls accept source query recordings only")
    for record in events.values():
        labels, times = record.arrays["labels"], record.arrays["times"]
        if (
            not np.isin(labels, [-1, 0, 1]).all()
            or not np.isfinite(times).all()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("invalid reviewed labels/clock")
        if record.recording.verified_normal and not (labels == 0).all():
            raise ValueError("verified-normal recording contains nonnormal labels")
    validation = {k: r for k, r in events.items() if r.recording.role == "val_query"}
    normals = [k for k, r in validation.items() if r.recording.verified_normal]
    if not normals:
        raise ValueError("verified-normal validation exposure required")
    saved, coverage = read_scores(score_directory, events, set(validation))
    profiles = fit_controls(events.values(), min_support=min_support)
    byte_profiles = (
        fit_payload_profiles(events.values()) if payload_transition else None
    )
    values, supports, features = {}, {}, {}
    artifacts = {}
    for key, record in validation.items():
        control = control_scores(record, profiles)
        features[key] = control
        values[key] = {name: saved[key][name] for name in ("original", "payload")} | {
            name: control[name] for name in ("id_novelty", "gap")
        }
        supports[key] = dict(
            original=np.ones(record.count, bool),
            payload=saved[key]["supported"],
            id_novelty=control["id_novelty_supported"],
            gap=control["gap_supported"],
        )
        if byte_profiles is not None:
            control.update(payload_scores(record, byte_profiles))
            for name in ("byte_marginal", "byte_transition"):
                values[key][name] = control[name]
                supports[key][name] = control["byte_supported"]
        if output is not None:
            path = Path(output) / f"controls-{len(artifacts):02d}.npz"
            np.savez_compressed(path, **control)
            artifacts[key] = dict(
                file=path.name, source_digest=record.digest, rows=record.count
            )
    points = {}
    detector_names = tuple(next(iter(values.values())))
    for name in detector_names:
        score = np.concatenate([values[k][name] for k in normals])
        support = np.concatenate([supports[k][name] for k in normals])
        # An entirely unsupported normal cohort gives no threshold evidence.
        if support.any():
            points[name] = [budget_point(score, support, rate) for rate in BUDGETS]
        else:
            raise ValueError(f"no supported normal validation frames for {name}")
    groups = {}
    for key, record in validation.items():
        labels, feature = record.arrays["labels"], features[key]
        strata = {}
        masks = dict(
            known_id=feature["known_id"],
            novel_id=feature["id_novelty"] > 0,
            first_seen=feature["first_seen"],
            gap_supported=feature["gap_supported"],
        )
        if byte_profiles is not None:
            masks.update(
                {
                    n: feature[n]
                    for n in (
                        "byte_supported",
                        "byte_backoff",
                        "byte_fully_conditioned",
                    )
                }
            )
        for name, mask in masks.items():
            strata[name] = dict(
                rows=int(mask.sum()),
                positives=int(((labels == 1) & mask).sum()),
                negatives=int(((labels == 0) & mask).sum()),
                unknown=int(((labels < 0) & mask).sum()),
                detectors=summarize(
                    {n: v[mask] for n, v in values[key].items()},
                    {n: v[mask] for n, v in supports[key].items()},
                    labels[mask],
                    points,
                ),
                fixed_id_novelty=decisions(
                    values[key]["id_novelty"][mask],
                    supports[key]["id_novelty"][mask],
                    labels[mask],
                    dict(threshold=0.5),
                ),
            )
        groups[key] = dict(
            vehicle=record.recording.vehicle,
            source_digest=record.digest,
            rows=record.count,
            positives=int((labels == 1).sum()),
            negatives=int((labels == 0).sum()),
            unknown=int((labels < 0).sum()),
            vehicle_profile_available=record.recording.vehicle in profiles,
            detectors=summarize(values[key], supports[key], labels, points),
            strata=strata,
            fixed_id_novelty=decisions(
                values[key]["id_novelty"],
                supports[key]["id_novelty"],
                labels,
                dict(threshold=0.5),
            ),
        )
        if byte_profiles is not None:
            groups[key]["payload_control_bytes"] = dict(
                scored=int(feature["byte_scored_bytes"].sum()),
                with_training_transition_context=int(
                    feature["byte_context_bytes"].sum()
                ),
            )
    labels = np.concatenate([r.arrays["labels"] for r in validation.values()])
    aggregate = summarize(
        {n: np.concatenate([v[n] for v in values.values()]) for n in detector_names},
        {n: np.concatenate([s[n] for s in supports.values()]) for n in detector_names},
        labels,
        points,
    )
    for name in detector_names:
        aps = [
            g["detectors"][name]["supported_metrics"]["frame_ap"]
            for g in groups.values()
            if "frame_ap" in g["detectors"][name]["supported_metrics"]
        ]
        aggregate[name].update(
            recording_macro_ap=float(np.mean(aps)) if aps else None,
            macro_recordings=len(aps),
        )
    index = json.loads((Path(score_directory) / "original/index.json").read_text())
    parity = 0
    for group, expected in [(aggregate, index["metrics"])] + [
        (groups[key]["detectors"], index["by_recording"][key]) for key in validation
    ]:
        for name, prefix in (("original", "classification"), ("payload", "payload")):
            for metric, value in group[name]["supported_metrics"].items():
                if expected.get(f"{prefix}_{metric}") != value:
                    raise ValueError("saved original/payload metrics do not reproduce")
                parity += 1
    report = dict(
        status="finished",
        created_at=datetime.now(UTC).isoformat(),
        interpretation="Exploratory source selection-validation diagnostics; vehicle-scoped controls do not establish zero-shot transfer or independent improvement.",
        control_protocol="ID novelty: fixed score 1 for IDs absent from vehicle's verified-normal train_query union, else 0; fixed alarm >=0.5. Gap: absolute robust log-gap deviation using vehicle-local training tables and the existing training-quantile clip. Minimum support counts sampled predecessor pairs, with <=65,536 deterministic sampled rows per training recording. Query history is preceding-only and resets per recording. No pooled-ID fallback; unprofiled vehicles, first occurrences and unsupported timing IDs abstain for gap.",
        metric_protocol="AP/AUC use each detector's supported known rows; denominators are explicit. Strata provide matched known-ID and gap-supported comparisons. Macro AP averages only recordings with both classes on that detector's support. Unsupported frames remain in budgeted-recall/FPR denominators and never alarm. Unknown labels are excluded from metrics, retained in coverage/history. Payload support is independent of ID/timing support.",
        budget_protocol="Thresholds use only verified-normal val_query frames, pooled across source vehicles, with >= comparison and conservative ties. These are empirical normal-frame FPR budgets on reused selection data, not calibration or alerts/hour. Fixed ID novelty is also reported without a validation-fitted threshold. Discrete scores may leave the budget unused.",
        coverage=coverage,
        min_support=min_support,
        profiles=profiles,
        normal_budget_recordings=normals,
        normal_budget_points=points,
        aggregate=aggregate,
        saved_metric_parity=dict(passed=True, metrics_checked=parity),
        by_recording=groups,
        control_artifacts=artifacts,
    )
    if byte_profiles is not None:
        report["payload_control"] = dict(
            marginal_mass=MARGINAL_MASS,
            transition_mass=TRANSITION_MASS,
            profiles=save_profiles(
                byte_profiles, Path(output) if output is not None else None
            ),
            protocol="All verified-normal train_query bytes/pairs with verified lengths; vehicle/ID/position-scoped sparse counts, no query updates. Marginal P(b)=(count(b)+1/256)/(N+1). Transition P(b|a)=(count(a,b)+32*P(b))/(count(a)+32). Mean NLL across valid current bytes. Missing preceding positions/first occurrences/unseen contexts back off to the marginal; sparse contexts blend continuously. Both controls abstain on missing/unverified payloads, unprofiled vehicles or any current byte position without training support. No cross-ID/vehicle fallback. Query labels never affect history. byte_supported stratum supplies identical-row comparisons to neural scores; unsupported positives remain budgeted misses. Constants fixed before this run; no validation tuning.",
        )
    return report


def markdown(report):
    lines = ["# Causal source controls", ""]
    for field in (
        "interpretation",
        "control_protocol",
        "metric_protocol",
        "budget_protocol",
    ):
        lines += [report[field], ""]
    if "payload_control" in report:
        lines += [report["payload_control"]["protocol"], ""]
    lines += [
        f"Verified {sum(report['coverage'].values()):,} physical validation rows.",
        "",
        "| Detector | Supported known frames | AP | Recording macro AP | Macro captures |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def ap(value):
        return "undefined" if value is None else f"{value:.6f}"

    for name, value in report["aggregate"].items():
        metric = value["supported_metrics"]
        lines.append(
            f"| {name} | {metric['frames']:,} | {ap(metric.get('frame_ap'))} | {ap(value['recording_macro_ap'])} | {value['macro_recordings']} |"
        )
    lines += [
        "",
        "## Fixed ID-novelty alarm",
        "",
        "| Capture | TP / positives | FP / negatives | Novel first-seen positives | Unknown rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, value in report["by_recording"].items():
        fixed = value["fixed_id_novelty"]
        novel_first = value["strata"]["first_seen"]["fixed_id_novelty"]["tp"]
        lines.append(
            f"| {key} | {fixed['tp']:,} / {fixed['positives']:,} | {fixed['fp']:,} / {fixed['negatives']:,} | {novel_first:,} | {value['unknown']} |"
        )
    for j, rate in enumerate(BUDGETS):
        lines += [
            "",
            f"## Empirical normal-frame FPR budget {rate:g}",
            "",
            "| Capture | Detector | AP on support | TP / positives | FP / negatives | Unsupported positives |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for key, value in report["by_recording"].items():
            for name, detector in value["detectors"].items():
                point = detector["at_normal_budgets"][j]
                lines.append(
                    f"| {key} | {name} | {ap(detector['supported_metrics'].get('frame_ap'))} | {point['tp']:,} / {point['positives']:,} | {point['fp']:,} / {point['negatives']:,} | {point['unsupported_positives']:,} |"
                )
    lines += [
        "",
        "Known-ID and common support comparisons, support counts, frozen profiles and full control predictions are recorded in report.json and controls-*.npz.",
        "",
    ]
    return "\n".join(lines)


def main():
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.run import build_component
    from graphids.exp.slurm import dependency_versions, file_hash
    from scripts.event_model_gate import source_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=32)
    parser.add_argument(
        "--payload-transition",
        action="store_true",
        help="Include frozen vehicle-scoped marginal and same-ID byte-transition controls",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run full source analysis inside a CPU Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    pilot_path = args.pilot / "pilot/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    checkpoint = Path(pilot["checkpoint"])
    if (
        pilot["status"] != "finished"
        or file_hash(checkpoint) != pilot["checkpoint_sha256"]
    ):
        raise ValueError("pilot incomplete or base checkpoint changed")
    configuration = args.pilot / "pilot/configuration.yml"
    cfg = ExperimentConfig.from_yaml(configuration)
    dm = build_component(cfg.config["data"])
    dm.setup()
    if dm.manifest.digest != pilot["source_manifest_digest"]:
        raise ValueError("source manifest identity differs")
    report = evaluate(
        dm,
        args.pilot / "pilot/scores",
        min_support=args.min_support,
        output=args.output,
        payload_transition=args.payload_transition,
    )
    git_head, diff, hashes = source_provenance(Path(__file__).resolve().parents[1])
    files = [pilot_path, configuration, checkpoint] + [
        args.pilot / "pilot/scores" / arm / name
        for arm in ("original", "ordinary", "balanced")
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
        control_sha256={
            p.name: file_hash(p) for p in args.output.glob("controls-*.npz")
        },
        payload_profile_sha256={
            p.name: file_hash(p) for p in args.output.glob("payload-profile-*.npz")
        },
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
