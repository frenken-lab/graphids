"""Descriptive post-hoc audit of recurring CAN periods in all normal captures."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from graphids.can_dynamics import quantile
from graphids.paths import lake_root

SETS = ("set_01", "set_02", "set_03", "set_04")
BANDS = (0.005, 0.01, 0.0125, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0)


def summarize_capture(path: Path, *, cap: int = 2000) -> dict:
    previous: dict[str, float] = {}
    counts: Counter[str] = Counter()
    gaps: dict[str, list[float]] = defaultdict(list)
    seen: Counter[str] = Counter()
    decreasing = 0
    previous_raw = -math.inf
    rng = random.Random(91)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp", "arbitration_id"} <= set(reader.fieldnames or []):
            raise ValueError(f"missing timing columns: {path}")
        for row in reader:
            t = float(row["timestamp"])
            if t < previous_raw:
                decreasing += 1
            previous_raw = t
            key = row["arbitration_id"].strip()
            counts[key] += 1
            if key in previous and t > previous[key]:
                seen[key] += 1
                values = gaps[key]
                if len(values) < cap:
                    values.append(t - previous[key])
                else:
                    index = rng.randrange(seen[key])
                    if index < cap:
                        values[index] = t - previous[key]
            previous[key] = t
    if decreasing:
        raise ValueError(f"{path} has {decreasing} decreasing timestamps; sort before gap audit")
    profiles = {}
    for key, values in gaps.items():
        if len(values) < 30:
            continue
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        p90 = quantile(values, .9)
        stable = median > 0 and mad / median <= .15 and p90 / median <= 1.5
        nearest = min(BANDS, key=lambda period: abs(math.log(max(median, 1e-9) / period)))
        band = f"{nearest:g}s" if abs(median / nearest - 1) <= .2 else "other"
        profiles[key] = {"count": counts[key], "sampled_gaps": len(values),
                         "median_seconds": median, "mad_seconds": mad,
                         "p90_seconds": p90, "stable": stable, "band": band}
    stable_mass = sum(item["count"] for item in profiles.values() if item["stable"])
    band_mass = Counter()
    band_ids = Counter()
    for item in profiles.values():
        if item["stable"]:
            band_mass[item["band"]] += item["count"]
            band_ids[item["band"]] += 1
    return {"source": str(path), "rows": sum(counts.values()), "ids": len(counts),
            "decreasing_timestamps": decreasing, "stable_ids": sum(x["stable"] for x in profiles.values()),
            "stable_frame_fraction": stable_mass / max(1, sum(counts.values())),
            "band_frame_counts": dict(sorted(band_mass.items())),
            "band_id_counts": dict(sorted(band_ids.items())), "profiles": profiles}


def _id_times(path: Path, selected: set[str]) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["arbitration_id"].strip()
            if key in selected:
                times[key].append(float(row["timestamp"]))
    return times


def _phase_stats(triggers: list[float], targets: list[float], period: float) -> tuple[float, float, int]:
    if not triggers or not targets or period <= 0:
        return 0.0, 0.0, 0
    stride = max(1, math.ceil(len(targets) / 500))
    sin_sum = cos_sum = 0.0
    count = 0
    for t in targets[::stride]:
        index = bisect.bisect_left(triggers, t) - 1
        if index < 0:
            continue
        phase = ((t - triggers[index]) / period) % 1
        angle = 2 * math.pi * phase
        sin_sum += math.sin(angle)
        cos_sum += math.cos(angle)
        count += 1
    if not count:
        return 0.0, 0.0, 0
    concentration = math.hypot(sin_sum, cos_sum) / count
    mean_phase = (math.atan2(sin_sum, cos_sum) / (2 * math.pi)) % 1
    return concentration, mean_phase, count


def _phase_distance(a: float, b: float) -> float:
    distance = abs(a - b)
    return min(distance, 1 - distance)


def phase_locks(paths: dict[str, list[Path]], captures: dict[str, list[dict]]) -> dict:
    references = captures["reference"]
    common = ({key for key, value in references[0]["profiles"].items() if value["stable"]} &
              {key for key, value in references[1]["profiles"].items() if value["stable"]})
    times = [_id_times(path, common) for path in [*paths["reference"], *paths["normal_test"]]]
    candidates = []
    for trigger in sorted(common):
        period = references[0]["profiles"][trigger]["median_seconds"]
        other_period = references[1]["profiles"][trigger]["median_seconds"]
        if max(period, other_period) / min(period, other_period) > 1.05:
            continue
        for target in sorted(common - {trigger}):
            target_period = references[0]["profiles"][target]["median_seconds"]
            if max(period, target_period) / min(period, target_period) > 1.05:
                continue
            first = _phase_stats(times[0].get(trigger, []), times[0].get(target, []), period)
            second = _phase_stats(times[1].get(trigger, []), times[1].get(target, []), other_period)
            if min(first[2], second[2]) < 30 or min(first[0], second[0]) < .9:
                continue
            if _phase_distance(first[1], second[1]) > .1:
                continue
            test_period = captures["normal_test"][0]["profiles"].get(trigger, {}).get("median_seconds", period)
            test = _phase_stats(times[2].get(trigger, []), times[2].get(target, []), test_period)
            test_stable = test[2] >= 30 and test[0] >= .8 and _phase_distance(first[1], test[1]) <= .15
            nearest = min(BANDS, key=lambda value: abs(math.log(period / value)))
            band = f"{nearest:g}s" if abs(period / nearest - 1) <= .2 else "other"
            candidates.append({"trigger": trigger, "target": target, "period_seconds": period,
                               "band": band, "reference_concentration": [first[0], second[0]],
                               "reference_phase": [first[1], second[1]],
                               "test_concentration": test[0], "test_phase": test[1],
                               "test_stable": test_stable})
    candidates.sort(key=lambda row: (-min(row["reference_concentration"]), row["trigger"], row["target"]))
    return {"reference_locked_pairs": len(candidates),
            "normal_test_stable_pairs": sum(row["test_stable"] for row in candidates),
            "locked_period_bands": sorted({row["band"] for row in candidates}),
            "examples": candidates[:30]}


def audit_set(raw: Path, assignment: dict) -> dict:
    paths = {role: [raw / item["path"] for item in assignment[role]]
             for role in ("reference", "calibration", "normal_test")}
    captures = {role: [summarize_capture(path) for path in files] for role, files in paths.items()}
    all_captures = [item for group in captures.values() for item in group]
    stable_sets = [{key for key, profile in capture["profiles"].items() if profile["stable"]}
                   for capture in all_captures]
    recurring = set.intersection(*stable_sets)
    recurring = {key for key in recurring if max(capture["profiles"][key]["median_seconds"] for capture in all_captures)
                 / min(capture["profiles"][key]["median_seconds"] for capture in all_captures) <= 1.2}
    test = captures["normal_test"][0]
    recurring_mass = sum(test["profiles"][key]["count"] for key in recurring)
    common_bands = set.intersection(*[set(capture["band_id_counts"]) for capture in all_captures])
    locks = phase_locks(paths, captures)
    return {"set": raw.name, "captures": captures,
            "stable_ids_all_four_files": sorted(recurring),
            "stable_id_count_all_four_files": len(recurring),
            "stable_test_frame_fraction_all_four_files": recurring_mass / max(1, test["rows"]),
            "period_bands_all_four_files": sorted(common_bands),
            "same_period_phase_locks": locks,
            "reference_ids": sorted(set(captures["reference"][0]["profiles"]) |
                                    set(captures["reference"][1]["profiles"])),
            "normal_test_ids_absent_from_reference": sorted(set(test["profiles"]) -
                                                          set(captures["reference"][0]["profiles"]) -
                                                          set(captures["reference"][1]["profiles"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_root = args.raw_root or Path(lake_root()) / "raw"
    results = {}
    for name in SETS:
        assignment = json.loads((args.pilot_root / name / "assignment.json").read_text())
        results[name] = audit_set(raw_root / name, assignment)
        print(name, results[name]["stable_id_count_all_four_files"], flush=True)
    cross_set_bands = set.intersection(*[set(result["period_bands_all_four_files"])
                                         for result in results.values()])
    report = {"version": 1, "purpose": "descriptive_post_hoc_normal_periodicity",
              "band_centers_seconds": BANDS, "stable_rule": "MAD/median <= .15 and p90/median <= 1.5; at least 30 sampled gaps",
              "common_period_bands_all_sets": sorted(cross_set_bands), "sets": results,
              "note": "Calibration and test recordings are included only for descriptive audit, never rule fitting."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
