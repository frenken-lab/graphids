"""Reference-only comparison of timing, cross-ID phase, and change coupling."""
from __future__ import annotations

import bisect
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field

from graphids.core.data.discovery.behavior.patterns import Capture, Event


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile of empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class DynamicsConfig:
    bin_width: float = 0.1
    mode_window: float = 1.0
    min_id_samples: int = 30
    min_pair_samples: int = 30
    max_candidate_ids: int = 24
    max_pairs: int = 16
    max_pair_lag: float = 0.1
    sample_per_stream: int = 600
    reservoir_cap: int = 5000
    timing_quantile: float = 0.995
    timing_multiplier: float = 1.5
    max_period_ratio: float = 2.5
    min_pair_completion: float = 0.7
    min_control_gain: float = 0.1
    phase_margin: float = 0.002
    synthetic_min_seconds: float = 3.0


@dataclass(frozen=True)
class TimingRule:
    arb_id: str
    all_limit: float
    low_limit: float | None
    high_limit: float | None
    reference_counts: list[int]
    reference_medians: list[float]


@dataclass(frozen=True)
class PairRule:
    trigger: str
    target: str
    kind: str
    lag_low: float
    lag_high: float
    support: list[int]
    completion: list[float]
    control_completion: list[float]
    reference_medians: list[float]


@dataclass
class DynamicsProfile:
    version: int
    config: dict
    source_hashes: dict[str, str]
    mode_count_threshold: float
    timing: list[TimingRule]
    phase: list[PairRule]
    change_coupling: list[PairRule]
    rejected: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _reservoir_add(samples: list[float], value: float, seen: int, cap: int, rng: random.Random) -> None:
    if len(samples) < cap:
        samples.append(value)
    else:
        index = rng.randrange(seen)
        if index < cap:
            samples[index] = value


def _reference_gaps(capture: Capture, threshold: float, config: DynamicsConfig) -> dict[tuple[str, str], list[float]]:
    recent: deque[float] = deque()
    previous: dict[str, float] = {}
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: Counter[tuple[str, str]] = Counter()
    rng = random.Random(117)
    for event in capture.events:
        if event.kind != "arrival":
            continue
        t = event.timestamp
        while recent and recent[0] < t - config.mode_window:
            recent.popleft()
        mode = "high" if len(recent) >= threshold else "low"
        if event.arb_id in previous:
            gap = t - previous[event.arb_id]
            if gap > 0:
                for label in ("all", mode):
                    key = (event.arb_id, label)
                    seen[key] += 1
                    _reservoir_add(samples[key], gap, seen[key], config.reservoir_cap, rng)
        previous[event.arb_id] = t
        recent.append(t)
    return samples


def _times(capture: Capture, kind: str, selected: set[str]) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    for event in capture.events:
        if event.kind == kind and event.arb_id in selected:
            times[event.arb_id].append(event.timestamp)
    return times


def _lag_sample(targets: list[float], triggers: list[float],
                config: DynamicsConfig) -> tuple[list[float], int]:
    if not targets or not triggers:
        return [], 0
    stride = max(1, math.ceil(len(targets) / config.sample_per_stream))
    sampled = targets[::stride]
    lags = []
    for target in sampled:
        index = bisect.bisect_left(triggers, target) - 1
        if index >= 0 and 0 < target - triggers[index] <= config.max_pair_lag:
            lags.append(target - triggers[index])
    return lags, len(sampled)


def _pairs(captures: list[Capture], kind: str, config: DynamicsConfig,
           rejected: list[dict]) -> list[PairRule]:
    per_capture = [Counter(e.arb_id for e in capture.events if e.kind == kind)
                   for capture in captures]
    shared = set.intersection(*(set(counts) for counts in per_capture))
    counts = Counter({key: sum(counts[key] for counts in per_capture) for key in shared})
    ids = sorted(shared, key=lambda key: (-counts[key], key))[:config.max_candidate_ids]
    streams = [_times(capture, kind, set(ids)) for capture in captures]
    shifted_streams = []
    for capture, stream in zip(captures, streams, strict=True):
        span = capture.end - capture.start
        shifted_streams.append({key: sorted(capture.start + ((t - capture.start + span / 2) % span) for t in values) if span else []
                                for key, values in stream.items()})
    selected = []
    for trigger in ids:
        for target in ids:
            if trigger == target:
                continue
            samples = []
            controls = []
            totals = []
            for stream, shifted in zip(streams, shifted_streams, strict=True):
                lags, total = _lag_sample(stream.get(target, []), stream.get(trigger, []), config)
                control, _ = _lag_sample(stream.get(target, []), shifted.get(trigger, []), config)
                samples.append(lags)
                controls.append(control)
                totals.append(total)
            support = [len(x) for x in samples]
            completion = [len(x) / max(1, n) for x, n in zip(samples, totals, strict=True)]
            control_rate = [len(x) / max(1, n) for x, n in zip(controls, totals, strict=True)]
            reason = None
            if not all(n >= config.min_pair_samples and rate >= config.min_pair_completion
                       for n, rate in zip(support, completion, strict=True)):
                reason = "insufficient_or_unstable_support"
            else:
                medians = [statistics.median(x) for x in samples]
                if abs(medians[0] - medians[1]) > max(0.01, 0.5 * statistics.median(medians)):
                    reason = "lag_mode_shift"
                elif not all(rate >= control + config.min_control_gain
                             for rate, control in zip(completion, control_rate, strict=True)):
                    true_width = [quantile(x, 0.9) - quantile(x, 0.1) for x in samples]
                    control_width = [quantile(x, 0.9) - quantile(x, 0.1) if x else 0 for x in controls]
                    if not all(c > 0 and w < 0.5 * c for w, c in zip(true_width, control_width, strict=True)):
                        reason = "shift_control"
            if reason:
                rejected.append({"kind": kind, "trigger": trigger, "target": target,
                                 "support": support, "completion": completion,
                                 "control_completion": control_rate, "reason": reason})
                continue
            pooled = samples[0] + samples[1]
            low = max(0.0, quantile(pooled, 0.01) - config.phase_margin)
            high = min(config.max_pair_lag, quantile(pooled, 0.99) + config.phase_margin)
            rule = PairRule(trigger, target, kind, low, high, support, completion,
                            control_rate, [statistics.median(x) for x in samples])
            rank = min(completion) - max(control_rate) + min(0.1, 0.01 / max(high - low, 1e-6))
            selected.append((rank, rule))
    selected.sort(key=lambda item: (-item[0], item[1].trigger, item[1].target))
    for _, rule in selected[config.max_pairs:]:
        rejected.append({"kind": kind, "trigger": rule.trigger,
                         "target": rule.target, "reason": "pair_cap"})
    return [rule for _, rule in selected[:config.max_pairs]]


def fit_dynamics(captures: list[Capture], config: DynamicsConfig) -> DynamicsProfile:
    if len(captures) != 2:
        raise ValueError("two independent reference captures are required")
    per_second = []
    for capture in captures:
        counts = Counter(int(e.timestamp - capture.start) for e in capture.events if e.kind == "arrival")
        per_second.extend(counts.get(i, 0) for i in range(max(1, int(capture.end - capture.start) + 1)))
    threshold = statistics.median(per_second)
    gap_sets = [_reference_gaps(capture, threshold, config) for capture in captures]
    ids = sorted({key[0] for sample in gap_sets for key in sample})
    timing = []
    rejected = []
    for arb_id in ids:
        all_gaps = [sample.get((arb_id, "all"), []) for sample in gap_sets]
        if any(len(x) < config.min_id_samples for x in all_gaps):
            rejected.append({"kind": "timing", "id": arb_id, "reason": "insufficient_support"})
            continue
        medians = [statistics.median(x) for x in all_gaps]
        if max(medians) / max(min(medians), 1e-9) > config.max_period_ratio:
            rejected.append({"kind": "timing", "id": arb_id, "reason": "period_mode_shift"})
            continue
        def limit(label: str, current_id: str = arb_id) -> float | None:
            groups = [sample.get((current_id, label), []) for sample in gap_sets]
            if any(len(x) < config.min_id_samples for x in groups):
                return None
            return config.timing_multiplier * quantile(groups[0] + groups[1], config.timing_quantile)
        all_limit = limit("all")
        assert all_limit is not None
        timing.append(TimingRule(arb_id, all_limit, limit("low"), limit("high"),
                                 [len(x) for x in all_gaps], medians))
    timing.sort(key=lambda rule: rule.arb_id)
    phase = _pairs(captures, "arrival", config, rejected)
    coupling = _pairs(captures, "change", config, rejected)
    return DynamicsProfile(1, asdict(config), {c.source: c.sha256 for c in captures}, threshold,
                           timing, phase, coupling, rejected)


def score_events(events: list[Event], start: float, end: float, profile: DynamicsProfile) -> list[dict]:
    """Return causal bin scores; each recording starts with empty state."""
    if not events or end < start:
        return []
    cfg = profile.config
    width = cfg["bin_width"]
    timing = {rule.arb_id: rule for rule in profile.timing}
    phase_by_target: dict[str, list[PairRule]] = defaultdict(list)
    change_by_target: dict[str, list[PairRule]] = defaultdict(list)
    for rule in profile.phase:
        phase_by_target[rule.target].append(rule)
    for rule in profile.change_coupling:
        change_by_target[rule.target].append(rule)
    last_arrival: dict[str, float] = {}
    last_change: dict[str, float] = {}
    last_mode: dict[str, str] = {}
    timeout_active: set[str] = set()
    recent: deque[float] = deque()
    rows = []
    cursor = 0
    bin_number = 0
    while start + bin_number * width < end:
        close = min(end, start + (bin_number + 1) * width)
        phase_alerts = coupling_alerts = 0
        while cursor < len(events) and events[cursor].timestamp <= close:
            event = events[cursor]
            cursor += 1
            t = event.timestamp
            if event.kind == "arrival":
                while recent and recent[0] < t - cfg["mode_window"]:
                    recent.popleft()
                mode = "high" if len(recent) >= profile.mode_count_threshold else "low"
                for rule in phase_by_target.get(event.arb_id, []):
                    previous = last_arrival.get(rule.trigger)
                    if previous is not None and (t - previous < rule.lag_low or t - previous > rule.lag_high):
                        phase_alerts += 1
                last_arrival[event.arb_id] = t
                last_mode[event.arb_id] = mode
                timeout_active.discard(event.arb_id)
                recent.append(t)
            elif event.kind == "change":
                for rule in change_by_target.get(event.arb_id, []):
                    previous = last_change.get(rule.trigger)
                    if previous is not None and (t - previous < rule.lag_low or t - previous > rule.lag_high):
                        coupling_alerts += 1
                last_change[event.arb_id] = t
        timing_ids = []
        for arb_id, rule in timing.items():
            stamp = last_arrival.get(arb_id)
            if stamp is None or arb_id in timeout_active:
                continue
            mode = last_mode[arb_id]
            limit = rule.high_limit if mode == "high" else rule.low_limit
            if close > stamp + (limit if limit is not None else rule.all_limit):
                timeout_active.add(arb_id)
                timing_ids.append(arb_id)
        rows.append({"timestamp": close, "timing": len(timing_ids), "phase": phase_alerts,
                     "change_coupling": coupling_alerts, "timing_ids": timing_ids})
        bin_number += 1
    return rows
