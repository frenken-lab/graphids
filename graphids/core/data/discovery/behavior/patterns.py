"""Reference-conditioned relationships over raw CAN events.

All time values are seconds. Captures are independent; no state crosses files.
"""
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: float
    arb_id: str
    kind: str  # arrival or change
    payload_length: int
    ordinal: int = 0
    raw_payload_chars: int | None = None

    @property
    def key(self) -> str:
        return f"{self.arb_id}|{self.kind}"


@dataclass
class Capture:
    source: str
    sha256: str
    events: list[Event]
    start: float
    end: float
    rows: int
    payload_lengths: dict[int, int]
    decreasing_timestamps: int = 0
    invalid_payloads: int = 0
    raw_payload_chars: dict[int, int] = field(default_factory=dict)


def load_capture(path: Path, *, drop_interval: tuple[str, float, float] | None = None) -> Capture:
    """Compatibility view of the shared parsed capture."""
    from .capture import derive_events
    from .capture import load_capture as read_capture

    raw = read_capture(path, drop_interval=drop_interval)
    lengths = {frame.ordinal: frame.raw_payload_chars for frame in raw.frames}
    events = [Event(stamp, arb_id, kind, lengths[ordinal] // 2, ordinal, lengths[ordinal])
              for stamp, ordinal, arb_id, kind in derive_events(raw.frames)]
    return Capture(raw.source, raw.sha256, events, raw.start, raw.end, len(raw.frames),
                   raw.payload_lengths, raw.decreasing_timestamps, raw.invalid_payloads,
                   raw.raw_payload_chars)


@dataclass(frozen=True)
class MinerConfig:
    max_lag: float = 0.2
    min_support: int = 20
    min_completion: float = 0.6
    min_lift: float = 1.5
    max_streams: int = 64
    max_neighbors: int = 4
    max_patterns: int = 128
    max_active: int = 128
    sample_per_stream: int = 2000
    bin_width: float = 0.1
    alert_threshold: float = 0.5


@dataclass
class Pattern:
    steps: list[str]
    lag_windows: list[list[float]]
    support: int
    completion_rate: float
    uncertainty: float
    reference_support: list[int]
    control_rate: float


@dataclass
class PatternCatalog:
    version: int
    id_keys: list[str]
    event_types: list[str]
    patterns: list[Pattern]
    source_hashes: dict[str, str]
    settings: dict
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "patterns": [asdict(p) for p in self.patterns]}


def _times(capture: Capture, selected: set[str] | None = None) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for event in capture.events:
        if selected is None or event.key in selected:
            result[event.key].append(event.timestamp)
    return result


def _rate(starts: list[float], ends: list[float], lo: float, hi: float, cap: int) -> tuple[int, int, list[float]]:
    if not starts or not ends:
        return 0, 0, []
    stride = max(1, math.ceil(len(starts) / cap))
    sampled = starts[::stride]
    delays = []
    for t in sampled:
        idx = bisect.bisect_right(ends, t + lo)
        if idx < len(ends) and ends[idx] <= t + hi:
            delays.append(ends[idx] - t)
    return len(delays), len(sampled), delays


def _sequence(times: dict[str, list[float]], steps: list[str], windows: list[list[float]], cap: int) -> tuple[int, int]:
    roots = times.get(steps[0], [])
    stride = max(1, math.ceil(len(roots) / cap))
    roots = roots[::stride]
    complete = 0
    for root in roots:
        previous = root
        for key, (lo, hi) in zip(steps[1:], windows, strict=True):
            options = times.get(key, [])
            idx = bisect.bisect_right(options, previous + lo)
            if idx >= len(options) or options[idx] > previous + hi:
                break
            previous = options[idx]
        else:
            complete += 1
    return complete, len(roots)


class PatternMiner:
    @staticmethod
    def fit(normal_captures: list[Capture], config: MinerConfig) -> PatternCatalog:
        if len(normal_captures) < 2:
            raise ValueError("at least two independent normal reference captures required")
        counts = Counter(e.key for c in normal_captures for e in c.events)
        keys = sorted(counts, key=lambda k: (-counts[k], k))[:config.max_streams]
        per_file = [_times(c, set(keys)) for c in normal_captures]
        shifted_file = []
        for capture, streams in zip(normal_captures, per_file, strict=True):
            span = capture.end - capture.start
            shifted_file.append({key: sorted(capture.start + ((t - capture.start + span / 2) % span) for t in values) if span else []
                                 for key, values in streams.items()})
        candidates = []
        retained: dict[tuple[str, str], tuple[list[float], int, float, list[int], float]] = {}
        for a in keys:
            ranked = []
            for b in keys:
                if a == b:
                    continue
                supports, totals, controls, delays = [], [], [], []
                for ts, shifted_streams in zip(per_file, shifted_file, strict=True):
                    starts, ends = ts.get(a, []), ts.get(b, [])
                    hit, total, lag = _rate(starts, ends, 0.0, config.max_lag, config.sample_per_stream)
                    # Half-capture circular shift breaks local phase while preserving B's rate.
                    control, _, _ = _rate(starts, shifted_streams.get(b, []), 0.0, config.max_lag, config.sample_per_stream)
                    supports.append(hit)
                    totals.append(total)
                    controls.append(control)
                    delays.extend(lag)
                rate = sum(supports) / max(1, sum(totals))
                control_rate = sum(controls) / max(1, sum(totals))
                stable = all(n >= config.min_support and n / max(1, d) >= config.min_completion
                             for n, d in zip(supports, totals, strict=True))
                accepted = stable and rate >= config.min_completion and rate >= config.min_lift * max(control_rate, 1 / max(1, sum(totals)))
                reason = "retained" if accepted else ("unsupported_reference" if not stable else "shift_control")
                candidates.append({"from": a, "to": b, "support": supports, "rate": rate,
                                   "control_rate": control_rate, "reason": reason})
                if accepted:
                    delays.sort()
                    upper = min(config.max_lag, max(delays[-1], delays[int(0.99 * (len(delays) - 1))] * 1.1))
                    ranked.append((rate - control_rate, b, [0.0, upper], sum(supports), rate, supports, control_rate))
            ranked.sort(key=lambda x: (-x[0], x[1]))
            for _, b, window, n, rate, supports, control in ranked[:config.max_neighbors]:
                retained[a, b] = (window, n, rate, supports, control)
            for _, b, *_ in ranked[config.max_neighbors:]:
                next(row for row in candidates if row["from"] == a and row["to"] == b)["reason"] = "neighbor_cap"
        patterns: list[Pattern] = []
        for (a, b), (window, n, rate, supports, control) in retained.items():
            patterns.append(Pattern([a, b], [window], n, rate,
                                    math.sqrt(rate * (1 - rate) / max(1, n)), supports, control))
        # Compose only along the sparse retained graph. A cycle may return to A.
        composed = 0
        for a, b in sorted(retained):
            for b2, c in sorted(retained):
                if b2 != b or c in {a, b}:
                    continue
                chains = [[a, b, c]]
                if (c, a) in retained:
                    chains.append([a, b, c, a])
                for steps in chains:
                    if composed >= config.max_patterns * 4:
                        candidates.append({"steps": steps, "reason": "composition_cap"})
                        continue
                    composed += 1
                    windows = [retained[x, y][0] for x, y in zip(steps, steps[1:], strict=False)]
                    per = [_sequence(ts, steps, windows, config.sample_per_stream)
                           for ts in per_file]
                    support = [x for x, _ in per]
                    total = sum(y for _, y in per)
                    rate = sum(support) / max(1, total)
                    if all(x >= config.min_support and x / max(1, y) >= config.min_completion for x, y in per):
                        patterns.append(Pattern(steps, windows, sum(support), rate,
                                                math.sqrt(rate * (1 - rate) / max(1, total)), support, 0.0))
                    else:
                        candidates.append({"steps": steps, "support": support, "rate": rate,
                                           "reason": "unsupported_composition"})
        patterns.sort(key=lambda p: (-len(p.steps), -p.completion_rate, -p.support, p.steps))
        for pattern in patterns[config.max_patterns:]:
            candidates.append({"steps": pattern.steps, "reason": "pattern_cap"})
        return PatternCatalog(1, sorted({k.split("|")[0] for k in keys}), ["arrival", "change"],
                              patterns[:config.max_patterns], {c.source: c.sha256 for c in normal_captures},
                              asdict(config), candidates)


@dataclass
class TrackerState:
    active: list[tuple[int, int, float]] = field(default_factory=list)  # pattern, next step, previous timestamp
    last_close: float | None = None


class PatternTracker:
    @staticmethod
    def step(events_through_bin_close: list[Event], catalog: PatternCatalog, state: TrackerState,
             bin_close: float | None = None) -> tuple[dict, dict, TrackerState]:
        if bin_close is None:
            if not events_through_bin_close:
                raise ValueError("bin_close required for empty bins")
            bin_close = events_through_bin_close[-1].timestamp
        if state.last_close is not None and bin_close < state.last_close:
            raise ValueError("reset tracker state at recording boundary")
        if any(e.timestamp > bin_close or (state.last_close is not None and e.timestamp <= state.last_close)
               for e in events_through_bin_close):
            raise ValueError("events must belong to current bin and precede close")
        n = len(catalog.patterns)
        completed = [0] * n
        overdue = [0] * n
        late = [0.0] * n
        id_completed = Counter()
        id_overdue = Counter()
        id_lateness: dict[str, float] = defaultdict(float)
        prefix_dropped = 0
        def expire(now: float) -> None:
            remaining = []
            for idx, next_step, previous in state.active:
                deadline = previous + catalog.patterns[idx].lag_windows[next_step - 1][1]
                if now > deadline:
                    overdue[idx] += 1
                    late[idx] = max(late[idx], now - deadline)
                    target_id = catalog.patterns[idx].steps[next_step].split("|")[0]
                    id_overdue[target_id] += 1
                    id_lateness[target_id] = max(id_lateness[target_id], now - deadline)
                else:
                    remaining.append((idx, next_step, previous))
            state.active = remaining
        for event in events_through_bin_close:
            expire(event.timestamp)
            key = event.key
            advanced = []
            for idx, next_step, previous in state.active:
                pattern = catalog.patterns[idx]
                lo, hi = pattern.lag_windows[next_step - 1]
                if key == pattern.steps[next_step] and previous + lo < event.timestamp <= previous + hi:
                    if next_step + 1 == len(pattern.steps):
                        completed[idx] += 1
                        for completed_id in {step.split("|")[0] for step in pattern.steps}:
                            id_completed[completed_id] += 1
                    else:
                        advanced.append((idx, next_step + 1, event.timestamp))
            state.active.extend(advanced)
            for idx, pattern in enumerate(catalog.patterns):
                if key == pattern.steps[0]:
                    state.active.append((idx, 1, event.timestamp))
            if len(state.active) > catalog.settings["max_active"]:
                prefix_dropped += len(state.active) - catalog.settings["max_active"]
                state.active = state.active[-catalog.settings["max_active"]:]
        expire(bin_close)
        state.last_close = bin_close
        active = [0] * n
        for idx, _, _ in state.active:
            active[idx] += 1
        id_keys = catalog.id_keys
        features = {"timestamp": bin_close, "id_keys": id_keys, "prefix_dropped": prefix_dropped,
                    "id_mask": [1] * len(id_keys) + [0] * (catalog.settings["max_streams"] - len(id_keys)),
                    "id_completed": [id_completed[k] for k in id_keys] + [0] * (catalog.settings["max_streams"] - len(id_keys)),
                    "id_overdue": [id_overdue[k] for k in id_keys] + [0] * (catalog.settings["max_streams"] - len(id_keys)),
                    "id_lateness": [id_lateness[k] for k in id_keys] + [0.0] * (catalog.settings["max_streams"] - len(id_keys)),
                    "relationship_mask": [1] * n + [0] * (catalog.settings["max_patterns"] - n),
                    "relationship_completed": completed + [0] * (catalog.settings["max_patterns"] - n),
                    "relationship_overdue": overdue + [0] * (catalog.settings["max_patterns"] - n),
                    "relationship_lateness": late + [0.0] * (catalog.settings["max_patterns"] - n),
                    "relationship_active": active + [0] * (catalog.settings["max_patterns"] - n)}
        score = sum(overdue[i] * max(0.0, catalog.patterns[i].completion_rate - catalog.patterns[i].control_rate)
                    for i in range(n))
        return features, {"timestamp": bin_close, "pattern_score": score,
                          "pattern_alert": int(score >= catalog.settings["alert_threshold"])}, state
