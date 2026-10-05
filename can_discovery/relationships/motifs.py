"""Bounded, timestamped CAN-event motifs with explicit temporal edges."""
from __future__ import annotations

import bisect
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..preprocessing.capture import Capture, derive_events


@dataclass(frozen=True, slots=True)
class MotifRule:
    shape: str
    steps: tuple[str, ...]
    edges: tuple[tuple[int, int, float, float], ...]
    max_span: float
    support: tuple[int, ...]
    completion_rate: float
    control_rate: float


@dataclass
class MotifState:
    active: list[tuple[int, tuple[float, ...]]] = field(default_factory=list)
    last_close: float | None = None
    last_dropped: int = 0


def _event_times(capture: Capture) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    for stamp, _ordinal, arb_id, kind in derive_events(capture.frames):
        times[f'{arb_id}|{kind}'].append(stamp)
    return times


def _count(times: dict[str, list[float]], steps: tuple[str, ...],
           edges: tuple[tuple[int, int, float, float], ...], max_span: float,
           sample_cap: int) -> tuple[int, int]:
    roots = times.get(steps[0], [])
    if len(roots) > sample_cap:
        indices = sorted(random.Random(117).sample(range(len(roots)), sample_cap))
        roots = [roots[index] for index in indices]
    complete = 0
    for root in roots:
        matched = [root]
        for index, key in enumerate(steps[1:], 1):
            lower = matched[-1]
            upper = root + max_span
            for src, dst, lo, hi in edges:
                if dst == index:
                    lower = max(lower, matched[src] + lo)
                    upper = min(upper, matched[src] + hi)
            values = times.get(key, [])
            pos = bisect.bisect_right(values, lower)
            if pos == len(values) or values[pos] > upper:
                break
            matched.append(values[pos])
        else:
            complete += 1
    return complete, len(roots)


def _templates(pairs: list[tuple[str, str, float, float]], max_candidates: int):
    """Compose supported directed pairs into a declared shape library."""
    pairs = sorted(set(pairs))
    outgoing: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for a, b, lo, hi in pairs:
        outgoing[a].append((b, lo, hi))
        incoming[b].append((a, lo, hi))
    seen = set()
    for a, b, lo, hi in pairs:
        item = ('pair', (a, b), ((0, 1, lo, hi),))
        if item not in seen:
            seen.add(item)
            yield item
        if len(seen) >= max_candidates:
            return
    for a in sorted(outgoing):
        for b, low_ab, high_ab in outgoing[a]:
            for c, low_bc, high_bc in outgoing.get(b, []):
                if len({a, b, c}) == 3:
                    item = ('chain', (a, b, c), ((0, 1, low_ab, high_ab), (1, 2, low_bc, high_bc)))
                    if item not in seen:
                        seen.add(item)
                        yield item
                        if len(seen) >= max_candidates:
                            return
                    if any(dst == a for dst, _, _ in outgoing.get(c, [])):
                        low_ca, high_ca = next((lo, hi) for dst, lo, hi in outgoing[c] if dst == a)
                        item = ('return', (a, b, c, a),
                                ((0, 1, low_ab, high_ab), (1, 2, low_bc, high_bc),
                                 (2, 3, low_ca, high_ca)))
                        if item not in seen:
                            seen.add(item)
                            yield item
                            if len(seen) >= max_candidates:
                                return
            for c, low_ac, high_ac in outgoing[a]:
                if c != b:
                    item = ('fork', (a, b, c), ((0, 1, low_ab, high_ab), (0, 2, low_ac, high_ac)))
                    if item not in seen:
                        seen.add(item)
                        yield item
                        if len(seen) >= max_candidates:
                            return
        if len(seen) >= max_candidates:
            return
    for c in sorted(incoming):
        for a, low_ac, high_ac in incoming[c]:
            for b, low_bc, high_bc in incoming[c]:
                if a != b:
                    item = ('join', (a, b, c), ((0, 2, low_ac, high_ac), (1, 2, low_bc, high_bc)))
                    if item not in seen:
                        seen.add(item)
                        yield item
                        if len(seen) >= max_candidates:
                            return
        if len(seen) >= max_candidates:
            return


def fit_motifs(captures: list[Capture], pairs: list[tuple[str, str, float, float]], *,
               min_support: int = 20, min_completion: float = .6,
               min_control_gain: float = .1, max_span: float = .5,
               max_candidates: int = 512, max_motifs: int = 128,
               sample_cap: int = 2000) -> tuple[MotifRule, ...]:
    if len(captures) < 2:
        raise ValueError('motif discovery needs independent normal references')
    streams = [_event_times(capture) for capture in captures]
    result = []
    for shape, steps, edges in _templates(pairs, max_candidates):
        true = [_count(times, steps, edges, max_span, sample_cap) for times in streams]
        if not all(hits >= min_support and hits / max(1, roots) >= min_completion
                   for hits, roots in true):
            continue
        controls = []
        for capture, times in zip(captures, streams, strict=True):
            span = capture.end - capture.start
            shifted = dict(times)
            key = steps[1]
            shifted[key] = sorted(capture.start + ((t - capture.start + span / 2) % span)
                                  for t in times.get(key, [])) if span else []
            controls.append(_count(shifted, steps, edges, max_span, sample_cap)[0])
        support = tuple(hits for hits, _ in true)
        total = sum(roots for _, roots in true)
        rate = sum(support) / max(1, total)
        control_rate = sum(controls) / max(1, total)
        if rate < control_rate + min_control_gain:
            continue
        result.append(MotifRule(shape, steps, edges, max_span, support, rate, control_rate))
    result.sort(key=lambda rule: (-(rule.completion_rate - rule.control_rate),
                                  -min(rule.support), rule.shape, rule.steps))
    return tuple(result[:max_motifs])


class MotifTracker:
    @staticmethod
    def step(events: list[tuple[float, int, str, str]], close: float,
             rules: tuple[MotifRule, ...], state: MotifState, *, max_active: int = 128):
        if state.last_close is not None and close <= state.last_close:
            raise ValueError('reset motif state at recording boundary or advance bin close')
        if any(stamp > close or (state.last_close is not None and stamp <= state.last_close)
               for stamp, *_ in events):
            raise ValueError('events outside current bin')
        completed = [0] * len(rules)
        overdue = [0] * len(rules)
        active = state.active
        state.last_dropped = 0
        starts: dict[str, list[int]] = defaultdict(list)
        for index, rule in enumerate(rules):
            starts[rule.steps[0]].append(index)
        def expire(now: float):
            nonlocal active
            kept = []
            for idx, stamps in active:
                rule = rules[idx]
                next_index = len(stamps)
                deadlines = [stamps[0] + rule.max_span]
                deadlines.extend(stamps[src] + hi for src, dst, _lo, hi in rule.edges if dst == next_index)
                if now > min(deadlines):
                    overdue[idx] += 1
                else:
                    kept.append((idx, stamps))
            active = kept
        for stamp, _ordinal, arb_id, kind in events:
            expire(stamp)
            key = f'{arb_id}|{kind}'
            advanced = []
            remaining = []
            for idx, stamps in active:
                rule = rules[idx]
                next_index = len(stamps)
                if key != rule.steps[next_index] or stamp <= stamps[-1] or stamp > stamps[0] + rule.max_span:
                    remaining.append((idx, stamps))
                    continue
                if not all(stamps[src] + lo < stamp <= stamps[src] + hi
                           for src, dst, lo, hi in rule.edges if dst == next_index):
                    remaining.append((idx, stamps))
                    continue
                if next_index + 1 == len(rule.steps):
                    completed[idx] += 1
                else:
                    advanced.append((idx, (*stamps, stamp)))
            active = remaining + advanced
            active.extend((idx, (stamp,)) for idx in starts.get(key, ()))
            if len(active) > max_active:
                state.last_dropped += len(active) - max_active
                active = active[-max_active:]
        expire(close)
        state.active = active
        state.last_close = close
        counts = Counter(idx for idx, _ in active)
        return completed, overdue, [counts[idx] for idx in range(len(rules))], state
