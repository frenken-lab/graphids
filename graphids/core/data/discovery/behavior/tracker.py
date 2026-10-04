"""Causal, fixed-slot features from a frozen behavior profile."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field

from .bitfields import extract
from .capture import Capture, Frame
from .motifs import MotifState, MotifTracker
from .profile import BehaviorProfile

NODE_CHANNELS = ('arrival_count', 'change_count', 'age_seconds', 'payload_bit_flips',
                 'payload_deviation', 'run_length', 'run_deviation',
                 'change_rate_deviation', 'field_violations')
RELATION_CHANNELS = ('target_count', 'lag_violations')
MOTIF_CHANNELS = ('completed', 'overdue', 'active_prefixes')


@dataclass(frozen=True, slots=True)
class BehaviorFeatures:
    timestamp: float
    node_keys: tuple[str, ...]
    node: tuple[tuple[float, ...], ...]
    node_mask: tuple[bool, ...]
    relation: tuple[tuple[float, ...], ...]
    relation_mask: tuple[bool, ...]
    motif: tuple[tuple[float, ...], ...]
    motif_mask: tuple[bool, ...]
    prefix_dropped: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BehaviorScores:
    timing: float
    payload: float
    field: float
    relation: float
    motif: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorState:
    last_close: float | None = None
    previous_payload: dict[str, bytes | None] = field(default_factory=dict)
    run_length: dict[str, int] = field(default_factory=dict)
    run_alerted: set[str] = field(default_factory=set)
    recent_changes: dict[str, deque[int]] = field(default_factory=dict)
    last_arrival: dict[str, float] = field(default_factory=dict)
    last_change: dict[str, float] = field(default_factory=dict)
    last_mode: dict[str, str] = field(default_factory=dict)
    timeout_active: set[str] = field(default_factory=set)
    recent_arrivals: deque[float] = field(default_factory=deque)
    motif: MotifState = field(default_factory=MotifState)


class BehaviorTracker:
    @staticmethod
    def step(frames: list[Frame] | tuple[Frame, ...], bin_close: float,
             profile: BehaviorProfile, state: BehaviorState) -> tuple[BehaviorFeatures, BehaviorScores, BehaviorState]:
        if state.last_close is not None and bin_close <= state.last_close:
            raise ValueError('reset tracker state at recording boundary or advance bin close')
        if any(frame.timestamp > bin_close or
               (state.last_close is not None and frame.timestamp <= state.last_close)
               for frame in frames):
            raise ValueError('frames outside current bin')
        if any((a.timestamp, a.ordinal) > (b.timestamp, b.ordinal)
               for a, b in zip(frames, frames[1:])):
            raise ValueError('frames must be sorted by timestamp and source ordinal')
        arrivals: Counter[str] = Counter()
        changes: Counter[str] = Counter()
        bit_flips: Counter[str] = Counter()
        payload_deviation: Counter[str] = Counter()
        run_deviation: Counter[str] = Counter()
        change_rate_deviation: Counter[str] = Counter()
        field_violations: Counter[str] = Counter()
        relation_targets = [0] * len(profile.relations)
        relation_violations = [0] * len(profile.relations)
        payload_rules = {rule.arb_id: rule for rule in profile.payload}
        fields_by_id = defaultdict(list)
        relations_by_key = defaultdict(list)
        for rule in profile.fields:
            fields_by_id[rule.arb_id].append(rule)
        for index, rule in enumerate(profile.relations):
            relations_by_key[(rule.target, rule.kind)].append((index, rule))
        events = []
        for frame in frames:
            stamp, arb_id = frame.timestamp, frame.arb_id
            previous = state.previous_payload.get(arb_id)
            payload_rule = payload_rules.get(arb_id)
            if frame.payload is None:
                state.run_length.pop(arb_id, None)
                state.run_alerted.discard(arb_id)
                state.recent_changes.pop(arb_id, None)
            elif previous is not None and previous == frame.payload:
                state.run_length[arb_id] = state.run_length.get(arb_id, 1) + 1
            else:
                state.run_length[arb_id] = 1
                state.run_alerted.discard(arb_id)
            if payload_rule is not None and previous is not None and frame.payload is not None:
                history = state.recent_changes.setdefault(arb_id, deque(maxlen=32))
                history.append(int(previous != frame.payload))
                if len(history) >= 16 and abs(sum(history) / len(history) - payload_rule.change_rate) > .3:
                    change_rate_deviation[arb_id] += 1
                if state.run_length[arb_id] > payload_rule.run_limit and arb_id not in state.run_alerted:
                    run_deviation[arb_id] += 1
                    state.run_alerted.add(arb_id)
            for kind in ('arrival', 'change'):
                if kind == 'change' and (previous is None or frame.payload is None or previous == frame.payload):
                    continue
                events.append((stamp, frame.ordinal, arb_id, kind))
                last = state.last_arrival if kind == 'arrival' else state.last_change
                for index, rule in relations_by_key.get((arb_id, kind), ()):
                    before = last.get(rule.trigger)
                    if before is not None:
                        relation_targets[index] += 1
                        lag = stamp - before
                        if not rule.lag_low <= lag <= rule.lag_high:
                            relation_violations[index] += 1
                if kind == 'arrival':
                    while state.recent_arrivals and state.recent_arrivals[0] < stamp - profile.config.dynamics.mode_window:
                        state.recent_arrivals.popleft()
                    state.last_mode[arb_id] = ('high' if len(state.recent_arrivals) >=
                                               profile.mode_count_threshold else 'low')
                    state.recent_arrivals.append(stamp)
                    state.last_arrival[arb_id] = stamp
                    state.timeout_active.discard(arb_id)
                    arrivals[arb_id] += 1
                else:
                    state.last_change[arb_id] = stamp
                    changes[arb_id] += 1
            if previous is not None and frame.payload is not None:
                if len(previous) == len(frame.payload):
                    distance = (int.from_bytes(previous, 'big') ^
                                int.from_bytes(frame.payload, 'big')).bit_count()
                    bit_flips[arb_id] += distance
                    rule = payload_rules.get(arb_id)
                    if rule is not None and distance > rule.hamming_limit:
                        payload_deviation[arb_id] += 1
                any_field_violation = False
                for rule in fields_by_id.get(arb_id, ()) if len(previous) == len(frame.payload) else ():
                    spec = (rule.start, rule.width, rule.endian, rule.signed)
                    before, after = extract(previous, *spec), extract(frame.payload, *spec)
                    if before is None or after is None:
                        continue
                    violated = (before != after and (after - before) % (1 << rule.width) != 1)
                    if rule.role == 'smooth':
                        violated = abs(after - before) > rule.limit
                    any_field_violation |= violated
                field_violations[arb_id] += any_field_violation
            state.previous_payload[arb_id] = frame.payload
        completed, overdue, active, state.motif = MotifTracker.step(
            events, bin_close, profile.motifs, state.motif,
            max_active=profile.config.max_active_motifs)
        timed_out = Counter()
        for rule in profile.timing:
            stamp = state.last_arrival.get(rule.arb_id)
            if stamp is None or rule.arb_id in state.timeout_active:
                continue
            mode = state.last_mode[rule.arb_id]
            limit = rule.high_limit if mode == 'high' else rule.low_limit
            if bin_close > stamp + (limit if limit is not None else rule.all_limit):
                timed_out[rule.arb_id] += 1
                state.timeout_active.add(rule.arb_id)
        node = []
        for key in profile.node_keys:
            age = bin_close - state.last_arrival[key] if key in state.last_arrival else 0.0
            node.append((float(arrivals[key]), float(changes[key]), age,
                         float(bit_flips[key]), float(payload_deviation[key]),
                         float(state.run_length.get(key, 0)), float(run_deviation[key]),
                         float(change_rate_deviation[key]), float(field_violations[key])))
        node.extend([(0.0,) * len(NODE_CHANNELS)] * (profile.config.max_nodes - len(node)))
        relation = [(float(n), float(v)) for n, v in zip(relation_targets, relation_violations)]
        relation.extend([(0.0,) * len(RELATION_CHANNELS)] * (profile.config.max_relations - len(relation)))
        motif = [(float(c), float(o), float(a)) for c, o, a in zip(completed, overdue, active)]
        motif.extend([(0.0,) * len(MOTIF_CHANNELS)] * (profile.config.max_motifs - len(motif)))
        features = BehaviorFeatures(bin_close, profile.node_keys, tuple(node),
                                    (True,) * len(profile.node_keys) +
                                    (False,) * (profile.config.max_nodes - len(profile.node_keys)),
                                    tuple(relation), (True,) * len(profile.relations) +
                                    (False,) * (profile.config.max_relations - len(profile.relations)),
                                    tuple(motif), (True,) * len(profile.motifs) +
                                    (False,) * (profile.config.max_motifs - len(profile.motifs)),
                                    state.motif.last_dropped)
        scores = BehaviorScores(float(sum(timed_out.values())),
                                float(sum(payload_deviation.values()) + sum(run_deviation.values()) +
                                      sum(change_rate_deviation.values())),
                                float(sum(field_violations.values())),
                                float(sum(relation_violations)),
                                sum(overdue[i] * max(0.0, rule.completion_rate - rule.control_rate)
                                    for i, rule in enumerate(profile.motifs)))
        state.last_close = bin_close
        return features, scores, state


def score_capture(capture: Capture, profile: BehaviorProfile):
    """Yield one feature/score pair per physical-time bin, including silent bins."""
    if not capture.frames:
        return
    state = BehaviorState()
    origin = capture.start
    cursor = 0
    bin_number = 0
    while origin + bin_number * profile.config.bin_width < capture.end:
        close = min(capture.end, origin + (bin_number + 1) * profile.config.bin_width)
        batch = []
        while cursor < len(capture.frames) and capture.frames[cursor].timestamp <= close:
            batch.append(capture.frames[cursor])
            cursor += 1
        yield BehaviorTracker.step(batch, close, profile, state)[:2]
        bin_number += 1
    if cursor < len(capture.frames) or capture.start == capture.end:
        yield BehaviorTracker.step(capture.frames[cursor:], capture.end + profile.config.bin_width,
                                   profile, state)[:2]
