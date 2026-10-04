"""Frozen reference profile for causal CAN behavior discovery."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .capture import Capture, derive_events
from .dynamics import DynamicsConfig, PairRule, TimingRule, fit_dynamics
from .motifs import MotifRule, fit_motifs
from .patterns import Capture as EventCapture
from .patterns import Event
from .payload import FieldCandidate, PayloadRule, fit_fields, fit_payload


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    bin_width: float = .1
    max_nodes: int = 64
    max_relations: int = 32
    max_motifs: int = 128
    max_active_motifs: int = 128
    max_motif_span: float = .5
    motif_min_support: int = 20
    motif_min_completion: float = .6
    motif_min_control_gain: float = .1
    payload_min_transitions: int = 20
    field_samples_per_id: int = 512
    field_min_changes: int = 20
    max_fields_per_id: int = 4
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    def __post_init__(self):
        if self.bin_width <= 0 or any(value <= 0 for value in
            (self.max_nodes, self.max_relations, self.max_motifs, self.max_active_motifs)):
            raise ValueError('bin width and capacities must be positive')


@dataclass(frozen=True, slots=True)
class BehaviorProfile:
    version: int
    config: BehaviorConfig
    sources: tuple[tuple[str, str], ...]
    node_keys: tuple[str, ...]
    timing: tuple[TimingRule, ...]
    payload: tuple[PayloadRule, ...]
    fields: tuple[FieldCandidate, ...]
    relations: tuple[PairRule, ...]
    motifs: tuple[MotifRule, ...]
    mode_count_threshold: float
    rejected: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorProfile:
        if data.get('version') != 2:
            raise ValueError('unsupported behavior profile version')
        raw_cfg = dict(data['config'])
        raw_cfg['dynamics'] = DynamicsConfig(**raw_cfg['dynamics'])
        return cls(2, BehaviorConfig(**raw_cfg), tuple(tuple(x) for x in data['sources']),
                   tuple(data['node_keys']),
                   tuple(TimingRule(**item) for item in data['timing']),
                   tuple(PayloadRule(item['arb_id'], tuple(item['reference_transitions']),
                                     item['hamming_limit'], item['change_rate'], item['run_limit'],
                                     tuple(item['changed_bit_fraction'])) for item in data['payload']),
                   tuple(FieldCandidate(item['arb_id'], item['start'], item['width'], item['endian'],
                                        item['signed'], item['role'], item['limit'], item['confidence'],
                                        tuple(item['reference_support']),
                                        tuple(tuple(x) for x in item['label_hypotheses']),
                                        tuple(item['evidence'])) for item in data['fields']),
                   tuple(PairRule(**item) for item in data['relations']),
                   tuple(MotifRule(item['shape'], tuple(item['steps']),
                                   tuple(tuple(edge) for edge in item['edges']), item['max_span'],
                                   tuple(item['support']), item['completion_rate'], item['control_rate'])
                         for item in data['motifs']), data['mode_count_threshold'],
                   tuple(data.get('rejected', ())))


def _legacy_capture(capture: Capture) -> EventCapture:
    lengths = {frame.ordinal: frame.raw_payload_chars for frame in capture.frames}
    events = [Event(stamp, arb_id, kind, lengths[ordinal] // 2, ordinal, lengths[ordinal])
              for stamp, ordinal, arb_id, kind in derive_events(capture.frames)]
    return EventCapture(capture.source, capture.sha256, events, capture.start, capture.end,
                        len(capture.frames), capture.payload_lengths,
                        capture.decreasing_timestamps, capture.invalid_payloads,
                        capture.raw_payload_chars)


def fit_profile(reference_captures: list[Capture], config: BehaviorConfig | None = None) -> BehaviorProfile:
    """Fit only from independent, trusted normal capture inputs."""
    config = config or BehaviorConfig()
    if len(reference_captures) != 2:
        raise ValueError('two independent normal references are required')
    hashes = [capture.sha256 for capture in reference_captures]
    if len(set(hashes)) != len(hashes):
        raise ValueError('reference captures have identical hashes')
    if any(any(frame.attack for frame in capture.frames) for capture in reference_captures):
        raise ValueError('reference capture contains labeled attack frames')
    counts = Counter(frame.arb_id for capture in reference_captures for frame in capture.frames)
    nodes = tuple(sorted(sorted(counts, key=lambda key: (-counts[key], key))[:config.max_nodes]))
    selected = set(nodes)
    dynamics = fit_dynamics([_legacy_capture(capture) for capture in reference_captures], config.dynamics)
    timing = tuple(rule for rule in dynamics.timing if rule.arb_id in selected)
    payload = tuple(rule for rule in fit_payload(reference_captures,
                    min_transitions=config.payload_min_transitions) if rule.arb_id in selected)
    fields = tuple(rule for rule in fit_fields(reference_captures, max_ids=config.max_nodes,
                   samples_per_id=config.field_samples_per_id, min_changes=config.field_min_changes,
                   max_fields_per_id=config.max_fields_per_id) if rule.arb_id in selected)
    relations = tuple(rule for rule in (*dynamics.phase, *dynamics.change_coupling)
                      if rule.trigger in selected and rule.target in selected)[:config.max_relations]
    pairs = [(f'{rule.trigger}|{rule.kind}', f'{rule.target}|{rule.kind}',
              rule.lag_low, rule.lag_high) for rule in relations]
    motifs = fit_motifs(reference_captures, pairs, min_support=config.motif_min_support,
                        min_completion=config.motif_min_completion,
                        min_control_gain=config.motif_min_control_gain,
                        max_span=config.max_motif_span, max_motifs=config.max_motifs)
    return BehaviorProfile(2, config,
                           tuple((capture.source, capture.sha256) for capture in reference_captures),
                           nodes, timing, payload, fields, relations, motifs,
                           dynamics.mode_count_threshold, tuple(dynamics.rejected))
