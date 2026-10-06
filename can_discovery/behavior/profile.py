"""Frozen reference profile for predictive CAN behavior discovery."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from ..decoding.signals import (
    FieldCandidate,
    UnresolvedSpan,
    discover_fields,
    unresolved_spans,
)
from ..preprocessing.capture import Capture
from ..preprocessing.message_types import MessageType, classify_messages
from ..relationships.behavior.dynamics import (
    DynamicsConfig,
    PairRule,
    TimingRule,
    fit_dynamics,
)
from ..relationships.behavior.motifs import MotifRule, fit_motifs
from ..relationships.physical.combination_ranker import fit_physical_roles
from ..relationships.physical.role_types import CandidateCombination, RoleHypothesis
from ..relationships.physical.source_calibration import (
    SourceFactorValidation,
    SourceRoleCalibration,
)
from ..relationships.shared.evidence import FactorEvidence
from .payload import PayloadRule, fit_payload


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
    field_tokenizer: str = 'can_d'
    field_max_layouts: int = 1
    field_min_changes: int = 20
    max_fields_per_id: int = 4
    physical_enabled: bool = True
    physical_max_role_candidates: int = 64
    physical_max_family_candidates: int = 8
    physical_beam_width: int = 64
    physical_max_combinations: int = 8
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    def __post_init__(self):
        if self.bin_width <= 0 or any(value <= 0 for value in
            (self.max_nodes, self.max_relations, self.max_motifs, self.max_active_motifs,
             self.physical_max_role_candidates, self.physical_max_family_candidates,
             self.physical_beam_width, self.physical_max_combinations)):
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
    unresolved_spans: tuple[UnresolvedSpan, ...] = ()
    message_families: tuple[MessageType, ...] = ()
    physical_hypotheses: tuple[RoleHypothesis, ...] = ()
    physical_combinations: tuple[CandidateCombination, ...] = ()
    physical_rejected: tuple[dict, ...] = ()
    source_role_calibration: tuple[SourceRoleCalibration, ...] = ()
    physical_factor_evidence: tuple[FactorEvidence, ...] = ()
    source_factor_validation: tuple[SourceFactorValidation, ...] = ()
    tokenization_evidence: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorProfile:
        if data.get('version') != 11:
            raise ValueError('behavior profile version is unsupported; regenerate a version-11 profile')
        raw_cfg = dict(data['config'])
        raw_cfg['dynamics'] = DynamicsConfig(**raw_cfg['dynamics'])
        def field_from_dict(item):
            return FieldCandidate(item['arb_id'], item['start'], item['width'], item['endian'],
                                  item['signed'], item['role'], item['limit'], item['confidence'],
                                  tuple(item['reference_support']),
                                  tuple(tuple(x) for x in item['label_hypotheses']),
                                  tuple(item['evidence']), item['payload_length'],
                                  item.get('mux_start'), item.get('mux_value'), item.get('bus'))
        return cls(data['version'], BehaviorConfig(**raw_cfg), tuple(tuple(x) for x in data['sources']),
                   tuple(data['node_keys']),
                   tuple(TimingRule(**item) for item in data['timing']),
                   tuple(PayloadRule(item['arb_id'], tuple(item['reference_transitions']),
                                     item['hamming_limit'], item['change_rate'], item['run_limit'],
                                     tuple(item['changed_bit_fraction'])) for item in data['payload']),
                   tuple(field_from_dict(item) for item in data['fields']),
                   tuple(PairRule(**item) for item in data['relations']),
                   tuple(MotifRule(item['shape'], tuple(item['steps']),
                                   tuple(tuple(edge) for edge in item['edges']), item['max_span'],
                                   tuple(item['support']), item['completion_rate'], item['control_rate'])
                         for item in data['motifs']), data['mode_count_threshold'],
                   tuple(data.get('rejected', ())),
                   tuple(UnresolvedSpan(**item) for item in data.get('unresolved_spans', ())),
                   tuple(MessageType(**item) for item in data.get('message_families', ())),
                   tuple(RoleHypothesis(field_from_dict(item['candidate']), item['role'],
                                        item['unary_score'], tuple(item['support']),
                                        tuple(item['evidence']), item['equivalence_group'],
                                        item.get('update_limit', 0.0),
                                        tuple(item.get('allowed_values', ())),
                                        item.get('calibration', 'physics_only'))
                         for item in data.get('physical_hypotheses', ())),
                   tuple(CandidateCombination(tuple(item['assignments']), item['weight'],
                                              item['score'],
                                              tuple(tuple(edge) for edge in item['relation_evidence']),
                                              tuple(FactorEvidence(**{**factor, 'roles': tuple(factor['roles']),
                                                                      'members': tuple(factor.get('members', ()))})
                                                    for factor in item.get('factors', ())),
                                              item.get('runtime_weight'))
                         for item in data.get('physical_combinations', ())),
                   tuple(data.get('physical_rejected', ())),
                   tuple(SourceRoleCalibration(item['role'], item['positive_mean'],
                                               item['negative_mean'], item['scale'],
                                               tuple(item['source_vehicles']),
                                               item['positive_count'], item['negative_count'],
                                               item.get('inclusion_threshold', 0.0),
                                               tuple(tuple(check) for check in item.get('leave_one_vehicle_out_recall', ())))
                         for item in data.get('source_role_calibration', ())),
                   tuple(FactorEvidence(**{**item, 'roles': tuple(item['roles']),
                                           'members': tuple(item.get('members', ()))})
                         for item in data.get('physical_factor_evidence', ())),
                   tuple(SourceFactorValidation(item['name'],
                                                tuple(item['supporting_vehicles']), item['weight'],
                                                tuple(tuple(signature) for signature in item['signatures']))
                         for item in data.get('source_factor_validation', ())),
                   tuple(data.get('tokenization_evidence', ())))


def fit_profile(reference_captures: list[Capture], config: BehaviorConfig | None = None,
                *, source_calibration: tuple[SourceRoleCalibration, ...] = (),
                validated_factors: tuple[SourceFactorValidation, ...] = (),
                source_model=None, target_vehicle: str | None = None) -> BehaviorProfile:
    """Fit only from independent, trusted normal capture inputs."""
    config = config or BehaviorConfig()
    if len(reference_captures) != 2:
        raise ValueError('two independent normal references are required')
    hashes = [capture.sha256 for capture in reference_captures]
    if len(set(hashes)) != len(hashes):
        raise ValueError('reference captures have identical hashes')
    if config.field_tokenizer == 'supervised_can_d':
        if source_model is None:
            raise ValueError('supervised profile requires a source boundary model')
        source_model.check_target(hashes, target_vehicle)
    elif source_model is not None:
        raise ValueError('source boundary model requires supervised_can_d tokenizer')
    if any(any(frame.attack for frame in capture.frames) for capture in reference_captures):
        raise ValueError('reference capture contains labeled attack frames')
    counts = Counter(frame.arb_id for capture in reference_captures for frame in capture.frames)
    nodes = tuple(sorted(sorted(counts, key=lambda key: (-counts[key], key))[:config.max_nodes]))
    selected = set(nodes)
    dynamics = fit_dynamics(reference_captures, config.dynamics)
    timing = tuple(rule for rule in dynamics.timing if rule.arb_id in selected)
    payload = tuple(rule for rule in fit_payload(reference_captures,
                    min_transitions=config.payload_min_transitions) if rule.arb_id in selected)
    registry = discover_fields(reference_captures, max_ids=config.max_nodes,
                               samples_per_id=config.field_samples_per_id,
                               method=config.field_tokenizer,
                               source_model=source_model,
                               max_layouts=config.field_max_layouts,
                               min_changes=config.field_min_changes,
                               max_fields_per_id=config.max_fields_per_id)
    discovered = tuple(rule for rule in registry.selected if rule.arb_id in selected)
    fields = discovered
    physical = (fit_physical_roles(reference_captures, max_ids=config.max_nodes,
                samples_per_id=config.field_samples_per_id,
                max_role_candidates=config.physical_max_role_candidates,
                max_family_candidates=config.physical_max_family_candidates,
                beam_width=config.physical_beam_width,
                max_combinations=config.physical_max_combinations,
                protocol_fields=discovered,
                candidates=registry.candidates,
                source_calibration=source_calibration,
                validated_factors=validated_factors)
                if config.physical_enabled else None)
    unresolved = unresolved_spans(fields, reference_captures)
    relations = tuple(rule for rule in (*dynamics.phase, *dynamics.change_coupling)
                      if rule.trigger in selected and rule.target in selected)[:config.max_relations]
    pairs = [(f'{rule.trigger}|{rule.kind}', f'{rule.target}|{rule.kind}',
              rule.lag_low, rule.lag_high) for rule in relations]
    motifs = fit_motifs(reference_captures, pairs, min_support=config.motif_min_support,
                        min_completion=config.motif_min_completion,
                        min_control_gain=config.motif_min_control_gain,
                        max_span=config.max_motif_span, max_motifs=config.max_motifs)
    rejected = list(dynamics.rejected)
    families = classify_messages(reference_captures)
    tokenization_evidence = tuple({'id': result.trace.arb_id,
        'payload_length': result.trace.payload_length, 'bus': result.trace.bus,
        'method': result.method, 'parameters': result.parameters,
        'source_hashes': list(result.trace.source_hashes), 'support': list(result.trace.support),
        'spans': [{'physical_bits': list(span.physical_bits), 'byte_order': span.byte_order}
                  for span in result.spans], 'uncovered_bits': list(result.uncovered_bits),
        'abstention_reasons': list(result.abstention_reasons)}
        for result in registry.tokenizations) + registry.conflicts
    return BehaviorProfile(11, config,
                           tuple((capture.source, capture.sha256) for capture in reference_captures),
                           nodes, timing, payload, fields, relations, motifs,
                           dynamics.mode_count_threshold, tuple(rejected), unresolved,
                           families,
                           physical.hypotheses if physical else (),
                           physical.combinations if physical else (),
                           physical.rejected if physical else (), source_calibration,
                           physical.factor_evidence if physical else (), validated_factors,
                           tokenization_evidence)
