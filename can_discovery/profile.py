"""Frozen reference profile for predictive CAN behavior discovery."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path

from .decoding.signals import (
    FieldCandidate,
    UnresolvedSpan,
    discover_fields,
    unresolved_spans,
)
from .payload import PayloadRule, fit_payload
from .preprocessing.capture import Capture
from .preprocessing.message_types import MessageType, classify_messages
from .relationships.dynamics import DynamicsConfig, PairRule, TimingRule, fit_dynamics
from .relationships.field_analysis import FieldEdge, predictive_edges
from .relationships.motifs import MotifRule, fit_motifs
from .relationships.physical_roles import (
    CandidateCombination,
    RoleHypothesis,
    SourceFactorValidation,
    SourceRoleCalibration,
    fit_physical_roles,
)
from .relationships.physics_factors import FactorEvidence
from .relationships.candidate_graph import (GraphWeights, field_key, layout_memberships,
                                             rerank, selected_layout_fields)


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
    field_graph_enabled: bool = False
    max_field_edges: int = 32
    physical_enabled: bool = True
    physical_max_role_candidates: int = 64
    physical_max_family_candidates: int = 8
    physical_beam_width: int = 64
    physical_max_combinations: int = 8
    candidate_graph_artifact: str | None = None
    dbc_paths: tuple[str, ...] = ()
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
    field_edges: tuple[FieldEdge, ...] = ()
    unresolved_spans: tuple[UnresolvedSpan, ...] = ()
    message_families: tuple[MessageType, ...] = ()
    layout_evidence: tuple[dict, ...] = ()
    role_evidence: tuple[dict, ...] = ()
    semantic_evidence: tuple[dict, ...] = ()
    physical_hypotheses: tuple[RoleHypothesis, ...] = ()
    physical_combinations: tuple[CandidateCombination, ...] = ()
    physical_rejected: tuple[dict, ...] = ()
    source_role_calibration: tuple[SourceRoleCalibration, ...] = ()
    physical_factor_evidence: tuple[FactorEvidence, ...] = ()
    source_factor_validation: tuple[SourceFactorValidation, ...] = ()
    tokenization_evidence: tuple[dict, ...] = ()
    candidate_graph_evidence: tuple[dict, ...] = ()
    candidate_graph_layouts: tuple[dict, ...] = ()
    candidate_graph_memberships: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorProfile:
        if data.get('version') not in (7, 8, 9):
            raise ValueError('behavior profile version is unsupported; regenerate a version-7, version-8, or version-9 profile')
        raw_cfg = dict(data['config'])
        raw_cfg.pop('field_use_canlab', None)
        raw_cfg['dynamics'] = DynamicsConfig(**raw_cfg['dynamics'])
        raw_cfg['dbc_paths'] = tuple(raw_cfg.get('dbc_paths', ()))
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
                   tuple(FieldEdge(item['source'], item['target'], item['relation'],
                                   item['lag_seconds'], item['effect'], item['q_value'],
                                   tuple(item['support']), item['step_seconds'],
                                   tuple(item['provenance']))
                         for item in data.get('field_edges', ())),
                   tuple(UnresolvedSpan(**item) for item in data.get('unresolved_spans', ())),
                   tuple(MessageType(**item) for item in data.get('message_families', ())),
                   tuple({**item, 'evidence': tuple(item['evidence'])}
                         for item in data.get('layout_evidence', ())),
                   tuple({**item, 'support': tuple(item['support'])}
                         for item in data.get('role_evidence', ())),
                   tuple({**item, 'labels': tuple(tuple(label) for label in item['labels'])}
                         for item in data.get('semantic_evidence', ())),
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
                   tuple(data.get('tokenization_evidence', ())),
                   tuple({**item, 'members': tuple(item['members']),
                          'support': tuple(item['support'])}
                         for item in data.get('candidate_graph_evidence', ())),
                   tuple({**item, 'assignments': tuple(item['assignments']),
                          'layouts': tuple(tuple(choice) for choice in item['layouts'])}
                         for item in data.get('candidate_graph_layouts', ())),
                   tuple({**item, 'layouts': tuple(item['layouts'])}
                         for item in data.get('candidate_graph_memberships', ())))


def fit_profile(reference_captures: list[Capture], config: BehaviorConfig | None = None,
                *, source_calibration: tuple[SourceRoleCalibration, ...] = (),
                validated_factors: tuple[SourceFactorValidation, ...] = (),
                source_model=None, target_vehicle: str | None = None) -> BehaviorProfile:
    """Fit only from independent, trusted normal capture inputs."""
    config = config or BehaviorConfig()
    if config.dbc_paths:
        raise ValueError('dbc_paths cannot be used in target fitting; build source_role_calibration from other vehicles')
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
    artifact = None
    if config.candidate_graph_artifact:
        artifact = GraphWeights.read(Path(config.candidate_graph_artifact))
        artifact.check_target(target_vehicle, reference_captures)
    memberships = (layout_memberships(registry.candidates, registry.tokenizations)
                   if artifact else {})
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
                layout_memberships=memberships if artifact else None,
                source_calibration=source_calibration,
                validated_factors=validated_factors)
                if config.physical_enabled else None)
    if artifact and physical is None:
        raise ValueError('candidate graph requires physical role discovery')
    graph = (rerank(physical, memberships,
        artifact, max_combinations=config.physical_max_combinations)
        if artifact else None)
    if graph:
        fields = selected_layout_fields(registry.candidates, registry.selected,
            memberships, graph.layout_choices[0]['layouts'], physical.protocol_fields)
    unresolved = unresolved_spans(fields, reference_captures)
    relations = tuple(rule for rule in (*dynamics.phase, *dynamics.change_coupling)
                      if rule.trigger in selected and rule.target in selected)[:config.max_relations]
    pairs = [(f'{rule.trigger}|{rule.kind}', f'{rule.target}|{rule.kind}',
              rule.lag_low, rule.lag_high) for rule in relations]
    motifs = fit_motifs(reference_captures, pairs, min_support=config.motif_min_support,
                        min_completion=config.motif_min_completion,
                        min_control_gain=config.motif_min_control_gain,
                        max_span=config.max_motif_span, max_motifs=config.max_motifs)
    field_edges: tuple[FieldEdge, ...] = ()
    rejected = list(dynamics.rejected)
    if config.field_graph_enabled:
        if find_spec('tigramite') is None:
            rejected.append({'kind': 'field_graph', 'reason': 'tigramite_not_installed'})
        else:
            field_edges = predictive_edges(reference_captures, fields)[:config.max_field_edges]
    families = classify_messages(reference_captures)
    layout_fields = tuple(fields) + tuple(item.candidate for item in
                                          (physical.hypotheses if physical else ()))
    layout_evidence = tuple({'id': f.arb_id, 'start': f.start, 'width': f.width,
                             'endian': f.endian, 'confidence': f.confidence,
                             'evidence': f.evidence} for f in layout_fields)
    role_evidence = (tuple({'id': f.arb_id, 'start': f.start, 'role': f.role,
                            'support': f.reference_support,
                            'status': 'verified_protocol' if f.role in ('counter', 'checksum')
                            else 'provisional'}
                           for f in fields) +
                     tuple({'id': item.candidate.arb_id, 'start': item.candidate.start,
                            'role': item.role, 'support': item.support,
                            'status': 'provisional', 'score': item.unary_score,
                            'calibration': item.calibration}
                           for item in (physical.hypotheses if physical else ())))
    semantic_evidence = tuple({'id': f.arb_id, 'start': f.start,
                               'labels': f.label_hypotheses} for f in fields
                              if any(name != 'unknown' for name, _ in f.label_hypotheses))
    tokenization_evidence = tuple({'id': result.trace.arb_id,
        'payload_length': result.trace.payload_length, 'bus': result.trace.bus,
        'method': result.method, 'parameters': result.parameters,
        'source_hashes': list(result.trace.source_hashes), 'support': list(result.trace.support),
        'spans': [{'physical_bits': list(span.physical_bits), 'byte_order': span.byte_order}
                  for span in result.spans], 'uncovered_bits': list(result.uncovered_bits),
        'abstention_reasons': list(result.abstention_reasons)}
        for result in registry.tokenizations) + registry.conflicts
    return BehaviorProfile(9 if graph else (8 if source_model is not None else 7), config,
                           tuple((capture.source, capture.sha256) for capture in reference_captures),
                           nodes, timing, payload, fields, relations, motifs,
                           dynamics.mode_count_threshold, tuple(rejected), field_edges, unresolved,
                           families, layout_evidence, role_evidence, semantic_evidence,
                           physical.hypotheses if physical else (),
                           graph.combinations if graph else (physical.combinations if physical else ()),
                           physical.rejected if physical else (), source_calibration,
                           physical.factor_evidence if physical else (), validated_factors,
                           tokenization_evidence,
                           tuple(__import__('dataclasses').asdict(item) for item in graph.evidence) if graph else (),
                           graph.layout_choices if graph else (),
                           tuple({'field': field_key(item),
                                  'layouts': memberships[field_key(item)]}
                                 for item in registry.candidates) if graph else ())
