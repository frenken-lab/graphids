"""Versioned, uncertainty-preserving physical-signal search catalog."""
from __future__ import annotations

import re
import math
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from ..decoding.signals import FieldCandidate
from ..decoding.dbc import load_dbc_fields
from ..preprocessing.capture import Capture
from ..relationships.physical.source_calibration import _dbc_role
from ..relationships.shared.field_analysis import field_key


EXPECTED_RELATIONS = {
    'speed': ({'operation': 'derivative', 'target_role': 'acceleration'},
              {'operation': 'four_wheel_mean', 'source_role': 'wheel_speed_*'},
              {'operation': 'integral', 'target_role': 'distance_or_odometer'}),
    'acceleration': ({'operation': 'derivative', 'source_role': 'speed'},),
    'yaw': ({'operation': 'kinematic_yaw', 'source_roles': ['speed', 'steering']},),
    'lateral_acceleration': ({'operation': 'kinematic_lateral',
                             'source_roles': ['speed', 'yaw']},),
    'rpm': ({'operation': 'speed_rpm_gear', 'source_roles': ['speed', 'gear']},),
}


def semantic_key(label: str) -> str:
    """Keep wheel corners distinct while sharing conservative role aliases."""
    normalized = re.sub(r'[^A-Z0-9]+', '_', label.upper()).strip('_')
    if 'WHEEL' in normalized and 'SPEED' in normalized:
        for corner in ('FL', 'FR', 'RL', 'RR'):
            if normalized.endswith('_' + corner):
                return 'wheel_speed_' + corner.lower()
        return 'wheel_speed'
    return _dbc_role(label) or 'label:' + normalized


def _family_key(bus: str | None, arb_id: str, length: int) -> str:
    return f'{bus or "unknown"}|{arb_id.upper()}|{length}'


def family_profiles(captures: list[Capture]) -> dict[str, dict]:
    """Summarize ID/DLC behavior per route without merging route evidence."""
    grouped = defaultdict(lambda: [[] for _ in captures])
    for route, capture in enumerate(captures):
        for frame in capture.frames:
            if frame.payload is None or not 1 <= len(frame.payload) <= 8:
                continue
            key = _family_key(frame.bus or capture.known_bus, frame.arb_id,
                              len(frame.payload))
            grouped[key][route].append(frame)
    output = {}
    for key, routes in sorted(grouped.items()):
        per_route = []
        for frames in routes:
            if not frames:
                per_route.append({'frames': 0, 'median_period_seconds': None,
                                  'period_mad_seconds': None,
                                  'bit_flip_rates': None, 'bit_one_rates': None})
                continue
            stamps = np.asarray([frame.timestamp for frame in frames])
            gaps = np.diff(stamps)
            gaps = gaps[gaps > 0]
            bits = np.unpackbits(np.stack([np.frombuffer(frame.payload,
                    dtype=np.uint8) for frame in frames]), axis=1)
            period = float(np.median(gaps)) if len(gaps) else None
            per_route.append({'frames': len(frames),
                'median_period_seconds': period,
                'period_mad_seconds': (float(np.median(np.abs(gaps-period)))
                                       if period is not None else None),
                'bit_flip_rates': (np.abs(np.diff(bits.astype(float), axis=0)).mean(axis=0)
                                   if len(bits) > 1 else np.zeros(bits.shape[1])).round(6).tolist(),
                'bit_one_rates': bits.mean(axis=0).round(6).tolist()})
        bus, arb_id, length = key.split('|')
        output[key] = {'bus': None if bus == 'unknown' else bus,
                       'arb_id': arb_id, 'payload_length': int(length),
                       'routes': per_route}
    return output


def _behavior_features(profile: dict) -> dict | None:
    routes = profile['routes']
    if not routes or any(item['frames'] < 40 or item['median_period_seconds'] is None
                         for item in routes):
        return None
    periods = [item['median_period_seconds'] for item in routes]
    if max(periods) / max(min(periods), 1e-9) > 2.5:
        return None
    return {'period_seconds': float(np.median(periods)),
            'payload_length': profile['payload_length'],
            'active_bit_fraction': float(np.mean([
                np.mean(np.asarray(item['bit_flip_rates']) > .01) for item in routes])),
            'mean_bit_flip_rate': float(np.mean([
                np.mean(item['bit_flip_rates']) for item in routes]))}


def source_role_prototypes(sources: list[tuple[str, list[Capture], str]]) -> list[dict]:
    """Source-DBC role behavior, with each source family represented once per role."""
    output = []
    for vehicle, captures, dbc in sources:
        profiles = family_profiles(captures)
        seen = set()
        for field in load_dbc_fields(dbc):
            role = semantic_key(field.label_hypotheses[0][0])
            if role.startswith('label:'):
                continue
            for key, profile in profiles.items():
                if (profile['arb_id'], profile['payload_length']) != \
                        (field.arb_id.upper(), field.payload_length):
                    continue
                identity = (vehicle, role, key)
                if identity in seen:
                    continue
                seen.add(identity)
                features = _behavior_features(profile)
                if features is not None:
                    output.append({'vehicle': vehicle, 'role': role,
                                   'family': key, 'features': features})
    return output


def behavior_prior(target: dict, reference: dict) -> float:
    """Weak search prior; never sufficient to assign a physical name."""
    period_ratio = abs(math.log2(max(target['period_seconds'], 1e-9) /
                                 max(reference['period_seconds'], 1e-9)))
    length_difference = abs(target['payload_length'] - reference['payload_length'])
    activity_difference = abs(target['active_bit_fraction'] -
                              reference['active_bit_fraction'])
    flip_difference = abs(target['mean_bit_flip_rate'] -
                          reference['mean_bit_flip_rate'])
    return float(1 / (1 + period_ratio + .25*length_difference +
                       activity_difference + 2*flip_difference))


def behavior_priority_families(target_profiles: dict[str, dict],
                               prototypes: list[dict], *, top_k: int = 5
                               ) -> set[tuple[str | None, str, int]]:
    by_role = defaultdict(list)
    for item in prototypes:
        by_role[item['role']].append(item)
    selected = set()
    for references in by_role.values():
        ranked = []
        for profile in target_profiles.values():
            features = _behavior_features(profile)
            if features is None:
                continue
            score = max(behavior_prior(features, item['features']) for item in references)
            ranked.append((score, profile))
        for score, profile in sorted(ranked, key=lambda item:
                                     (-item[0], item[1]['arb_id']))[:top_k]:
            if score >= .45:
                selected.add((profile['bus'], profile['arb_id'], profile['payload_length']))
    return selected


def build_signal_catalog(result: dict, fitting: list[Capture]) -> dict:
    """Create role-first and ID-first views from one frozen integrated result."""
    families = family_profiles(fitting)
    roles = defaultdict(lambda: {'labels': set(), 'families': {}, 'fields': {},
                                 'relationships': []})
    by_field = {}
    def add_family(role: str, label: str, family: str, evidence: str,
                   priority: int, source: str | None = None) -> None:
        row = roles[role]
        row['labels'].add(label)
        current = row['families'].setdefault(family, {'family': family,
            'priority': 0, 'evidence': []})
        current['priority'] = max(current['priority'], priority)
        item = {'kind': evidence}
        if source:
            item['source_vehicle'] = source
        if item not in current['evidence']:
            current['evidence'].append(item)

    for match in result['frame_matches']:
        family = _family_key(match['bus'], match['arb_id'], match['payload_length'])
        for option in match['alternatives']:
            label = option['field']['label_hypotheses'][0][0]
            add_family(semantic_key(label), label, family,
                'accepted_frame_match' if match['status'] == 'anchor' else
                'same_id_unverified', 3 if match['status'] == 'anchor' else 1,
                option['vehicle'])
    for record in result['anchors']:
        field = FieldCandidate(**record)
        label = field.label_hypotheses[0][0]
        role = semantic_key(label)
        family = _family_key(field.bus, field.arb_id, field.payload_length)
        add_family(role, label, family, 'dbc_anchor' if
                   'target_dbc_anchor' in field.evidence else 'accepted_frame_match',
                   4 if 'target_dbc_anchor' in field.evidence else 3)
        by_field[field_key(field)] = role
    for proposal in result['review_queue']:
        field = FieldCandidate(**proposal['field'])
        label = proposal['suggested_label']
        role = semantic_key(label)
        family = _family_key(field.bus, field.arb_id, field.payload_length)
        add_family(role, label, family, proposal['evidence'], 2)
        by_field.setdefault(field_key(field), role)
    # Timing and bit activity can suggest an ID even when no ID is reused.
    # Keep these explicitly weaker than a same-ID match.
    behavior_by_role = defaultdict(list)
    for prototype in result.get('source_role_prototypes', []):
        behavior_by_role[prototype['role']].append(prototype)
    for role, prototypes in behavior_by_role.items():
        ranked = []
        for key, profile in families.items():
            features = _behavior_features(profile)
            if features is None:
                continue
            score = max(behavior_prior(features, item['features']) for item in prototypes)
            ranked.append((score, key))
        for score, key in sorted(ranked, key=lambda item: (-item[0], item[1]))[:5]:
            if score < .45:
                continue
            add_family(role, role, key, 'source_behavior_prior', 0)
            roles[role]['families'][key]['behavior_similarity'] = round(score, 4)
    for record in result['candidates']:
        field = FieldCandidate(**record)
        family = _family_key(field.bus, field.arb_id, field.payload_length)
        for role, row in roles.items():
            if family not in row['families']:
                continue
            key = field_key(field)
            if key not in row['fields']:
                row['fields'][key] = {'field': record, 'field_key': key,
                    'priority': row['families'][family]['priority'],
                    'evidence': ['id_family_candidate']}
            if key in by_field and by_field[key] == role:
                row['fields'][key]['priority'] += 2
                row['fields'][key]['evidence'].append('field_level_evidence')
    selected = [FieldCandidate(**record) for record in result['selected']
                if record['signed'] is not None]
    for edge in result['field_relations']:
        left, right = selected[edge['source']], selected[edge['target']]
        left_role, right_role = by_field.get(field_key(left)), by_field.get(field_key(right))
        for role in {left_role, right_role} - {None}:
            roles[role]['relationships'].append({'kind': edge['relation'],
                'source_field': field_key(left), 'target_field': field_key(right),
                'status': edge['status'], 'support': edge['support'],
                'lag_seconds': edge['lag_seconds']})
    for relation in result['physical_relations']:
        target = relation['target']
        if not isinstance(target, dict):
            continue
        target_field = FieldCandidate(**target)
        role = by_field.get(field_key(target_field))
        if role:
            roles[role]['relationships'].append({'kind': relation.get('operation'),
                'target_field': field_key(target_field),
                'status': relation['status'],
                'routes': relation.get('routes', [])})
    profiles = {}
    for role, row in sorted(roles.items()):
        family_rows = sorted(row['families'].values(),
                             key=lambda item: (-item['priority'],
                                               -item.get('behavior_similarity', 0.),
                                               item['family']))
        field_rows = sorted(row['fields'].values(),
                            key=lambda item: (-item['priority'],
                                -row['families'][_family_key(
                                    item['field']['bus'], item['field']['arb_id'],
                                    item['field']['payload_length'])].get(
                                        'behavior_similarity', 0.),
                                item['field_key']))
        for rank, item in enumerate(family_rows, 1):
            item['rank'] = rank
        for rank, item in enumerate(field_rows, 1):
            item['rank'] = rank
        verified = any('dbc_anchor' in [ev['kind'] for ev in item['evidence']]
                       for item in family_rows)
        profiles[role] = {'labels': sorted(row['labels']),
            'status': 'dbc_assisted_anchor' if verified else
                      'hypothesis_with_field' if field_rows else 'id_only_hypothesis',
            'expected_relations': EXPECTED_RELATIONS.get(role, ()),
            'family_hypotheses': family_rows,
            'field_hypotheses': field_rows,
            'relationships': row['relationships']}
    return {'version': 1, 'kind': 'physical_signal_search_catalog',
            'vehicle': result['vehicle'], 'track': result['track'],
            'fitting_hashes': result['fitting_hashes'],
            'source_provenance': result['source_provenance'],
            'target_dbc_fitting_hash': result['target_dbc_fitting_hash'],
            'source_role_prototypes': result.get('source_role_prototypes', []),
            'families': families, 'signals': profiles,
            'selection_rule': 'ID/DLC evidence first; field and replicated relation evidence refine rank',
            'name_claim': 'hypotheses_except_declared_dbc_anchors'}
