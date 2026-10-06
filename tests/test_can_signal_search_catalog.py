from __future__ import annotations

from dataclasses import asdict

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.engineering.signal_catalog import (
    behavior_priority_families, build_signal_catalog, family_profiles,
)
from can_discovery.preprocessing.capture import Capture, Frame


def capture(name: str, ids: tuple[str, ...]) -> Capture:
    frames = tuple(Frame(i*.02+j*.001, i*len(ids)+j, arb_id,
                         bytes([i % 256, 0]))
                   for i in range(120) for j, arb_id in enumerate(ids))
    return Capture(name, name, frames, 0., 2.4, 0, 0, {}, {})


def field(arb_id: str) -> FieldCandidate:
    return FieldCandidate(arb_id, 0, 8, 'little', False, 'unknown', 0., 1., (),
                          (('unknown', 1.),), (), 2)


def test_family_profile_retains_period_and_bit_rates_per_route():
    profiles = family_profiles([capture('a', ('100',)), capture('b', ('100',))])
    family = profiles['unknown|100|2']
    assert family['routes'][0]['frames'] == 120
    assert abs(family['routes'][0]['median_period_seconds'] - .02) < 1e-8
    assert len(family['routes'][0]['bit_flip_rates']) == 16
    assert family['routes'][0]['bit_flip_rates'][8:] == [0.] * 8


def test_id_first_catalog_preserves_role_without_known_bit_span():
    source = field('100')
    source_record = asdict(source)
    source_record['label_hypotheses'] = (('SPEED', 1.),)
    result = {'vehicle': 'target', 'track': 'source_only',
              'fitting_hashes': ['a', 'b'], 'source_provenance': [],
              'target_dbc_fitting_hash': None,
              'frame_matches': [{'bus': None, 'arb_id': '100', 'payload_length': 2,
                                 'status': 'alternative',
                                 'alternatives': [{'field': source_record,
                                                   'vehicle': 'source'}]}],
              'anchors': [], 'candidates': [asdict(field('200'))],
              'selected': [], 'review_queue': [], 'field_relations': [],
              'physical_relations': []}
    catalog = build_signal_catalog(result, [capture('a', ('100', '200')),
                                            capture('b', ('100', '200'))])
    speed = catalog['signals']['speed']
    assert speed['status'] == 'id_only_hypothesis'
    assert speed['family_hypotheses'][0]['family'] == 'unknown|100|2'
    assert speed['field_hypotheses'] == []


def test_source_behavior_profile_suggests_id_without_claiming_bits():
    a = capture('a', ('100', '200'))
    b = capture('b', ('100', '200'))
    profiles = family_profiles([a, b])
    prototype = {'vehicle': 'source', 'role': 'speed', 'family': 'other|999|2',
                 'features': {'period_seconds': .02, 'payload_length': 2,
                              'active_bit_fraction': .25,
                              'mean_bit_flip_rate': .08}}
    priorities = behavior_priority_families(profiles, [prototype])
    assert (None, '100', 2) in priorities
    result = {'vehicle': 'target', 'track': 'source_only',
              'fitting_hashes': ['a', 'b'], 'source_provenance': [],
              'target_dbc_fitting_hash': None, 'source_role_prototypes': [prototype],
              'frame_matches': [], 'anchors': [], 'candidates': [],
              'selected': [], 'review_queue': [], 'field_relations': [],
              'physical_relations': []}
    catalog = build_signal_catalog(result, [a, b])
    assert catalog['signals']['speed']['status'] == 'id_only_hypothesis'
    assert catalog['signals']['speed']['field_hypotheses'] == []
    assert catalog['signals']['speed']['expected_relations'][0]['operation'] == 'derivative'
