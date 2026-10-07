"""Experimental integrated CAN decoding benchmark; never reads final routes."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from can_discovery.manifests.graph import load_graph_cohort, split_paths
from can_discovery.decoding.dbc import evaluate_discovery, load_dbc_fields
from can_discovery.decoding.signals import (
    FieldCandidate,
    _bit_positions,
    discover_fields,
)
from can_discovery.engineering.integrated import (
    dbc_formats,
    derivative_relation,
    discover_with_anchors,
    field_relations,
    match_frames,
    observed_families,
    relation_groups,
    wheel_mean_relation,
)
from can_discovery.engineering.signal_catalog import (
    behavior_priority_families,
    build_signal_catalog,
    family_profiles,
    source_role_prototypes,
)
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical.physics_factors import (
    FACTOR_ROLES,
    evaluate_factor,
)
from can_discovery.relationships.shared.field_analysis import field_key
from can_discovery.relationships.shared.signal_time import observations


def _captures(paths: list[str]) -> list:
    return [load_capture(Path(path)) for path in paths]


def run(entries: list[dict], target: dict, *, track: str,
        max_ids: int = 64) -> dict:
    if track not in ('source_only', 'dbc_assisted'):
        raise ValueError(track)
    fitting = _captures(split_paths(target, 'fitting'))
    validation = _captures(split_paths(target, 'validation'))
    sources = [(entry['vehicle'], _captures(split_paths(entry, 'fitting')),
                entry['dbc']) for entry in entries if entry is not target]
    source_prototypes = source_role_prototypes(sources)
    hashes = [capture.sha256 for capture in (*fitting, *validation)]
    if any(frame.attack for capture in (*fitting, *validation)
           for frame in capture.frames):
        raise ValueError('integrated discovery requires normal target recordings')
    if any(frame.attack for _, captures, _ in sources for capture in captures
           for frame in capture.frames):
        raise ValueError('integrated discovery requires normal source recordings')
    if len(set(hashes)) != len(hashes) or any(c.sha256 in hashes
            for _, captures, _ in sources for c in captures):
        raise ValueError('capture overlap across fitting, validation, or source vehicles')
    anchors, matches = match_frames(fitting, sources)
    target_formats = None
    if track == 'dbc_assisted':
        # The target DBC is an explicitly declared fitting input in this track.
        own = load_dbc_fields(Path(target['dbc']))
        target_formats = dbc_formats(str(target['dbc']))
        from dataclasses import replace
        observed = observed_families(fitting)
        own_active = tuple(replace(field, bus=bus,
                           evidence=('target_dbc_anchor',))
                           for field in own for bus, arb_id, length in observed
                           if (field.arb_id.upper(), field.payload_length) == (arb_id, length))
        existing = {(field.bus, field.arb_id.upper(), field.payload_length,
                     field.start, field.width, field.endian) for field in own_active}
        anchors = own_active + tuple(field for field in anchors if
            (field.bus, field.arb_id.upper(), field.payload_length,
             field.start, field.width, field.endian) not in existing)
    registry = discover_with_anchors(fitting, anchors, max_ids=max_ids)
    active = tuple(field for field in registry.selected if field.signed is not None)
    priority_families = {(item['bus'], item['arb_id'], item['payload_length'])
                         for item in matches}
    priority_families.update((field.bus, field.arb_id.upper(), field.payload_length)
                             for field in anchors)
    priority_families.update(behavior_priority_families(
        family_profiles(fitting), source_prototypes))
    edges = field_relations(fitting, validation, active,
                            priority_families=priority_families)
    # Physical expressions are evaluated for labeled anchors only. They remain
    # review evidence; the model does not promote labels from correlation.
    activity_cache = {}
    def activity(field: FieldCandidate) -> int:
        key = field_key(field)
        if key not in activity_cache:
            counts = [len(set(observations(capture, field)[1])) for capture in fitting]
            activity_cache[key] = min(counts) if counts else 0
        return activity_cache[key]
    names = {}
    for field in anchors:
        name = field.label_hypotheses[0][0].lower()
        if name not in names or activity(field) > activity(names[name]):
            names[name] = field
    physics = []
    selected_unknown = tuple(sorted((field for field in active
                             if field.label_hypotheses[0][0] == 'unknown'),
                             key=lambda field: (-(
                                 (field.bus, field.arb_id.upper(), field.payload_length)
                                 in priority_families), -activity(field), field_key(field))))
    for name, field in names.items():
        if 'speed' not in name or 'wheel' in name:
            continue
        for other_name, other in names.items():
            operation = ('derivative' if 'accel' in other_name else
                         'integral' if 'odometer' in other_name or 'distance' in other_name
                         else None)
            if operation is None:
                continue
            physics.append({'source': asdict(field), 'target': asdict(other),
                            **derivative_relation(fitting, validation, field, other,
                                                  operation=operation)})
        # AutoCAN-style searches are evidence for a review queue, not labels.
        for other in selected_unknown[:32]:
            if (field.bus, field.arb_id, field.payload_length) == \
                    (other.bus, other.arb_id, other.payload_length):
                continue
            for operation in ('derivative', 'integral'):
                evidence = derivative_relation(fitting, validation, field, other,
                                               operation=operation)
                if evidence['status'] == 'supported':
                    physics.append({'source': asdict(field), 'target': asdict(other),
                                    **evidence})
    wheel_fields = sorted((field for name, field in names.items()
                           if 'wheel' in name and 'speed' in name),
                          key=lambda field: -activity(field))
    speed_fields = sorted((field for name, field in names.items()
                           if 'speed' in name and 'wheel' not in name),
                          key=lambda field: -activity(field))
    if len(wheel_fields) >= 4 and speed_fields:
        evidence = wheel_mean_relation(fitting, validation, speed_fields[0],
                                       tuple(wheel_fields[:4]))
        physics.append({'source': [asdict(field) for field in wheel_fields[:4]],
                        'target': asdict(speed_fields[0]), **evidence})
    role_fields = {}
    for field in anchors:
        name = field.label_hypotheses[0][0].lower()
        for role, parts in {'speed': ('speed',), 'acceleration': ('accel',),
                            'rpm': ('rpm',), 'brake': ('brake',),
                            'steering': ('steer',), 'yaw': ('yaw',),
                            'gear': ('gear',), 'wheel_speed': ('wheel', 'speed')}.items():
            if all(part in name for part in parts):
                if role not in role_fields or activity(field) > activity(role_fields[role]):
                    role_fields[role] = field
    factors = []
    for name, roles in FACTOR_ROLES.items():
        if all(role in role_fields for role in roles):
            evidence = evaluate_factor([*fitting, *validation],
                tuple(role_fields[role] for role in roles), name)
            factors.append(asdict(evidence))
    review = []
    candidate_keys = {field_key(field): field for field in registry.candidates}
    for decision in matches:
        if decision['status'] != 'alternative':
            continue
        for option in decision['alternatives']:
            source_field = FieldCandidate(**option['field'])
            for bus in {item.bus for item in registry.candidates
                        if item.arb_id.upper() == decision['arb_id'] and
                        item.payload_length == decision['payload_length']}:
                from dataclasses import replace
                candidate = candidate_keys.get(field_key(replace(source_field, bus=bus)))
                if candidate is not None:
                    review.append({'field': asdict(candidate),
                        'suggested_label': source_field.label_hypotheses[0][0],
                        'evidence': 'same_id_unverified_reference',
                        'source_vehicle': option['vehicle'],
                        'status': 'review'})
    for edge in edges:
        left, right = active[edge['source']], active[edge['target']]
        for known, candidate in ((left, right), (right, left)):
            label = known.label_hypotheses[0][0]
            if label == 'unknown' or candidate.label_hypotheses[0][0] != 'unknown':
                continue
            review.append({'field': asdict(candidate), 'suggested_label': label,
                           'evidence': 'replicated_field_relation',
                           'relation': edge['relation'],
                           'support': edge['support'], 'status': 'review'})
    for item in physics:
        if item['status'] != 'supported' or item['target']['label_hypotheses'][0][0] != 'unknown':
            continue
        review.append({'field': item['target'],
                       'suggested_label': ('acceleration' if item['operation'] == 'derivative'
                                           else 'distance_or_odometer'),
                       'evidence': 'replicated_physical_expression',
                       'relation': item['operation'], 'status': 'review'})
    review = list({(json.dumps(item['field'], sort_keys=True),
                    item['suggested_label'], item['evidence'],
                    item.get('source_vehicle')): item for item in review}.values())
    return {'version': 1, 'method': 'canmatch_autocan_integrated',
            'track': track, 'vehicle': target['vehicle'],
            'max_ids': max_ids,
            'fitting_hashes': [c.sha256 for c in fitting],
            'validation_hashes': [c.sha256 for c in validation],
            'source_provenance': [{'vehicle': vehicle,
                'capture_hashes': [c.sha256 for c in captures],
                'dbc_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for vehicle, captures, path in sources],
            'source_role_prototypes': source_prototypes,
            'target_dbc_fitting_hash': (hashlib.sha256(Path(target['dbc']).read_bytes()).hexdigest()
                                        if track == 'dbc_assisted' else None),
            'target_anchor_formats': [{'arb_id': arb_id, 'signal': signal, **value}
                                      for (arb_id, signal), value in sorted(target_formats.items())]
                                     if target_formats is not None else None,
            'frame_matches': matches,
            'anchors': [asdict(field) for field in anchors],
            'candidates': [asdict(field) for field in registry.candidates],
            'selected': [asdict(field) for field in registry.selected],
            'conflicts': list(registry.conflicts),
            'field_relations': edges, 'relation_groups': relation_groups(active, edges),
            'physical_relations': physics,
            'physical_factors': factors,
            'review_queue': review,
            'label_status': 'review_only'}


def evaluate(result: dict, target: dict) -> dict:
    """Use target truth only after run has returned; assisted anchors are excluded."""
    truth = load_dbc_fields(Path(target['dbc']))
    if result['track'] == 'dbc_assisted':
        fitting = _captures(split_paths(target, 'fitting'))
        validation = _captures(split_paths(target, 'validation'))[0]
        # This DBC-free pool prevents assisted anchors from inflating recovery.
        blind = discover_fields(fitting, max_ids=result['max_ids'], max_layouts=8)
        proposals = [(FieldCandidate(**option['field']), decision['arb_id'],
                      decision['payload_length'])
                     for decision in result['frame_matches']
                     for option in decision['alternatives']]
        from can_discovery.decoding.field_values import extract
        rows = []
        for known in truth:
            if known.mux_start is not None:
                continue
            active = [frame.payload for frame in validation.frames
                      if frame.payload is not None and
                      frame.arb_id.upper() == known.arb_id.upper() and
                      len(frame.payload) == known.payload_length]
            if len(active) < 40 or len({extract(payload, known.start, known.width,
                    known.endian, known.signed) for payload in active}) < 2:
                continue
            candidates = [field for field in blind.candidates
                          if field.arb_id.upper() == known.arb_id.upper() and
                          field.payload_length == known.payload_length]
            def exact(field: FieldCandidate, known: FieldCandidate = known) -> bool:
                return (field.endian == known.endian and field.signed == known.signed and
                        _bit_positions(field.start, field.width, field.endian) ==
                        _bit_positions(known.start, known.width, known.endian))
            exact_rank = next((i for i, field in enumerate(candidates, 1)
                               if exact(field)), None)
            label = known.label_hypotheses[0][0]
            names = [field for field, arb_id, length in proposals
                     if (arb_id, length) == (known.arb_id.upper(), known.payload_length)]
            name_rank = next((i for i, field in enumerate(names, 1)
                              if exact(field) and field.label_hypotheses[0][0] == label), None)
            rows.append({'arb_id': known.arb_id, 'label': label,
                         'exact_layout_rank': exact_rank, 'source_name_rank': name_rank})
        return {'mode': 'all_target_labels_hidden_control', 'assessable': len(rows),
                'exact_layout_top_5': sum(row['exact_layout_rank'] is not None and
                                          row['exact_layout_rank'] <= 5 for row in rows),
                'source_name_top_5': sum(row['source_name_rank'] is not None and
                                         row['source_name_rank'] <= 5 for row in rows),
                'fields': rows}
    validation = _captures(split_paths(target, 'validation'))[0]
    fitting = _captures(split_paths(target, 'fitting'))
    baseline = discover_fields(fitting, max_ids=result['max_ids'], max_layouts=8)
    selected = tuple(FieldCandidate(**record) for record in result['selected'])
    candidates = tuple(FieldCandidate(**record) for record in result['candidates'])
    return {'selected_layout': evaluate_discovery(selected, truth, validation),
            'candidate_pool': evaluate_discovery(candidates, truth, validation),
            'tokenization_baseline': {
                'selected_layout': evaluate_discovery(baseline.selected, truth, validation),
                'candidate_pool': evaluate_discovery(baseline.candidates, truth, validation)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--track', choices=('source_only', 'dbc_assisted', 'both'),
                        default='both')
    parser.add_argument('--only-vehicle')
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--evaluate-dbc', action='store_true')
    args = parser.parse_args()
    entries = load_graph_cohort(args.benchmark_manifest, policy='signal_graph')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for target in entries:
        if args.only_vehicle and target['vehicle'].casefold() != args.only_vehicle.casefold():
            continue
        for track in (('source_only', 'dbc_assisted') if args.track == 'both'
                      else (args.track,)):
            result = run(entries, target, track=track, max_ids=args.max_ids)
            catalog = build_signal_catalog(result, _captures(split_paths(target, 'fitting')))
            if args.evaluate_dbc:
                result['evaluation'] = evaluate(result, target)
            path = args.output_dir / f"{target['vehicle'].lower()}-{track}-integrated-v1.json"
            path.write_text(json.dumps(result, indent=2) + '\n')
            catalog_path = args.output_dir / f"{target['vehicle'].lower()}-{track}-signal-catalog-v1.json"
            catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
            summary.append({'vehicle': target['vehicle'], 'track': track,
                            'artifact': str(path), 'anchors': len(result['anchors']),
                            'candidates': len(result['candidates']),
                            'relations': len(result['field_relations']),
                            'signal_catalog': str(catalog_path),
                            'catalog_roles': len(catalog['signals'])})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
