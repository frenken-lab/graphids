"""Mine provisional physical signal labels from normal CAN recordings.

This DBC-assisted workflow is separate from blind paper benchmarks. Scores
order a review queue; they are not probabilities or verified semantics.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate, _bit_positions, discover_fields
from ..preprocessing.capture import Capture
from ..relationships.physical.observational_edges import evaluate_graph_edge
from ..relationships.physical.role_types import RoleHypothesis
from ..relationships.physical.unary_roles import (
    _index_capture,
    _role_score,
    _sample_values,
)
from ..relationships.shared.signal_time import align, observations

ROLES = ('accelerator_pedal', 'brake_input', 'throttle_plate',
         'engine_rpm', 'vehicle_speed')
UNARY_ROLE = {'accelerator_pedal': 'throttle', 'brake_input': 'brake',
              'throttle_plate': 'throttle', 'engine_rpm': 'rpm',
              'vehicle_speed': 'speed'}
DBC_NAMES = {
    'accelerator_pedal': frozenset(('GAS_PEDAL', 'GAS_PEDAL_ALT', 'PEDAL_GAS',
        'ACCEL_PEDAL_POS', 'ACCELERATOR_PEDAL', 'THROTTLE_PEDAL')),
    'brake_input': frozenset(('BRAKE_AMOUNT', 'BRAKE_PEDAL', 'BRAKE_PEDAL_POS',
        'BRAKE_PRESSURE', 'BRAKE_PRESSURE1', 'BRAKE_PRESSURE2', 'BRAKE_1', 'BRAKE_2')),
    'throttle_plate': frozenset(('THROTTLE_PLATE_POSITION', 'THROTTLE_POSITION')),
    'engine_rpm': frozenset(('ENGINE_RPM', 'RPM', 'CLUSTER_ENGINE_RPM')),
    'vehicle_speed': frozenset(('SPEED', 'CAR_SPEED', 'CF_CLU_VEHICLESPEED')),
}
RELATIONS = {
    'accelerator_pedal': (('accelerator_to_rpm', 'source', 'engine_rpm', 'throttle_rpm'),
                          ('accelerator_to_speed', 'source', 'vehicle_speed', 'throttle_speed')),
    'brake_input': (('brake_to_speed', 'source', 'vehicle_speed', 'brake_speed'),),
    'throttle_plate': (('plate_to_rpm', 'source', 'engine_rpm', 'throttle_rpm'),
                       ('plate_to_speed', 'source', 'vehicle_speed', 'throttle_speed'),
                       ('pedal_to_plate', 'target', 'accelerator_pedal', 'throttle_rpm')),
    'engine_rpm': (('accelerator_to_rpm', 'target', 'accelerator_pedal', 'throttle_rpm'),),
    'vehicle_speed': (('accelerator_to_speed', 'target', 'accelerator_pedal', 'throttle_speed'),
                      ('brake_to_speed', 'target', 'brake_input', 'brake_speed')),
}


def field_key(field: FieldCandidate) -> tuple:
    return (field.bus, field.arb_id.upper(), field.payload_length, field.start,
            field.width, field.endian, field.signed, field.mux_start, field.mux_value)


def _active(field: FieldCandidate, captures: list[Capture]) -> bool:
    for capture in captures:
        _, values = observations(capture, field)
        if len(values) < 30 or len(np.unique(values)) < 2:
            return False
    return True


def anchors_by_role(dbc: Path, captures: list[Capture], *, hidden: str | None = None
                    ) -> dict[str, tuple[FieldCandidate, ...]]:
    """Only observed, changing DBC fields can anchor a physical role."""
    fields = load_dbc_fields(dbc)
    return {role: tuple(field for field in fields if role != hidden and
                        field.label_hypotheses[0][0].upper() in DBC_NAMES[role]
                        and _active(field, captures)) for role in ROLES}


def _overlaps(left: FieldCandidate, right: FieldCandidate) -> bool:
    return ((left.bus, left.arb_id.upper(), left.payload_length) ==
            (right.bus, right.arb_id.upper(), right.payload_length) and
            bool(_bit_positions(left.start, left.width, left.endian) &
                 _bit_positions(right.start, right.width, right.endian)))


def _signature(field: FieldCandidate, captures: list[Capture]) -> tuple:
    """Group equivalent traces without assuming exact bit boundaries."""
    result = []
    for capture in captures:
        _, values = observations(capture, field)
        if len(values) < 30:
            return ()
        sample = values[np.linspace(0, len(values)-1, min(128, len(values)), dtype=int)]
        spread = np.ptp(sample)
        normalized = (sample-sample[0])/spread if spread else np.zeros_like(sample)
        result.append(tuple(np.round(normalized, 3)))
    return tuple(result)


def _relations(role: str, field: FieldCandidate,
               anchors: dict[str, tuple[FieldCandidate, ...]],
               fitting: list[Capture], validation: list[Capture],
               frame_indices: list[dict], pair_cache: dict) -> list[dict]:
    out = []
    for label, position, other_role, factor in RELATIONS[role]:
        for anchor in anchors[other_role][:3]:
            if field_key(field) == field_key(anchor) or _overlaps(field, anchor):
                continue
            source, target = ((field, anchor) if position == 'source' else (anchor, field))
            families = frozenset(((source.arb_id, source.payload_length),
                                  (target.arb_id, target.payload_length)))
            if families not in pair_cache:
                pair_cache[families] = [replace(capture, frames=tuple(sorted(
                    (frame for family in families for frame in index.get(family, ())),
                    key=lambda frame: (frame.timestamp, frame.ordinal))))
                    for capture, index in zip([*fitting, *validation], frame_indices, strict=True)]
            local = pair_cache[families]
            fitted = evaluate_graph_edge(local[:len(fitting)], source, target, factor)
            assessed = (evaluate_graph_edge(local[len(fitting):], source, target, factor,
                        forced_lag_seconds=fitted.lag_seconds)
                        if fitted.status == 'supported' else None)
            out.append({'relation': label, 'anchor': field_key(anchor),
                        'fit': asdict(fitted),
                        'validation': asdict(assessed) if assessed else None,
                        'replicated': bool(assessed and assessed.status == 'supported')})
    return out


def _truth_metrics(rows: list[dict], truths: tuple[FieldCandidate, ...],
                   fitting: list[Capture], validation: list[Capture]) -> dict:
    truths = tuple(field for field in truths if _active(field, validation))
    if not truths:
        return {'assessable': False, 'exact_rank': None, 'value_rank': None,
                'top_1': None, 'top_5': None}
    exact = {field_key(field) for field in truths}
    exact_rank = next((row['rank'] for row in rows if tuple(row['field_key']) in exact), None)
    value_rank = next((row['rank'] for row in rows
                       if any(_value_equivalent(FieldCandidate(**row['field']), truth,
                                                [*fitting, *validation])
                              for truth in truths)), None)
    return {'assessable': True, 'exact_rank': exact_rank, 'value_rank': value_rank,
            'top_1': value_rank == 1, 'top_5': value_rank is not None and value_rank <= 5}


def _value_equivalent(candidate: FieldCandidate, truth: FieldCandidate,
                      captures: list[Capture]) -> bool:
    """Fit an affine raw-value map on one route and check another route."""
    paired = []
    for capture in captures:
        try:
            values, valid, _, _ = align(capture, (candidate, truth), step=.05,
                                        max_age=.2, max_bins=8000)
        except ValueError:
            return False
        mask = valid.all(axis=1)
        if mask.sum() < 100:
            return False
        x, y = values[mask].T
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return False
        paired.append((x, y))
    x, y = paired[0]
    design = np.column_stack((x, np.ones_like(x)))
    gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    if abs(gain) < 1e-10:
        return False
    for x, y in paired:
        residual = np.mean((gain*x+offset-y)**2)
        baseline = np.var(y)
        if baseline < 1e-8 or 1-residual/baseline < .99:
            return False
    return True


def _joint_matches(catalog: dict, *, width: int = 64) -> list[dict]:
    """Retain several compatible partial assignments, including null choices."""
    beam = [({}, 0., [])]
    for role in ROLES:
        if catalog[role]['anchors']:
            continue
        # Unknown review candidates remain visible in the catalog, but do not
        # become a forced physical-role assignment in the joint view.
        options = [None, *(row for row in catalog[role]['candidates']
                           if row['status'] == 'provisional')]
        expanded = []
        for assignments, score, used in beam:
            for row in options:
                if row is not None:
                    field = FieldCandidate(**row['field'])
                    if any(_overlaps(field, other) for other in used):
                        continue
                else:
                    field = None
                expanded.append(({**assignments, role: row['rank'] if row else None},
                                 score + (row['score'] if row else 0.),
                                 [*used, field] if field else used))
        beam = sorted(expanded, key=lambda item: (-item[1], repr(item[0])))[:width]
    return [{'assignments': assignments, 'score': score,
             'status': 'partial_hypothesis'} for assignments, score, _ in beam[:8]]


def _screen_candidates(candidates: tuple[FieldCandidate, ...],
                       fitting: list[Capture]) -> dict[str, list[RoleHypothesis]]:
    """Reuse the unary role evidence without the costly factor-combination fit."""
    indices = [_index_capture(capture) for capture in fitting]
    output: dict[str, list[RoleHypothesis]] = {role: [] for role in ROLES}
    for field in candidates:
        if field.role != 'unknown' or field.signed is None:
            continue
        history = [_sample_values(capture, field, cap=256, index=index)
                   for capture, index in zip(fitting, indices, strict=True)]
        if min(len(values) for _, values in history) < 30:
            continue
        for role in ROLES:
            score, evidence = _role_score(UNARY_ROLE[role], history, field)
            if np.isfinite(score) and score >= .30:
                output[role].append(RoleHypothesis(field, role, score,
                    tuple(len(values) for _, values in history), evidence))
    for role in ROLES:
        output[role].sort(key=lambda item: (-item.unary_score,
                                           repr(field_key(item.candidate))))
    return output


def _frame_index(capture: Capture) -> dict[tuple[str, int], list]:
    result: dict[tuple[str, int], list] = {}
    for frame in capture.frames:
        if frame.payload is not None:
            result.setdefault((frame.arb_id, len(frame.payload)), []).append(frame)
    return result


def recovery_gate(diagnostics: dict[str, dict]) -> dict:
    """Require more than one independently recovered known role per vehicle."""
    assessable = sum(item['assessable'] for item in diagnostics.values())
    recovered_top_1 = sum(item['top_1'] is True for item in diagnostics.values())
    return {'ready': assessable >= 2 and recovered_top_1 >= 2,
            'assessable_known_roles': assessable,
            'top_1_recovered': recovered_top_1, 'required_top_1': 2}


def mine_vehicle(vehicle: str, fitting: list[Capture], validation: list[Capture],
                 dbc: Path, *, max_ids: int = 64, max_layouts: int = 8,
                 top_k: int = 5) -> dict:
    """Build a review catalog and leave-one-role-out diagnostics."""
    if len(fitting) != 2 or not validation:
        raise ValueError('two fitting and at least one validation route are required')
    captures = [*fitting, *validation]
    if len({capture.sha256 for capture in captures}) != len(captures):
        raise ValueError('fitting and validation routes must be distinct')
    if any(any(frame.attack for frame in capture.frames) for capture in captures):
        raise ValueError('signal catalog requires normal captures')
    registry = discover_fields(fitting, max_ids=max_ids, max_layouts=max_layouts)
    hypotheses = _screen_candidates(registry.candidates, fitting)
    anchors = anchors_by_role(dbc, fitting)
    dbc_fields = load_dbc_fields(dbc)
    frame_indices = [_frame_index(capture) for capture in captures]
    pair_cache: dict = {}
    catalog, diagnostics = {}, {}
    for role in ROLES:
        other_anchors = {name: fields if name != role else ()
                         for name, fields in anchors.items()}
        ranked = hypotheses[role]
        rows, seen = [], set()
        for hypothesis in ranked:
            field = hypothesis.candidate
            if any(_overlaps(field, anchor) for group in other_anchors.values()
                   for anchor in group):
                continue
            signature = _signature(field, fitting)
            if not signature or signature in seen:
                continue
            seen.add(signature)
            activity = [len(np.unique(observations(capture, field)[1]))
                        for capture in captures]
            relations = _relations(role, field, other_anchors, fitting, validation,
                                   frame_indices, pair_cache)
            fit_supported = sum(item['fit']['status'] == 'supported' for item in relations)
            replicated = sum(item['replicated'] for item in relations)
            conflicting_dbc = sorted({known.label_hypotheses[0][0]
                for known in dbc_fields if _overlaps(field, known) and
                known.label_hypotheses[0][0].upper() not in DBC_NAMES[role]})
            # A speed trend is not enough to distinguish pedal from plate.
            # Plate needs an independently observed accelerator anchor as well
            # as separate RPM and speed support before it can be provisional.
            minimum_relations = 2 if role in ('accelerator_pedal', 'throttle_plate') else 1
            distinct_plate = role != 'throttle_plate' or bool(anchors['accelerator_pedal'])
            plate_order = (role != 'throttle_plate' or
                           any(item['relation'] == 'pedal_to_plate' and item['replicated']
                               for item in relations))
            rows.append({'field_key': field_key(field), 'field': asdict(field),
                         'unary_score': hypothesis.unary_score,
                         'score': hypothesis.unary_score + .15*fit_supported -
                                  (.5 if conflicting_dbc else 0.),
                         'distinct_values_by_route': activity,
                         'relations': relations,
                         'conflicting_dbc_labels': conflicting_dbc,
                         'status': ('provisional' if min(activity) >= 2 and
                                    replicated >= minimum_relations and distinct_plate and
                                    plate_order and
                                    not conflicting_dbc
                                    else 'unknown')})
            if len(rows) >= 16:
                break
        unary_rows = sorted(rows, key=lambda row: (-row['unary_score'],
                                                    repr(row['field_key'])))
        unary_rows = [{**row, 'rank': index} for index, row in enumerate(unary_rows, 1)]
        baseline = _truth_metrics(unary_rows, anchors[role], fitting, validation)
        rows.sort(key=lambda row: (-row['score'], repr(row['field_key'])))
        for index, row in enumerate(rows, 1):
            row['rank'] = index
        diagnostics[role] = _truth_metrics(rows, anchors[role], fitting, validation)
        diagnostics[role]['unary_baseline'] = baseline
        catalog[role] = {
            'anchors': [asdict(field) for field in anchors[role]],
            'status': 'verified_anchor' if anchors[role] else
                      ('provisional' if any(row['status'] == 'provisional' for row in rows[:top_k])
                       else 'unknown' if rows else 'unobserved'),
            'candidates': rows[:top_k], 'review_queue': rows[top_k:],
        }
    gate = recovery_gate(diagnostics)
    # One isolated success is not enough to trust semantic assignments on a
    # vehicle with no independently named truth for the missing role.
    if not gate['ready']:
        for role in ROLES:
            for row in (*catalog[role]['candidates'], *catalog[role]['review_queue']):
                if row['status'] == 'provisional':
                    row['status'] = 'unknown'
                    row['hold_reason'] = 'known_label_recovery_gate_failed'
            if catalog[role]['status'] == 'provisional':
                catalog[role]['status'] = 'unknown'
    return {'version': 1, 'method': 'dbc_assisted_engineering', 'vehicle': vehicle,
            'dbc': str(dbc), 'fitting_hashes': [capture.sha256 for capture in fitting],
            'validation_hashes': [capture.sha256 for capture in validation],
            'candidate_count': len(registry.candidates), 'catalog': catalog,
            'leave_one_role_out': diagnostics,
            'recovery_gate': gate,
            'joint_matches': _joint_matches(catalog),
            'limitations': ['raw_values_only', 'physical_names_provisional',
                            'complete_pica_can_graph_not_required']}
