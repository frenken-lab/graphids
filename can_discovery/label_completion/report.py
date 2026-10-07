"""Verify and summarize the frozen v6 development cases."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from .baseline import ROLES
from .full_search import verify_staged_manifest
from .manifest import (
    artifact_hash,
    file_hash,
    verify_manifest,
)
from .progressive_v6 import SCHEMA


def _best(case: dict, name: str) -> int | None:
    values = [row[name] for row in case['reference_transitions_evaluation_only']]
    return min(values) if values else None


def _near_layout_ranks(case: dict, case_dir: Path
                       ) -> tuple[int | None, int | None, int | None]:
    """Find a near bit span in the reference value class; evaluation only."""
    if not case['assessable']:
        return None, None, None
    stem = f"{case['vehicle'].lower()}-{case['role']}"
    baseline_path = case_dir / f'{stem}-baseline.json'
    v5_path = case_dir / f'{stem}-prepared.json'
    truths = json.loads(baseline_path.read_text())['dbc_annotations']['reference_layouts']
    v5 = json.loads(v5_path.read_text())
    if file_hash(v5_path) != case['v5_case_sha256']:
        raise ValueError(f'v5 summary changed: {v5_path}')
    if file_hash(Path(v5['candidate_pool_path'])) != case['candidate_pool_sha256']:
        raise ValueError(f'v5 candidate pool changed: {v5_path}')
    truth_classes = {(row['bus'], row['arb_id'], row['payload_length'],
                      row['class_rank'])
                     for row in case['reference_transitions_evaluation_only']}
    candidate_ranks = {}
    family_id_ranks = {(row['bus'], row['arb_id'], row['payload_length']):
                       row['joint_family_rank'] for row in case['families']}
    offsets = {}
    with gzip.open(case['class_artifact_path'], 'rt') as stream:
        for line in stream:
            value_class = json.loads(line)
            family = tuple(value_class['family'])
            class_rank = value_class['class_rank']
            offset = offsets.get(family, 0)
            offsets[family] = offset + len(value_class['member_layout_ranks'])
            if (*family, class_rank) not in truth_classes:
                continue
            for within, old_rank in enumerate(value_class['member_layout_ranks'], 1):
                candidate_ranks[(*family, old_rank)] = (class_rank, offset + within)
    best = None
    joint_min = None
    with gzip.open(v5['candidate_pool_path'], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            family = (row['bus'], row['arb_id'], row['payload_length'])
            ranks = candidate_ranks.get((*family, row['layout_rank']))
            if ranks is None:
                continue
            for truth in truths:
                if family != (truth['bus'], truth['arb_id'], truth['payload_length']):
                    continue
                if row['endian'] != truth['endian'] or row['signed'] != truth['signed']:
                    continue
                left = max(row['start'], truth['start'])
                right = min(row['start'] + row['width'],
                            truth['start'] + truth['width'])
                overlap = max(0, right - left)
                union = row['width'] + truth['width'] - overlap
                if overlap / union >= .8:
                    candidate = (ranks[1], ranks[0], row['layout_rank'])
                    best = min(best, candidate) if best is not None else candidate
                    joint_cost = max(family_id_ranks[family], ranks[1])
                    joint_min = min(joint_min, joint_cost) if joint_min is not None else joint_cost
                    break
    return (best[0], best[1], joint_min) if best else (None, None, None)


def aggregate(manifest_path: Path, source_manifest_path: Path, case_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    source = json.loads(source_manifest_path.read_text())
    verify_staged_manifest(source, manifest)
    verify_manifest(manifest)
    cases = []
    for entry in manifest['vehicles']:
        for role in ROLES:
            path = case_dir / f"{entry['vehicle'].lower()}-{role}-case.json"
            case = json.loads(path.read_text())
            if (case['schema'] != SCHEMA
                    or case['vehicle'] != entry['vehicle'] or case['role'] != role
                    or case['manifest_hash'] != artifact_hash(source)
                    or case['staged_manifest_hash'] != artifact_hash(manifest)):
                raise ValueError(f'case provenance mismatch: {path}')
            if file_hash(Path(case['class_artifact_path'])) != case['class_artifact_sha256']:
                raise ValueError(f'class artifact changed: {path}')
            if case['final_routes_opened'] or not case['review_ledger']['abstained']:
                raise ValueError(f'final route or decision violation: {path}')
            cases.append(case)
    assessed = [case for case in cases if case['assessable']]

    def count(name: str, limit: int) -> int:
        return sum((rank := _best(case, name)) is not None and rank <= limit
                   for case in assessed)

    def joint(limit: int, layout_name: str) -> int:
        return sum(any(row['new_id_rank'] <= limit and row[layout_name] <= limit
                       for row in case['reference_transitions_evaluation_only'])
                   for case in assessed)

    budgets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    recovery_by_budget = [{
        'budget': budget,
        'id_recovered': count('new_id_rank', budget),
        'value_class_recovered': count('class_rank', budget),
        'exact_layout_recovered': count('new_layout_rank', budget),
        'joint_id_exact_recovered': joint(budget, 'new_layout_rank')}
        for budget in budgets]

    rows = []
    for case in cases:
        near_rank, near_class, near_joint_cost = _near_layout_ranks(case, case_dir)
        exact_rank = _best(case, 'new_layout_rank')
        if case['assessable'] and (near_rank is None or exact_rank is None
                                   or near_rank > exact_rank):
            raise ValueError(f'close-layout evaluation missed exact reference: '
                             f"{case['vehicle']} {case['role']}")
        rows.append({'vehicle': case['vehicle'], 'role': case['role'],
            'assessable': case['assessable'],
            'initial_id_rank': _best(case, 'old_id_rank'),
            'unary_id_rank': _best(case, 'unary_id_rank'),
            'contrast_id_rank': _best(case, 'contrast_id_rank'),
            'joint_id_rank': _best(case, 'new_id_rank'),
            'best_class_rank': _best(case, 'class_rank'),
            'near_layout_rank': near_rank,
            'near_class_rank': near_class,
            'near_joint_review_cost': near_joint_cost,
            'v5_exact_layout_rank': _best(case, 'old_layout_rank'),
            'exact_layout_rank': exact_rank,
            'unary_role_rank': _best(case, 'unary_role_rank'),
            'source_contrast_gate': case['source_contrast']['status'],
            'relationship_gate': case['relationship_gate']['status'],
            'trajectory_classes': case['trajectory_class_count'],
            'scored_layouts': case['scored_layout_count']})
    for point in recovery_by_budget:
        point['near_layout_recovered'] = sum(
            row['near_layout_rank'] is not None and
            row['near_layout_rank'] <= point['budget'] for row in rows)
        point['joint_id_near_recovered'] = sum(
            row['near_joint_review_cost'] is not None and
            row['near_joint_review_cost'] <= point['budget'] for row in rows)
    return {'schema': SCHEMA, 'kind': 'development_summary',
        'manifest_hash': artifact_hash(source),
        'staged_manifest_hash': artifact_hash(manifest),
        'cases': rows, 'metrics': {
            'assessable': len(assessed),
            'initial_id_top5': count('old_id_rank', 5),
            'unary_id_top5': count('unary_id_rank', 5),
            'contrast_id_top5': count('contrast_id_rank', 5),
            'joint_id_top5': count('new_id_rank', 5),
            'initial_id_top8': count('old_id_rank', 8),
            'joint_id_top8': count('new_id_rank', 8),
            'class_top8': count('class_rank', 8),
            'v5_exact_layout_top8': count('old_layout_rank', 8),
            'exact_layout_top8': count('new_layout_rank', 8),
            'near_layout_top8': sum(row['near_layout_rank'] is not None and
                                    row['near_layout_rank'] <= 8 for row in rows),
            'joint_id_class_top8': joint(8, 'class_rank'),
            'joint_id_exact_top8': joint(8, 'new_layout_rank'),
            'unary_role_top1': count('unary_role_rank', 1),
            'source_gate_status': dict(Counter(
                case['source_contrast']['status'] for case in cases)),
            'relationship_gate_status': dict(Counter(
                case['relationship_gate']['status'] for case in cases)),
            'layouts_scored': sum(case['scored_layout_count'] for case in cases),
            'trajectory_classes': sum(case['trajectory_class_count'] for case in cases)},
        'recovery_by_budget': recovery_by_budget,
        'final_routes_opened': False, 'final_assessment_allowed': False}


def report(summary: dict) -> str:
    m = summary['metrics']
    lines = ['# Progressive value classes and source evidence, development v6', '',
        'This evaluation reranks the complete candidate pools prepared by the current workflow. Affine-equivalent '
        'value trajectories are grouped within each message ID while preserving all '
        'bit layouts and their exact DBC ranks. The source contrast model uses '
        'other vehicles only and must pass leave-one-vehicle-out tests. The '
        'speed–RPM relationship votes only after replication on source fitting '
        'and development routes. A failed gate contributes zero. Scores are '
        'ranking values, not calibrated probabilities.', '',
        'The hidden target annotation is read only after ranking. All 20 cases '
        'abstain from DBC edits and physical verification; final routes remain sealed.', '',
        '## Aggregate', '', '| Measure | Result |', '| --- | ---: |',
        f"| Assessable cases | {m['assessable']} / {len(summary['cases'])} |",
        f"| Scored layouts / trajectory classes | {m['layouts_scored']} / {m['trajectory_classes']} |",
        f"| ID top 5: initial → unary → contrast → joint | {m['initial_id_top5']} → {m['unary_id_top5']} → {m['contrast_id_top5']} → {m['joint_id_top5']} |",
        f"| ID top 8: initial → joint | {m['initial_id_top8']} → {m['joint_id_top8']} |",
        f"| Exact layout top 8: v5 → v6 | {m['v5_exact_layout_top8']} → {m['exact_layout_top8']} |",
        f"| Value class top 8 / close layout top 8 | {m['class_top8']} / {m['near_layout_top8']} |",
        f"| Joint ID/class top 8 / joint ID/exact layout top 8 | {m['joint_id_class_top8']} / {m['joint_id_exact_top8']} |",
        f"| Hidden reference unary role top 1 | {m['unary_role_top1']} |",
        f"| Source gate statuses | {m['source_gate_status']} |",
        f"| Relationship gate statuses | {m['relationship_gate_status']} |", '',
        'Top eight is shown only for comparison with the earlier pilot; eight is '
        'an arbitrary review budget. The main retrieval diagnostic is how soon a '
        'plausible layout appears. Here, "close" means the same ID, byte order, '
        'and signedness as a reference, at least 80% bit-span intersection over '
        'union, and membership in the reference\'s affine-equivalent value class '
        'on fitting routes. This is a DBC-relative review measure, not independent '
        'physical verification. Exact agreement is reported separately. The '
        'previously agreed development gate for final-route assessment is not met.', '',
        '## Recovery by review budget', '',
        '| Candidates reviewed per stage | ID | Value class within ID | Close layout within ID | Exact layout within ID | Joint ID and close layout | Joint ID and exact layout |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['recovery_by_budget']:
        lines.append(f"| {row['budget']} | {row['id_recovered']} | "
                     f"{row['value_class_recovered']} | {row['near_layout_recovered']} | "
                     f"{row['exact_layout_recovered']} | "
                     f"{row['joint_id_near_recovered']} | "
                     f"{row['joint_id_exact_recovered']} |")
    lines += ['',
        '## Per-vehicle comparison', '',
        '| Vehicle | Assessable | ID top 5 v5 → v6 | Exact layout top 8 v5 → v6 |',
        '| --- | ---: | ---: | ---: |']
    for vehicle in sorted({row['vehicle'] for row in summary['cases']}):
        rows = [row for row in summary['cases']
                if row['vehicle'] == vehicle and row['assessable']]
        old_id = sum(row['initial_id_rank'] is not None and
                     row['initial_id_rank'] <= 5 for row in rows)
        new_id = sum(row['joint_id_rank'] is not None and
                     row['joint_id_rank'] <= 5 for row in rows)
        old_layout = sum(row['v5_exact_layout_rank'] is not None and
                         row['v5_exact_layout_rank'] <= 8 for row in rows)
        new_layout = sum(row['exact_layout_rank'] is not None and
                         row['exact_layout_rank'] <= 8 for row in rows)
        lines.append(f'| {vehicle} | {len(rows)} | {old_id} → {new_id} | '
                     f'{old_layout} → {new_layout} |')
    lines += ['',
        '## Every case', '',
        '| Vehicle | Role | Assessable | ID initial → unary → contrast → joint | Value class / close / exact layout | Role rank | Source / relation gate |',
        '| --- | --- | --- | --- | --- | ---: | --- |']
    def fmt(value):
        return '—' if value is None else str(value)

    for row in sorted(summary['cases'], key=lambda item: (item['vehicle'], item['role'])):
        ids = ' → '.join(fmt(row[name]) for name in (
            'initial_id_rank', 'unary_id_rank', 'contrast_id_rank', 'joint_id_rank'))
        layout = (f"{fmt(row['best_class_rank'])} / "
                  f"{fmt(row['near_layout_rank'])} / {fmt(row['exact_layout_rank'])}")
        gates = f"{row['source_contrast_gate']} / {row['relationship_gate']}"
        lines.append(f"| {row['vehicle']} | {row['role']} | {'yes' if row['assessable'] else 'no'} | {ids} | {layout} | {fmt(row['unary_role_rank'])} | {gates} |")
    lines += ['', '## Interpretation', '',
        'Source hard competitors are nonreference candidate families, not '
        'independently disproved physical signals. Source annotations serve as '
        'training references only on other vehicles. The role rank is the unary '
        'role ordering of the exact reference layout and is an evaluation '
        'diagnostic, not a role assignment made by the review ledger. The two '
        'eligible relationship cases did not improve reference ID rank; one had '
        'no assessable DBC target. Per-case '
        'JSON files contain all rank transitions, gate evidence, and provenance.', '']
    return '\n'.join(lines)
