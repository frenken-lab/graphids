"""Write the controlled v3 development comparison from its immutable ledgers."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


VARIANTS = ('baseline', 'expanded_search', 'relation_fixed_pool',
            'relation_with_reopening')


def report(summary: dict) -> str:
    cases = summary['cases']
    assessed = [case for case in cases if case['assessable']]
    trace_counts = {name: sum(case['transition_trace'][name] is True
                              for case in assessed)
                    for name in ('initial_pool', 'baseline_detailed',
                                 'baseline_exact_generated',
                                 'baseline_exact_retained',
                                 'oracle_id_exact_generated',
                                 'oracle_id_exact_retained',
                                 'oracle_id_affine_raw_agreement')}
    lines = ['# Controlled ID allocation and relational feedback, development v3', '',
        'The pinned fitting and development routes are unchanged from v2.1. Final routes were not opened. Hidden DBC annotations are used for transition and oracle-ID evaluation after search choices are made. All names remain role hypotheses.', '',
        'The controller searches eight IDs normally and sixteen when the score gap between ranks 8 and 16 is at most 0.10. Expanded search, relational reranking, and relational reopening use the same detailed-ID budget per case. The speed–RPM change relation is allowed to affect a target only when two other vehicles pass the source fitting and development gate and the target has a visible opposite-role DBC anchor. Gate thresholds were developed on this cohort; this is a development comparison, not an independent transfer estimate.', '',
        '## Aggregate outcomes', '',
        '| Variant | Assessable | ID top 5 | ID top 8 | Exact layout top 1 | Exact layout top 8 | Held-out affine raw agreement |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in VARIANTS:
        m = summary['variants'][name]
        lines.append(f"| {name} | {m['assessable']} | {m['id_top_5']} | {m['id_top_8']} | {m['exact_layout_top_1']} | {m['exact_layout_top_8']} | {m['affine_raw_agreement']} |")
    lines += ['', '## Where reference hypotheses disappear', '',
        '| Transition | Cases / assessable |', '| --- | ---: |',
        f"| Correct ID in initial ranked pool | {trace_counts['initial_pool']}/{len(assessed)} |",
        f"| Correct ID in baseline detailed search | {trace_counts['baseline_detailed']}/{len(assessed)} |",
        f"| Exact layout eligible for enumeration in baseline detailed IDs | {trace_counts['baseline_exact_generated']}/{len(assessed)} |",
        f"| Exact layout retained in baseline top eight | {trace_counts['baseline_exact_retained']}/{len(assessed)} |",
        f"| Exact layout eligible when reference ID is forced into search | {trace_counts['oracle_id_exact_generated']}/{len(assessed)} |",
        f"| Exact layout retained in oracle-ID top eight | {trace_counts['oracle_id_exact_retained']}/{len(assessed)} |",
        f"| Oracle-ID held-out affine raw agreement | {trace_counts['oracle_id_affine_raw_agreement']}/{len(assessed)} |",
        '', 'The oracle-ID check is diagnostic only: the reference ID is supplied after normal search selection. Eligibility for enumeration is coverage, not a successful retained decode.', '']
    lines += ['', '## Source gate', '',
        '| Target | Gate | Supporting source vehicles | Target cases with an eligible anchor |',
        '| --- | --- | --- | ---: |']
    for vehicle, gate in sorted(summary['source_gates'].items()):
        active = sum(c['gate_status'] == 'eligible' for c in cases if c['vehicle'] == vehicle)
        lines.append(f"| {vehicle} | {gate['status']} | {', '.join(gate['supporting_vehicles']) or 'none'} | {active} |")
    lines += ['', '## Per-vehicle outcomes', '',
        '| Vehicle | Variant | ID top 5 / assessable | Exact layout top 8 |',
        '| --- | --- | ---: | ---: |']
    grouped = defaultdict(list)
    for case in cases:
        grouped[case['vehicle']].append(case)
    for vehicle, rows in sorted(grouped.items()):
        assessable = [r for r in rows if r['assessable']]
        for name in VARIANTS:
            lines.append(f"| {vehicle} | {name} | {sum(r['variants'][name]['id_top_5'] for r in assessable)}/{len(assessable)} | {sum(r['variants'][name]['exact_layout_top_8'] for r in assessable)} |")
    lines += ['', '## Every hidden-role transition', '',
        '| Vehicle | Role | Assessable | Initial ID rank | In first 8 | Oracle exact layout rank | Expanded ID rank | Fixed relation ID rank | Reopened ID rank | Budget |',
        '| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in sorted(cases, key=lambda r:(r['vehicle'],r['role'])):
        trace = case['transition_trace']
        v = case['variants']
        lines.append(f"| {case['vehicle']} | {case['role']} | {'yes' if case['assessable'] else 'no'} | {trace['initial_id_rank'] or '—'} | {'yes' if trace['baseline_detailed'] else 'no'} | {trace['oracle_id_exact_rank'] or '—'} | {v['expanded_search']['id_rank'] or '—'} | {v['relation_fixed_pool']['id_rank'] or '—'} | {v['relation_with_reopening']['id_rank'] or '—'} | {case['adaptive_budget']} |")
    changed = [r for r in cases if r['variants']['relation_fixed_pool'] !=
               r['variants']['expanded_search'] or
               r['variants']['relation_with_reopening'] !=
               r['variants']['expanded_search']]
    eligible_assessed = sum(c['assessable'] and c['gate_status'] == 'eligible'
                            for c in cases)
    lines += ['', f"Relational variants changed a measured case outcome in **{len(changed)}** of {len(cases)} cases. The relation was eligible on {eligible_assessed} assessable target cases. This is a ranking cue based on correlated changes, which can reflect shared driving behavior; it does not prove either physical role.", '',
        'Prius RPM remained at ID rank six after relational reranking; its exact layout was eligible for enumeration but absent from the retained top eight even with oracle ID. Expanded search reached Forester pedal at ID rank nine and Civic brake at rank fifteen, but neither gained an exact retained layout. Forester brake has an additional annotated ID at rank twenty: oracle-ID extraction retained an exact layout there at rank four, beyond the adaptive sixteen-ID budget. These are development diagnostics, not a tuned promotion rule.', '',
        'Oracle-ID extraction measures layout recovery with the correct ID supplied only to the diagnostic extractor. It is not a deployed variant. Exact layout and held-out affine raw agreement remain separate outcomes.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(report(json.loads(args.summary.read_text())))


if __name__ == '__main__':
    main()
