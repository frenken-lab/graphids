"""Summarize frozen development ledgers alongside the historical v1 audit."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def report(summary_path: Path, baseline_path: Path) -> str:
    summary = json.loads(summary_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    old = defaultdict(list)
    for case in baseline['cases']:
        old[case['vehicle']].append(case)
    new = defaultdict(list)
    for case in summary['cases']:
        new[case['vehicle']].append(case)
    ledgers = [json.loads(Path(case['path']).read_text()) for case in summary['cases']]
    ambiguous = sum(group['ambiguity'] for ledger in ledgers
                    for group in ledger['detailed_families'])
    lines = ['# ID-first DBC annotation completion, development v2.1', '',
        'This is a review ledger against DBC annotations. A proposed name remains a role hypothesis; no DBC was edited and no physical role was independently verified.', '',
        'The pinned manifest uses two fitting routes, two development routes, and one sealed final route per vehicle. The final routes were not opened for this report. Scores are ranks, not confidence probabilities. The method ranks message families by their best candidate score, details eight families, and retains eight ranked field-layout alternatives per family. Relationship factors were not used. Method v2.1 removes equivalent one-bit interpretations from v2; both development artifacts are preserved.', '',
        '## Development gate', '',
        '| Measure | Frozen v1 | v2.1 |', '| --- | ---: | ---: |',
        f"| Assessable cases | {summary['baseline_v1']['assessable']} | {summary['metrics']['assessable']} |",
        f"| ID top one | {summary['baseline_v1']['id_top_1']} | {summary['metrics']['id_top_1']} |",
        f"| ID top five | {summary['baseline_v1']['id_top_5']} | {summary['metrics']['id_top_5']} |",
        f"| ID top eight | — | {summary['metrics']['id_top_8']} |",
        f"| Exact layout top one | 0 | {summary['metrics']['exact_layout_top_1']} |",
        f"| Exact layout top eight | {summary['baseline_v1']['exact_layout_top_8']} | {summary['metrics']['exact_layout_top_8']} |",
        f"| Exact layout in enumerated pool | 0 | {summary['metrics']['pool_coverage']} |",
        f"| Held-out raw-value agreement | — | {summary['metrics']['raw_value_agreement']} |",
        f"| Physical role rank one, conditional on exact retained layout | — | {summary['metrics']['role_top_1']} |",
        f"| Ambiguous detailed families | — | {ambiguous} |",
        f"| Abstaining role ledgers | — | {sum(c['metrics']['abstained'] for c in summary['cases'])} |",
        '', f"Final-route assessment permitted: **{'yes' if summary['final_assessment_allowed'] else 'no'}**. The gate requires at least 7 ID top-five cases and at least one exact-layout top-eight case. Pool coverage does not satisfy the layout-rank gate.", '',
        f"Conditional on ID top-eight coverage, exact layout is top eight in {summary['metrics']['exact_layout_top_8']}/{summary['metrics']['id_top_8']} assessable cases and top one in {summary['metrics']['exact_layout_top_1']}/{summary['metrics']['id_top_8']}.", '',
        'On the nine cases shared with v1, ID top-five is 6/9. The additional assessable case is Forester accelerator pedal, ranked ninth. The lost shared case is Prius RPM, ranked sixth.', '',
        '## Per-vehicle comparison', '',
        '| Vehicle | v1 ID top five / assessable | v2.1 ID top five / assessable | v2.1 exact top eight |',
        '| --- | ---: | ---: | ---: |']
    for vehicle in sorted(new):
        prior = old[vehicle]
        current = [c for c in new[vehicle] if c['assessable']]
        lines.append(f"| {vehicle} | {sum(c['id_top_5'] for c in prior)}/{len(prior)} | "
                     f"{sum(c['metrics']['id_top_5'] for c in current)}/{len(current)} | "
                     f"{sum(c['metrics']['exact_layout_top_8'] for c in current)} |")
    lines += ['', '## Every hidden-role case', '',
        '| Vehicle | Hidden role | Assessable | ID rank | Exact layout rank | Role rank | Pool coverage | Raw agreement |',
        '| --- | --- | --- | ---: | ---: | ---: | --- | --- |']
    for vehicle in sorted(new):
        for case in new[vehicle]:
            m = case['metrics']
            lines.append(f"| {vehicle} | {case['role']} | {'yes' if case['assessable'] else 'no'} | "
                         f"{m.get('id_rank') or '—'} | {m.get('exact_layout_rank_conditional_on_id_top_8') or '—'} | {m.get('role_rank') or '—'} | "
                         f"{'yes' if m['exact_layout_pool_coverage'] else 'no'} | "
                         f"{'yes' if m['held_out_raw_value_agreement'] else 'no'} |")
    lines += ['', 'A layout in the exhaustive candidate pool is only coverage. Role rank is conditional on retaining the exact reference layout; unary role ranking is not independent physical verification. Historical v1 artifacts remain in the parent directory.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(report(args.summary, args.baseline))


if __name__ == '__main__':
    main()
