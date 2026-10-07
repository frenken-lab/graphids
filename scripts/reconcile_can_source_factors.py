"""Compare a historical factor audit with per-fold source gate diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.manifests.common import artifact_hash
from can_discovery.manifests.iterative import verify_manifest


def reconcile(manifest: dict, audit: dict, models: list[dict]) -> dict:
    verify_manifest(manifest)
    old = {item['vehicle']: item for item in audit['vehicles']}
    source = {}
    for model in models:
        if model['manifest_hash'] != artifact_hash(manifest):
            raise ValueError('shared model manifest hash mismatch')
        for item in model['factor_diagnostics']:
            source[(item['vehicle'], item['factor'])] = item
    vehicles = []
    for entry in manifest['vehicles']:
        name = entry['vehicle']
        historical = old.get(name)
        capture_hashes = [capture['sha256']
                          for split in ('fitting', 'development_validation')
                          for capture in entry['captures'][split]]
        if historical and capture_hashes != historical['capture_hashes']:
            raise ValueError(f'{name}: historical capture hash mismatch')
        factors = {}
        for factor in sorted({key[1] for key in source if key[0] == name}):
            current = source[(name, factor)]
            prior = historical['factors'].get(factor) if historical else None
            old_fields = prior.get('fields') if prior else None
            new_fields = list(current['fields'].values()) if current['fields'] else None
            factors[factor] = {'status': current['status'], 'reason': current['reason'],
                               'fields': current['fields'],
                               'fitting': current.get('fitting'),
                               'validation': current.get('validation'),
                               'historical_status': prior['status'] if prior else None,
                               'historical_fields': old_fields,
                               'same_field_choices': set(old_fields) == set(new_fields)
                               if old_fields and new_fields else None}
        vehicles.append({'vehicle': name, 'capture_hashes': capture_hashes,
                         'historical_audit_present': historical is not None,
                         'capture_hashes_match': historical is not None,
                         'factors': factors})
    return {'schema': 'can-factor-gate-reconciliation-v1',
            'manifest_hash': artifact_hash(manifest), 'historical_audit_version': audit['version'],
            'historical_validated_transfer_factors': audit['validated_transfer_factors'],
            'fold_gates': {model['target_vehicle']: model['factor_gate'] for model in models},
            'vehicles': vehicles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen_manifest', type=Path)
    parser.add_argument('historical_audit', type=Path)
    parser.add_argument('shared_models', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = reconcile(json.loads(args.frozen_manifest.read_text()),
                       json.loads(args.historical_audit.read_text()),
                       [json.loads(path.read_text()) for path in args.shared_models])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
