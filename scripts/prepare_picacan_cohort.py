"""Freeze contiguous commaCarSegments windows for the experimental ICE benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.prepare_can_prospective_cohort import _convert, _schema
from scripts.prepare_can_source_calibration import SOURCES, _assembled_dbc
from scripts.prepare_comma_decoder_benchmark import DATA_REV, DBC_REV, HF, PILOT, fetch

VEHICLES = ('HONDA_CIVIC', 'HYUNDAI_SONATA')
MINIMUM = {'fitting': 20, 'development': 8, 'sealed': 8}


def contiguous_windows(catalog, vehicle, excluded=()):
    sessions = {}
    excluded = set(excluded)
    for item in catalog.get(vehicle, ()):
        route = item.removesuffix('/s')
        if route in excluded:
            continue
        parts = route.split('/')
        if len(parts) != 3:
            continue
        try:
            segment = int(parts[2])
        except ValueError:
            continue
        sessions.setdefault(tuple(parts[:2]), set()).add(segment)
    windows = []
    for (device, session), indices in sessions.items():
        run = []
        for index in sorted(indices):
            if run and index != run[-1]+1:
                windows.append((device, session, tuple(run)))
                run = []
            run.append(index)
        if run:
            windows.append((device, session, tuple(run)))
    return windows


def select_windows(catalog, vehicle, excluded=()):
    unused = contiguous_windows(catalog, vehicle, excluded)
    selected = {}
    devices = set()
    for role, minimum in MINIMUM.items():
        choices = [w for w in unused if w[0] not in devices and len(w[2]) >= minimum]
        if not choices:
            raise ValueError(f'{vehicle}: no {role} window with {minimum} segments on a new device')
        chosen = sorted(choices, key=lambda w: (-len(w[2]), w[0], w[1], w[2][0]))[0]
        selected[role] = [f'{chosen[0]}/{chosen[1]}/{i}' for i in chosen[2]]
        devices.add(chosen[0])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--powertrain-evidence', type=Path,
                        help='JSON vehicle to {powertrain: ICE, source: capture metadata reference}')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_bytes = fetch(f'{HF}/database.json')
    catalog = json.loads(catalog_bytes)
    evidence = json.loads(args.powertrain_evidence.read_text()) if args.powertrain_evidence else {}
    manifest = {'version': 1, 'dataset_revision': DATA_REV, 'pilot_revision': PILOT,
                'opendbc_revision': DBC_REV, 'bus': 0,
                'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(), 'vehicles': []}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        schema = _schema(root)
        for vehicle in VEHICLES:
            historical = {route for route, _ in SOURCES[vehicle][1]}
            windows = select_windows(catalog, vehicle, historical)
            metadata = evidence.get(vehicle, {})
            if metadata and (metadata.get('powertrain') != 'ICE' or not metadata.get('source')):
                raise ValueError(f'{vehicle}: powertrain evidence needs ICE and source')
            dbc_source = SOURCES[vehicle][0]
            dbc = args.output / f'{vehicle}.dbc'
            dbc.write_text(_assembled_dbc(dbc_source))
            dbc_hash = hashlib.sha256(dbc.read_bytes()).hexdigest()
            records = {}
            for role, routes in windows.items():
                records[role] = []
                for index, route in enumerate(routes):
                    record = _convert(route, vehicle, f'{role}_{index:03d}', args.output, schema, root)
                    record['dbc_sha256'] = dbc_hash
                    records[role].append(record)
            manifest['vehicles'].append({'vehicle': vehicle,
                                         'powertrain': metadata.get('powertrain', 'unverified'),
                                         'powertrain_evidence': metadata.get('source',
                                                                     'catalog_platform_label_only'),
                                         'dbc': str(dbc), 'dbc_sha256': dbc_hash,
                                         'windows': records})
    path = args.output / 'picacan-manifest-v1.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
