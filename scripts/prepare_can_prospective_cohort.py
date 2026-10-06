"""Freeze disjoint public CAN routes for a prospective graph benchmark.

The final capture is converted and hashed, but never decoded or scored here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import capnp
import zstandard

from can_discovery.decoding.dbc import load_dbc_fields, rank_dbc
from can_discovery.decoding.field_values import extract
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical.source_calibration import _dbc_role

try:
    from .prepare_can_source_calibration import SOURCES, _assembled_dbc
    from .prepare_comma_decoder_benchmark import (
        DATA_REV,
        DBC_REV,
        HF,
        PILOT,
        ROUTES,
        fetch,
    )
except ImportError:  # direct script execution
    from prepare_can_source_calibration import SOURCES, _assembled_dbc
    from prepare_comma_decoder_benchmark import (
        DATA_REV,
        DBC_REV,
        HF,
        PILOT,
        ROUTES,
        fetch,
    )


NEW_VEHICLES = (
    ('NISSAN_LEAF', 'opendbc/dbc/generator/nissan/nissan_leaf_2018.dbc'),
    ('VOLKSWAGEN_GOLF_MK7', 'opendbc/dbc/vw_mqb.dbc'),
    ('CHEVROLET_VOLT', 'opendbc/dbc/generator/gm/gm_global_a_powertrain.dbc'),
)
FORESTER_DBC = 'opendbc/dbc/generator/subaru/subaru_global_2017.dbc'


def distinct_routes(catalog: dict, vehicle: str, count: int, pinned=()) -> list[str]:
    """Choose catalog routes in stable order, retaining existing historical trips."""
    available = {item.removesuffix('/s') for item in catalog.get(vehicle, ())}
    selected = []
    devices = set()
    for route in (*pinned, *sorted(available)):
        device = route.split('/', 1)[0]
        if route not in available:
            raise ValueError(f'pinned route absent: {vehicle} {route}')
        if device not in devices:
            selected.append(route)
            devices.add(device)
        if len(selected) == count:
            break
    return selected


def _schema(root: Path):
    (root / 'include').mkdir()
    for name in ('log.capnp', 'custom.capnp', 'deprecated.capnp'):
        (root / name).write_bytes(fetch(f'{PILOT}/{name}'))
    (root / 'car.capnp').write_bytes(fetch(
        f'https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/opendbc/car/car.capnp'))
    (root / 'include/c++.capnp').write_text(
        '@0xb4e3f8fa30b9318b; annotation namespace(file) :Text;\n')
    return capnp.load(str(root / 'log.capnp'), imports=[str(root)])


def _convert(route: str, vehicle: str, role: str, output: Path, schema, scratch: Path,
             expected_hash: str | None = None) -> dict:
    compressed = fetch(f'{HF}/segments/{route}/rlog.zst')
    compressed_hash = hashlib.sha256(compressed).hexdigest()
    if expected_hash and compressed_hash != expected_hash:
        raise ValueError(f'pinned compressed segment hash changed: {route}')
    raw = scratch / 'rlog'
    raw.write_bytes(zstandard.ZstdDecompressor().decompress(
        compressed, max_output_size=200_000_000))
    path = output / f'{vehicle}_{role}.csv'
    frames = 0
    with raw.open('rb') as input_file, path.open('w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(('timestamp', 'arbitration_id', 'data_field', 'attack'))
        for event in schema.Event.read_multiple(input_file):
            if event.which() == 'can':
                for frame in event.can:
                    if frame.src == 0:
                        writer.writerow((event.logMonoTime / 1e9,
                                         format(frame.address, 'X'), bytes(frame.dat).hex(), 0))
                        frames += 1
    return {'vehicle': vehicle, 'role': role, 'route': route,
            'device': route.split('/')[0], 'session': route.split('/')[1],
            'segment': route.split('/')[2], 'bus': 0,
            'dataset_revision': DATA_REV, 'opendbc_revision': DBC_REV,
            'compressed_sha256': compressed_hash,
            'converted_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'frames': frames, 'capture': str(path)}


def _active(field, capture) -> bool:
    values = set()
    for frame in capture.frames:
        if frame.arb_id != field.arb_id or frame.payload is None or len(frame.payload) != field.payload_length:
            continue
        if field.mux_start is not None and frame.payload[field.mux_start // 8] != field.mux_value:
            continue
        values.add(extract(frame.payload, field.start, field.width,
                           field.endian, field.signed))
        if len(values) > 1:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in ('prospective-manifest.json', 'prospective-failed-preflight.json'):
        (args.output / stale).unlink(missing_ok=True)
    catalog = json.loads(fetch(f'{HF}/database.json'))
    existing = [(name, path, tuple(route for route, _ in routes))
                for name, (path, routes) in SOURCES.items()]
    existing.append(('SUBARU_FORESTER', FORESTER_DBC,
                     tuple(route for route, _ in ROUTES)))
    candidates = [*existing, *((name, path, ()) for name, path in NEW_VEHICLES)]
    pinned_hashes = {route: digest for _, routes in SOURCES.values()
                     for route, digest in routes}
    pinned_hashes.update(dict(ROUTES))
    preflight = []
    manifest = []
    provenance = []
    accepted_new = 0
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        schema = _schema(scratch)
        for vehicle, dbc_source, pinned in candidates:
            if not pinned and accepted_new == 2:
                break
            routes = distinct_routes(catalog, vehicle, 5, pinned)
            outcome = {'vehicle': vehicle, 'routes_available': len(routes)}
            if len(routes) < 5:
                outcome['reason'] = 'fewer_than_five_device_distinct_routes'
                preflight.append(outcome)
                continue
            dbc = args.output / f'{vehicle}.dbc'
            if vehicle == 'SUBARU_FORESTER':
                base = fetch(f'https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/opendbc/dbc/generator/subaru/_subaru_global.dbc').decode()
                extension = fetch(f'https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/{dbc_source}').decode()
                dbc.write_text(base + '\n' + extension.split('\n', 1)[1])
            else:
                dbc.write_text(_assembled_dbc(dbc_source))
            dbc_hash = hashlib.sha256(dbc.read_bytes()).hexdigest()
            # Existing vehicles retain historical fitting and validation routes.
            labels = ('fit_1', 'fit_2', 'dev_1', 'dev_2', 'final') if not pinned else (
                'fit_1', 'fit_2', 'dev_1', 'dev_2', 'final')
            records = []
            failed = None
            for route, role in zip(routes, labels):
                record = _convert(route, vehicle, role, args.output, schema, scratch,
                                  pinned_hashes.get(route))
                record['dbc_sha256'] = dbc_hash
                records.append(record)
                if role == 'final':
                    continue  # sealed: no DBC fit or role assessment
                capture = load_capture(Path(record['capture']))
                fit = rank_dbc(capture, dbc)
                record['dbc_coverage'] = fit['coverage']
                if fit['coverage'] < .5:
                    failed = 'dbc_coverage_below_half'
                    break
            if failed is None:
                development = [load_capture(Path(item['capture'])) for item in records[:4]]
                roles = {_dbc_role(field.label_hypotheses[0][0])
                         for field in load_dbc_fields(dbc)
                         if all(_active(field, capture) for capture in development)}
                roles.discard(None)
                outcome['physical_roles'] = sorted(roles)
                if len(roles) < 3:
                    failed = 'fewer_than_three_physical_roles'
            outcome['dbc_sha256'] = dbc_hash
            outcome['reason'] = failed
            preflight.append(outcome)
            provenance.extend(records)
            if failed:
                continue
            if not pinned:
                accepted_new += 1
            manifest.append({'vehicle': vehicle,
                             'fitting': [item['capture'] for item in records[:2]],
                             'validation': [item['capture'] for item in records[2:4]],
                             'final': records[4]['capture'], 'dbc': str(dbc),
                             'routes': records})
            (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            (args.output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if accepted_new < 2 or len(manifest) < 6:
        (args.output / 'prospective-failed-preflight.json').write_text(
            json.dumps({'accepted_new_vehicles': accepted_new, 'preflight': preflight}, indent=2) + '\n')
        raise SystemExit('prospective cohort failed preflight; no final evaluation is allowed')
    (args.output / 'prospective-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output / 'prospective-manifest.json')


if __name__ == '__main__':
    main()
