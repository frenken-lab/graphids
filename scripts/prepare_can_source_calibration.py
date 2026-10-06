"""Prepare pinned source calibration pairs and distinct held-out benchmark routes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

import capnp
import zstandard

try:
    from .prepare_comma_decoder_benchmark import DATA_REV, DBC_REV, HF, PILOT, fetch
except ImportError:  # direct script execution
    from prepare_comma_decoder_benchmark import DATA_REV, DBC_REV, HF, PILOT, fetch

SOURCES = {
    "TOYOTA_PRIUS": (
        "opendbc/dbc/generator/toyota/toyota_new_mc_pt.dbc",
        (("f8d13f3d51e5adc7/000000e5--f449c725dd/17", "d07c837b63128ebc302cfee0a191542eac0f1ae400ac4f4d6023780e0d47513b"),
         ("cbc67147493fd496/000002f4--9c40d8b029/126", "44fddcb0a0a3f1879c711883698672943e2cefb1fdf2bdfb34e4324750bb4fac"),
         ("72c8dc9054796c61/00000000--b0fd5dc2f7/13", "66b5c9e47942501d583e2d0d16b5fd69623fddc81c5a2794aff5d007a8291e21"))),
    "HONDA_CIVIC": (
        "opendbc/dbc/generator/honda/honda_civic_touring_2016_can.dbc",
        (("29c365ba7d91f32f/000008de--9e87b8847f/10", "3454d97448c115dca6bc2db9a5f69430aae140407de5ca82cca90eb41105eae6"),
         ("0945b6c6b3ebaa9f/0000011f--2a2b5b9453/44", "395df5c9b270caa0c5daed26027ccfcdd8283018c5e408a455e633eb1c19cbb7"),
         ("b083f4855277bac4/00000023--4e93a8090b/13", "9564d4e8c1f8e8e239b1748b250da77f7631a677f1178136c0d526319f8b45c3"))),
    "HYUNDAI_SONATA": (
        "opendbc/dbc/generator/hyundai/hyundai_can.dbc",
        (("a3488ea00ac3bacc/00000099--d7710733db/30", "a9d2146c24f2dccbe8343c6e48e2249154d40114250a21e4957313b474939267"),
         ("20caa585ef1501af/000000f1--7061a7480a/34", "e7eca78bc41fd0d4a0e17c131a210fc9c1c648329a0a0f2883b61e2bca5de097"),
         ("c11af183f22aef82/00000053--ab441b2563/17", "5f7747431ae1641344a3cc952e22a7309fa0edf6e77cce138a067d75b283a90a"))),
}


def _assembled_dbc(path: str) -> str:
    base = f"https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/"
    raw = fetch(base + path).decode()
    includes = re.findall(r'CM_ "IMPORT ([^" ]+)";', raw)
    parts = [fetch(base + str(Path(path).parent / item)).decode() for item in includes]
    parts.append(re.sub(r'CM_ "IMPORT [^" ]+";', '', raw))
    content = '\n'.join(parts)
    # Some upstream comments omit the BO_ token or a signal name. These
    # comments do not affect any signal layout and cantools rejects them.
    content = '\n'.join(line for line in content.splitlines()
                        if not re.match(r'^CM_ SG_ \d+ "', line))
    return re.sub(r'^CM_ (\d+) ', r'CM_ BO_ \1 ', content, flags=re.M) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    database = json.loads(fetch(f'{HF}/database.json'))
    manifest = []
    benchmarks = []
    provenance = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'include').mkdir()
        for name in ('log.capnp', 'custom.capnp', 'deprecated.capnp'):
            (root / name).write_bytes(fetch(f'{PILOT}/{name}'))
        (root / 'car.capnp').write_bytes(fetch(
            f'https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/opendbc/car/car.capnp'))
        (root / 'include/c++.capnp').write_text(
            '@0xb4e3f8fa30b9318b; annotation namespace(file) :Text;\n')
        schema = capnp.load(str(root / 'log.capnp'), imports=[str(root)])
        for vehicle, (dbc_source, routes) in SOURCES.items():
            if len({route.split('/', 1)[0] for route, _ in routes}) != len(routes):
                raise RuntimeError(f'{vehicle} routes are not independent devices')
            dbc_path = args.output / (Path(dbc_source).stem + '_generated.dbc')
            dbc_path.write_text(_assembled_dbc(dbc_source))
            captures = []
            for route, expected_hash in routes:
                if route + '/s' not in database[vehicle]:
                    raise RuntimeError(f'pinned route absent: {vehicle} {route}')
                compressed = fetch(f'{HF}/segments/{route}/rlog.zst')
                digest = hashlib.sha256(compressed).hexdigest()
                if digest != expected_hash:
                    raise RuntimeError(f'compressed segment hash changed: {route}')
                raw_path = root / 'rlog'
                raw_path.write_bytes(zstandard.ZstdDecompressor().decompress(
                    compressed, max_output_size=200_000_000))
                capture_path = args.output / f'{vehicle}_{route.split("/")[0]}.csv'
                count = 0
                with raw_path.open('rb') as raw, capture_path.open('w', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerow(('timestamp', 'arbitration_id', 'data_field', 'attack'))
                    for event in schema.Event.read_multiple(raw):
                        if event.which() == 'can':
                            for frame in event.can:
                                if frame.src == 0:
                                    writer.writerow((event.logMonoTime / 1e9,
                                                     format(frame.address, 'X'),
                                                     bytes(frame.dat).hex(), 0))
                                    count += 1
                captures.append(str(capture_path))
                provenance.append({'vehicle': vehicle, 'route': route,
                                   'compressed_sha256': digest, 'frames_bus_0': count,
                                   'dataset_revision': DATA_REV, 'opendbc_revision': DBC_REV})
            manifest.append({'vehicle': vehicle, 'captures': captures[:2], 'dbc': str(dbc_path)})
            benchmarks.append({'vehicle': vehicle, 'references': captures[:2],
                               'held_out': captures[2], 'dbc': str(dbc_path)})
    (args.output / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'benchmarks.json').write_text(json.dumps(benchmarks, indent=2) + '\n')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
