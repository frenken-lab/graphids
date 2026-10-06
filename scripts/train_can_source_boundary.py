"""Train an explicit source-only CAN-D boundary artifact from a pinned manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.preprocessing.capture import load_capture
from can_discovery.preprocessing.source_model import train_source_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--target-vehicle',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples-per-capture',type=int,default=1000)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    entries = entries['captures'] if isinstance(entries,dict) else entries
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else args.manifest.parent/path
    vehicles = []
    for entry in entries:
        if entry['vehicle'] == args.target_vehicle:
            continue
        captures = [load_capture(resolve(p),metadata=entry)
                    for p in [*entry['references'],entry['held_out']]]
        vehicles.append((entry['vehicle'],captures,resolve(entry['dbc'])))
    if len(vehicles) != 3:
        raise ValueError('expected exactly three source vehicles')
    model = train_source_model(vehicles,samples_per_capture=args.samples_per_capture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    model.save(args.output)
    print(json.dumps({'version':model.version,'feature_version':model.feature_version,
                      'training_vehicles':model.training_vehicles,
                      'capture_hashes':model.capture_hashes,'dbc_hashes':model.dbc_hashes,
                      'validation':model.validation},indent=2))


if __name__ == '__main__':
    main()
