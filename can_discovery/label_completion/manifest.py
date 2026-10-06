"""Pinned five-route manifests and immutable artifact helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def artifact_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_artifact(path: Path, value: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f'immutable artifact already exists: {path}')
    path.write_text(encoded)
    return artifact_hash(value)


def freeze_manifest(entries: list[dict]) -> dict:
    if len({e['vehicle'] for e in entries}) != len(entries):
        raise ValueError('duplicate vehicle')
    used = set()
    frozen = []
    for entry in entries:
        routes = {}
        for split, count in (('fitting', 2), ('development', 2), ('final', 1)):
            paths = entry[split]
            if isinstance(paths, (str, dict)):
                paths = [paths]
            if len(paths) != count:
                raise ValueError(f"{entry['vehicle']}: {split} requires {count} routes")
            routes[split] = []
            for path in paths:
                if split == 'final':
                    if not isinstance(path, dict) or 'sha256' not in path:
                        raise ValueError('final route requires a pre-pinned sha256')
                    resolved = str(Path(path['path']).resolve())
                    digest = path['sha256']
                    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise ValueError('invalid final route sha256')
                else:
                    resolved = str(Path(path).resolve())
                    digest = file_hash(Path(resolved))
                if digest in used:
                    raise ValueError('capture reused across splits or vehicles')
                used.add(digest)
                routes[split].append({'path': resolved, 'sha256': digest})
        dbc = str(Path(entry['dbc']).resolve())
        frozen.append({'vehicle': entry['vehicle'], 'routes': routes,
                       'dbc': {'path': dbc, 'sha256': file_hash(Path(dbc))}})
    return {'schema': 'dbc-label-completion-manifest-v2', 'vehicles': frozen}


def verify_manifest(manifest: dict, *, include_final: bool = False) -> None:
    if manifest.get('schema') != 'dbc-label-completion-manifest-v2':
        raise ValueError('manifest schema mismatch')
    for entry in manifest['vehicles']:
        items = [entry['dbc'], *entry['routes']['fitting'], *entry['routes']['development']]
        if include_final:
            items.extend(entry['routes']['final'])
        for item in items:
            if file_hash(Path(item['path'])) != item['sha256']:
                raise ValueError(f"input hash changed: {item['path']}")
