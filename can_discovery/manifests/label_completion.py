"""Pinned five-route manifests and immutable artifact helpers."""
from __future__ import annotations

from pathlib import Path

from .common import artifact_hash, file_hash, write_artifact


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


def verify_staged_manifest(source: dict, staged: dict) -> None:
    """Require identical cohort, split order, and pinned content hashes."""
    if len(source['vehicles']) != len(staged['vehicles']):
        raise ValueError('staged manifest vehicle count mismatch')
    for original, copied in zip(source['vehicles'], staged['vehicles']):
        if original['vehicle'] != copied['vehicle']:
            raise ValueError('staged vehicle order mismatch')
        if original['dbc']['sha256'] != copied['dbc']['sha256']:
            raise ValueError('staged DBC hash mismatch')
        for split in ('fitting', 'development', 'final'):
            if [x['sha256'] for x in original['routes'][split]] != [
                    x['sha256'] for x in copied['routes'][split]]:
                raise ValueError(f'staged {split} hash mismatch')
        if original['routes']['final'] != copied['routes']['final']:
            raise ValueError('sealed final route changed in staged manifest')
