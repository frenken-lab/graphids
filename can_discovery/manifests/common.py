"""Shared content hashes and immutable JSON artifact writes."""
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
