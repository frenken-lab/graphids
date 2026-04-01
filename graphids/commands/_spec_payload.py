"""Shared helpers for command modules that consume serialized spec payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_payload(spec_file: str) -> dict[str, Any]:
    """Load JSON payload from a spec file path."""
    return json.loads(Path(spec_file).read_text())
