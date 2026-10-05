"""Input path resolution for standalone discovery scripts."""

import os
from pathlib import Path


def raw_root() -> Path:
    lake = os.environ.get("GRAPHIDS_LAKE_ROOT")
    if not lake:
        raise RuntimeError("provide --raw-root or set GRAPHIDS_LAKE_ROOT")
    return Path(lake) / "raw"
