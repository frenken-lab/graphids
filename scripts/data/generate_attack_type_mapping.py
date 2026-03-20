#!/usr/bin/env python3
"""Generate attack type mapping JSON for paper exports.

Reads the canonical ATTACK_TYPE_CODES from the CAN bus adapter and writes
a mapping file to ESS exports/paper/metadata/ for use by export_paper_data.py
and by the paper's pull_data.py validation.

Usage:
    python scripts/data/generate_attack_type_mapping.py [--dataset hcrl_sa]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure graphids is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphids.core.preprocessing import (
    ATTACK_TYPE_CODES,
    ATTACK_TYPE_NAMES,
)
from graphids.config.paths import lake_exports_dir, lake_root_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attack type mapping JSON")
    parser.add_argument("--dataset", default="hcrl_sa", help="Dataset name (default: hcrl_sa)")
    args = parser.parse_args()

    lake_root = lake_root_from_env()
    if lake_root is None:
        print("ERROR: KD_GAT_LAKE_ROOT not set", file=sys.stderr)
        sys.exit(1)

    out_dir = lake_exports_dir(lake_root) / "paper" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "dataset": args.dataset,
        "code_to_name": {str(k): v for k, v in ATTACK_TYPE_NAMES.items()},
        "name_to_code": ATTACK_TYPE_CODES,
    }

    out_path = out_dir / "attack_type_mapping.json"
    out_path.write_text(json.dumps(mapping, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
