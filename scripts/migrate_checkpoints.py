#!/usr/bin/env python3
"""Migrate checkpoint hyper_parameters from nested config to flat format.

The config refactor (2026-03-28) flattened all LightningModule __init__
signatures. Old checkpoints store nested hparams like:

    {"vgae": {"hidden_dims": [480, 240, 48], ...}, "training": {"lr": 0.003, ...}}

New format is flat:

    {"hidden_dims": [480, 240, 48], "lr": 0.003, ...}

This script rewrites hyper_parameters in-place (or with --dry-run).

Usage:
    python scripts/migrate_checkpoints.py experimentruns/
    python scripts/migrate_checkpoints.py experimentruns/ --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Nested keys that get promoted to top level (no prefix)
PROMOTE_KEYS = {"vgae", "gat", "dgi", "training", "fusion", "evaluation", "preprocessing"}

# Nested keys that get promoted with a prefix to avoid collision
PREFIX_KEYS = {"dqn": "dqn_", "bandit": "bandit_"}


def flatten_hparams(hp: dict) -> dict:
    """Flatten nested hparams dict to flat format."""
    flat = {}
    for k, v in hp.items():
        if k in PROMOTE_KEYS and isinstance(v, dict):
            for inner_k, inner_v in v.items():
                if inner_k not in flat:  # first writer wins (top-level > nested)
                    flat[inner_k] = inner_v
        elif k in PREFIX_KEYS and isinstance(v, dict):
            prefix = PREFIX_KEYS[k]
            for inner_k, inner_v in v.items():
                flat[f"{prefix}{inner_k}"] = inner_v
        else:
            flat[k] = v
    return flat


def is_nested(hp: dict) -> bool:
    """Check if hparams dict uses the old nested format."""
    return any(k in hp and isinstance(hp[k], dict) for k in PROMOTE_KEYS | PREFIX_KEYS.keys())


def migrate_checkpoint(path: Path, dry_run: bool = False) -> bool:
    """Migrate a single checkpoint file. Returns True if modified."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters")
    if hp is None or not is_nested(hp):
        return False

    flat = flatten_hparams(hp)

    if dry_run:
        nested_keys = [k for k in hp if k in PROMOTE_KEYS or k in PREFIX_KEYS]
        print(f"  WOULD migrate: {len(hp)} keys → {len(flat)} keys "
              f"(flatten: {', '.join(nested_keys)})")
        return True

    ckpt["hyper_parameters"] = flat
    torch.save(ckpt, path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoints to flat hparams format")
    parser.add_argument("root", type=Path, help="Root directory to search for .ckpt files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Error: {args.root} is not a directory", file=sys.stderr)
        sys.exit(1)

    ckpt_files = sorted(args.root.rglob("*.ckpt"))
    if not ckpt_files:
        print(f"No .ckpt files found under {args.root}")
        return

    print(f"Found {len(ckpt_files)} checkpoint(s) under {args.root}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    migrated = 0
    skipped = 0
    for path in ckpt_files:
        rel = path.relative_to(args.root)
        try:
            if migrate_checkpoint(path, dry_run=args.dry_run):
                print(f"  {'[dry-run] ' if args.dry_run else ''}migrated: {rel}")
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  FAILED: {rel}: {e}", file=sys.stderr)

    print(f"\nDone: {migrated} migrated, {skipped} already flat")


if __name__ == "__main__":
    main()
