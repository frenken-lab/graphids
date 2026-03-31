"""Rebuild preprocessed graph caches after preprocessing changes.

Usage:
    python -m graphids rebuild-caches --dataset hcrl_ch
    python -m graphids rebuild-caches --dataset hcrl_ch hcrl_sa --delete-existing
    python -m graphids rebuild-caches --all
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import structlog

from graphids.config import LAKE_ROOT, PREPROCESSING_VERSION, cache_dir

log = structlog.get_logger()

ALL_DATASETS = ("hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04")

STAGING_MARKER = Path("/fs/scratch/PAS1266/kd-gat-data/cache/.staged_marker")


def rebuild_caches(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild preprocessed graph caches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", nargs="+", choices=ALL_DATASETS)
    group.add_argument("--all", action="store_true", dest="all_datasets")
    parser.add_argument("--delete-existing", action="store_true", help="Remove stale cache before rebuild")
    args = parser.parse_args(argv)

    datasets = list(ALL_DATASETS) if args.all_datasets else args.dataset

    for ds in datasets:
        cdir = cache_dir(LAKE_ROOT, ds)
        if args.delete_existing and cdir.exists():
            log.info("removing_stale_cache", dataset=ds, path=str(cdir))
            shutil.rmtree(cdir)

        log.info("rebuilding_cache", dataset=ds, version=PREPROCESSING_VERSION)
        from graphids.core.preprocessing.datamodule import CANBusDataModule

        dm = CANBusDataModule(dataset=ds, lake_root=LAKE_ROOT)
        dm.setup("fit")

        n_train = len(dm.train_dataset)
        n_val = len(dm.val_dataset)
        log.info(
            "cache_ready",
            dataset=ds,
            graphs=n_train + n_val,
            num_ids=dm.num_ids,
            in_channels=dm.in_channels,
            edge_dim=dm.edge_dim,
        )

    # Invalidate scratch staging marker so next GPU job re-stages
    if STAGING_MARKER.exists():
        STAGING_MARKER.unlink()
        log.info("invalidated_staging_marker", path=str(STAGING_MARKER))


main = rebuild_caches
