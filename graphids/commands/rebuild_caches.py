"""Rebuild preprocessed graph caches after preprocessing changes.

Usage:
    python -m graphids rebuild-caches --dataset hcrl_ch
    python -m graphids rebuild-caches --dataset hcrl_ch hcrl_sa --delete-existing
    python -m graphids rebuild-caches --all
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from graphids.log import get_logger

from graphids.config import LAKE_ROOT, PREPROCESSING_VERSION, cache_dir, dataset_names

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    all_datasets = dataset_names()
    parser = argparse.ArgumentParser(description="Rebuild preprocessed graph caches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", nargs="+", choices=all_datasets)
    group.add_argument("--all", action="store_true", dest="all_datasets")
    parser.add_argument("--delete-existing", action="store_true", help="Remove stale cache before rebuild")
    args = parser.parse_args(argv)

    datasets = all_datasets if args.all_datasets else args.dataset

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
    scratch = os.environ.get("KD_GAT_SCRATCH")
    if scratch:
        marker = Path(scratch) / "kd-gat-data" / "cache" / ".staged_marker"
        if marker.exists():
            marker.unlink()
            log.info("invalidated_staging_marker", path=str(marker))
