"""Rebuild preprocessed graph caches after preprocessing changes.

Operation layer — argparse surface lives in ``graphids.commands.rebuild_caches``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from graphids.config.constants import LAKE_ROOT, PREPROCESSING_VERSION
from graphids.config.paths import cache_dir
from graphids.log import get_logger

log = get_logger(__name__)


def rebuild_caches(datasets: list[str], *, delete_existing: bool = False) -> None:
    """Rebuild preprocessed graph caches for each dataset in ``datasets``.

    When ``delete_existing`` is true, stale cache directories are removed
    before the datamodule rebuilds them. Invalidates the scratch staging
    marker on completion so the next GPU job re-stages from the refreshed
    NFS cache.
    """
    for ds in datasets:
        cdir = cache_dir(LAKE_ROOT, ds)
        if delete_existing and cdir.exists():
            log.info("removing_stale_cache", dataset=ds, path=str(cdir))
            shutil.rmtree(cdir)

        log.info("rebuilding_cache", dataset=ds, version=PREPROCESSING_VERSION)
        from graphids.core.data.datamodule.graph import GraphDataModule

        dm = GraphDataModule(dataset=ds, lake_root=LAKE_ROOT)
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
