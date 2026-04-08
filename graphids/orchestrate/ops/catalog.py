"""Rebuild DuckDB experiment catalog from run_record.json sidecars.

Scans ``{lake_root}/dev/`` for sidecars and ingests via DuckDB
``read_json_auto()``. The ``runs`` table includes a computed
``asset_name`` column (``stage || identity_hash || kd_tag``) for
exact-match joins with planner topology in ``pipeline-status``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from graphids.config.constants import CATALOG_SUBPATH, RUN_RECORD_FILENAME
from graphids.config.topology import catalog_path
from graphids.log import get_logger

log = get_logger(__name__)


def rebuild_catalog(
    *,
    lake_root: str,
    dry_run: bool = False,
) -> None:
    """Ingest all run_record.json sidecars into the DuckDB catalog."""
    cat_path = catalog_path(lake_root)
    glob_pattern = str(Path(lake_root) / "dev" / "**" / RUN_RECORD_FILENAME)

    sidecars = list(Path(lake_root).glob(f"dev/**/{RUN_RECORD_FILENAME}"))
    if not sidecars:
        log.info("no_sidecars_found", lake_root=lake_root)
        print("No sidecars found.")
        return

    if dry_run:
        print(f"Would ingest {len(sidecars)} sidecar(s)")
        return

    from graphids.config.settings import require_lake_write

    require_lake_write()
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(cat_path))

    try:
        db.execute(
            """
            CREATE OR REPLACE TABLE runs AS
            SELECT
                *,
                stage || identity_hash || COALESCE(kd_tag, '') AS asset_name,
                current_timestamp AS catalog_updated_at
            FROM read_json_auto(?, maximum_object_size=1048576, union_by_name=true)
        """,
            [glob_pattern],
        )

        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        summary = db.execute("""
            SELECT status, COUNT(*) AS n
            FROM runs GROUP BY status ORDER BY n DESC
        """).fetchall()

        log.info(
            "catalog_rebuilt",
            total=count,
            breakdown={s: n for s, n in summary},
            catalog_path=str(cat_path),
        )
        print(f"Catalog rebuilt: {count} run(s) in {lake_root}/{CATALOG_SUBPATH}")
    finally:
        db.close()
