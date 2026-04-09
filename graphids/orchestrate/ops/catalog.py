"""Rebuild DuckDB experiment catalog from traces.jsonl span data.

Scans ``{lake_root}/dev/`` for ``traces.jsonl`` files produced by
OTelTrainingCallback and ingests ``training.fit`` spans via DuckDB
``read_json_auto()``. The ``runs`` table extracts identity, status,
and timing from OTel span attributes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from graphids.config.constants import CATALOG_SUBPATH
from graphids.config.topology import catalog_path
from graphids.log import get_logger

log = get_logger(__name__)

_TRACES_FILENAME = "traces.jsonl"


def rebuild_catalog(
    *,
    lake_root: str,
    dry_run: bool = False,
) -> None:
    """Ingest all traces.jsonl span data into the DuckDB catalog."""
    cat_path = catalog_path(lake_root)
    glob_pattern = str(Path(lake_root) / "dev" / "**" / _TRACES_FILENAME)

    traces_files = list(Path(lake_root).glob(f"dev/**/{_TRACES_FILENAME}"))
    if not traces_files:
        log.info("no_traces_found", lake_root=lake_root)
        print("No traces.jsonl files found.")
        return

    if dry_run:
        print(f"Would ingest from {len(traces_files)} traces.jsonl file(s)")
        return

    from graphids.config.settings import require_lake_write

    require_lake_write()
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(cat_path))

    try:
        # OTel ConsoleSpanExporter writes one JSON object per span.
        # Filter to training.fit spans which carry run identity as attributes.
        db.execute(
            """
            CREATE OR REPLACE TABLE runs AS
            SELECT
                json_extract_string(resource, '$.attributes."service.name"') AS service,
                json_extract_string(resource, '$.attributes."slurm.job_id"') AS slurm_job_id,
                name AS span_name,
                json_extract_string(status, '$.status_code') AS status_code,
                start_time,
                end_time,
                json_extract_string(attributes, '$."ml.run_dir"') AS run_dir,
                json_extract_string(attributes, '$."ml.model_class"') AS model_class,
                CAST(json_extract(attributes, '$."ml.max_epochs"') AS INTEGER) AS max_epochs,
                CAST(json_extract(attributes, '$."ml.epochs_run"') AS INTEGER) AS epochs_run,
                CAST(json_extract(attributes, '$."ml.metric.val_loss"') AS DOUBLE) AS val_loss,
                CAST(json_extract(attributes, '$."ml.metric.train_loss"') AS DOUBLE) AS train_loss,
                json_extract_string(attributes, '$."ml.checkpoint.best_path"') AS best_ckpt_path,
                json_extract_string(attributes, '$."ml.stage"') AS stage,
                json_extract_string(attributes, '$."ml.dataset"') AS dataset,
                json_extract_string(attributes, '$."ml.scale"') AS scale,
                CAST(json_extract(attributes, '$."ml.seed"') AS INTEGER) AS seed,
                json_extract_string(attributes, '$."ml.model_type"') AS model_type,
                links AS upstream_links,
                current_timestamp AS catalog_updated_at
            FROM read_json_auto(?, format='newline_delimited', maximum_object_size=1048576,
                                union_by_name=true)
            WHERE name = 'training.fit'
        """,
            [glob_pattern],
        )

        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        summary = db.execute("""
            SELECT status_code, COUNT(*) AS n
            FROM runs GROUP BY status_code ORDER BY n DESC
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
