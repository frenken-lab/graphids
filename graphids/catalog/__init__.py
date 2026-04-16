"""Cross-run catalog backed by a single DuckDB file on the lake.

One file: ``{lake_root}/catalog/graphids.duckdb``. Two tables:

- ``runs``    — one row per run (identity + metadata + timestamps)
- ``metrics`` — final scalar metrics in long format (MLflow-style shadow
                for leaderboard queries; timeseries history stays in
                per-run ``metrics.jsonl``).

Concurrency: DuckDB is single-writer. ``{lake_root}`` is GPFS (not NFS),
which implements POSIX fcntl locking, so concurrent training jobs
serialize cleanly on the exclusive write lock. ``record_run`` uses a
short retry loop for transient contention.

Non-ablation run_dirs (bare ``stages/*.jsonnet`` smokes) return ``None``
from ``parse_run_dir`` — ``record_run`` short-circuits, no silent row.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from graphids.catalog.paths import RunIdentity, parse_run_dir, run_id_for
from graphids.config.settings import get_settings

__all__ = ["Catalog", "RunIdentity", "parse_run_dir", "run_id_for"]

CATALOG_SUBPATH = "catalog/graphids.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     VARCHAR PRIMARY KEY,
    run_dir    VARCHAR NOT NULL,
    "group"    VARCHAR NOT NULL,
    variant    VARCHAR NOT NULL,
    dataset    VARCHAR NOT NULL,
    seed       INTEGER NOT NULL,
    cluster    VARCHAR,
    git_sha    VARCHAR NOT NULL,
    status     VARCHAR NOT NULL,
    started_at BIGINT  NOT NULL,
    ended_at   BIGINT  NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id VARCHAR NOT NULL,
    key    VARCHAR NOT NULL,
    value  DOUBLE  NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE INDEX IF NOT EXISTS idx_runs_group   ON runs("group");
CREATE INDEX IF NOT EXISTS idx_runs_variant ON runs(variant);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

# Tuned for "20 concurrent SLURM jobs all calling record_run() at once":
# each insert is ~ms, so even serialized they clear in well under a second.
# A short retry with jitter is enough for the rare collision.
_RETRY_DELAYS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


class Catalog:
    """Cross-run metadata store — one DuckDB file per lake.

    Instances are cheap (no connection held); a fresh connection is opened
    per call and closed immediately. Reads use ``read_only=True`` so they
    don't contend with writers.
    """

    def __init__(self, lake_root: str | Path):
        self.db_path = Path(lake_root) / CATALOG_SUBPATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)

    @contextmanager
    def _connect(self, *, read_only: bool = False):
        """Open DuckDB with retry on ``IOException`` (exclusive lock held elsewhere)."""
        last_exc: Exception | None = None
        for delay in (0.0, *_RETRY_DELAYS_S):
            if delay:
                time.sleep(delay)
            try:
                con = duckdb.connect(str(self.db_path), read_only=read_only)
                break
            except duckdb.IOException as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"could not acquire DuckDB lock on {self.db_path} after "
                f"{len(_RETRY_DELAYS_S)} retries"
            ) from last_exc
        try:
            yield con
        finally:
            con.close()

    # ---- write path -------------------------------------------------------

    def record_run(
        self,
        run_dir: Path,
        *,
        metrics: dict[str, Any],
        git_sha: str,
        status: str,
        started_at_ns: int,
        ended_at_ns: int,
    ) -> str | None:
        """Upsert a run + its final metrics in one transaction.

        Returns the run_id on success, ``None`` if ``run_dir`` didn't
        match the ablation shape (parser returned None). Re-running the
        same ``(group, variant, dataset, seed, cluster)`` replaces the
        prior row — stale metric keys from the previous run are cleared
        first so orphans can't accumulate.
        """
        identity = parse_run_dir(run_dir)
        if identity is None:
            return None
        cluster = get_settings().cluster or None
        run_id = run_id_for(identity, cluster=cluster)
        metric_rows = list(_iter_metric_rows(metrics, run_id))

        with self._connect() as con:
            con.begin()
            try:
                con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
                con.execute("DELETE FROM metrics WHERE run_id = ?", [run_id])
                con.execute(
                    """
                    INSERT INTO runs
                    (run_id, run_dir, "group", variant, dataset, seed, cluster,
                     git_sha, status, started_at, ended_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id,
                        str(run_dir),
                        identity.group,
                        identity.variant,
                        identity.dataset,
                        identity.seed,
                        cluster,
                        git_sha,
                        status,
                        started_at_ns,
                        ended_at_ns,
                    ],
                )
                if metric_rows:
                    con.executemany(
                        "INSERT INTO metrics (run_id, key, value) VALUES (?, ?, ?)",
                        metric_rows,
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return run_id

    # ---- read path --------------------------------------------------------

    def query_runs(
        self,
        *,
        group: str | None = None,
        variant: str | None = None,
        dataset: str | None = None,
        seed: int | None = None,
        since_ns: int | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Filtered scan of the runs table (parameters bind, no injection)."""
        clauses: list[str] = []
        params: list[Any] = []
        if group is not None:
            clauses.append('"group" = ?')
            params.append(group)
        if variant is not None:
            clauses.append("variant = ?")
            params.append(variant)
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if seed is not None:
            clauses.append("seed = ?")
            params.append(int(seed))
        if since_ns is not None:
            clauses.append("started_at >= ?")
            params.append(int(since_ns))
        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._connect(read_only=True) as con:
            return con.execute(sql, params).pl()

    def query_metrics(
        self,
        *,
        key: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        sql = "SELECT * FROM metrics"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._connect(read_only=True) as con:
            return con.execute(sql, params).pl()

    def existing_run_ids(self) -> set[str]:
        """Fast existence set for the rebuild CLI's diff."""
        with self._connect(read_only=True) as con:
            return set(con.execute("SELECT run_id FROM runs").df()["run_id"])


def _iter_metric_rows(metrics: dict[str, Any], run_id: str):
    """Flatten the trainer's metrics dict into ``(run_id, key, value)`` rows.

    Accepts flat ``{"test_auroc": 0.9}`` and one-deep nested
    ``{"test_01": {"auroc": 0.9}}`` (per-test-set shape). Non-numeric
    values are skipped — catalog metrics are scalars by contract.
    """
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    yield (run_id, f"{key}/{sub_key}", float(sub_value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield (run_id, key, float(value))
