"""Parse per-run OpenTelemetry artifacts into polars DataFrames.

``{run_dir}/metrics.jsonl`` and ``{run_dir}/traces.jsonl`` are NDJSON —
one OTel record per line — written by ``core/monitoring.py`` via
``_otel.wire_file_exporters``. OTel has no reader API for these; this
module flattens the nested per-record schema into tabular form.

Public API:
    load_metrics(config)   -> polars.DataFrame
    load_traces(config)    -> polars.DataFrame
    run_dir_from_config(c) -> pathlib.Path

``config`` is any of: a rendered jsonnet dict (reads
``trainer.default_root_dir``), a str/Path pointing to the run_dir, or a
str/Path pointing to the metrics/traces file directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

_METRIC_SCHEMA = {
    "metric": pl.Utf8,
    "ts_ns": pl.Int64,
    "value": pl.Float64,
    "dp_kind": pl.Utf8,
    "count": pl.Int64,
    "attrs_json": pl.Utf8,
}
_TRACE_SCHEMA = {
    "name": pl.Utf8,
    "status_code": pl.Utf8,
    "start_ts_ns": pl.Int64,
    "end_ts_ns": pl.Int64,
    "attrs_json": pl.Utf8,
}


def run_dir_from_config(config: Mapping[str, Any] | str | Path) -> Path:
    """Coerce a config/path into a ``run_dir`` Path.

    - ``dict`` with ``trainer.default_root_dir`` (rendered jsonnet).
    - ``str`` / ``Path`` naming a directory (returned as-is).
    - ``str`` / ``Path`` naming ``metrics.jsonl`` or ``traces.jsonl``
      (parent returned).
    """
    if isinstance(config, Mapping):
        root = (config.get("trainer") or {}).get("default_root_dir")
        if not root:
            raise ValueError("rendered config has no trainer.default_root_dir")
        return Path(root)
    p = Path(config)
    if p.is_dir():
        return p
    if p.name in ("metrics.jsonl", "traces.jsonl"):
        return p.parent
    raise ValueError(f"cannot derive run_dir from {config!r}")


def _file_for(config, basename: str) -> Path:
    if isinstance(config, (str, Path)) and Path(config).name == basename:
        return Path(config)
    return run_dir_from_config(config) / basename


def _point_value(dp: dict[str, Any]) -> tuple[str, float | None, int | None]:
    """Extract (kind, value, count) from an OTel data point."""
    if "bucket_counts" in dp:
        s = dp.get("sum")
        return "histogram", float(s) if s is not None else None, dp.get("count")
    for key in ("value", "as_double", "as_int"):
        if dp.get(key) is not None:
            return "gauge", float(dp[key]), None
    s = dp.get("sum")
    if s is not None:
        return "sum", float(s), dp.get("count")
    return "unknown", None, None


def load_metrics(config: Mapping[str, Any] | str | Path) -> pl.DataFrame:
    """Return one row per (metric, data_point) from ``metrics.jsonl``.

    Columns: ``metric``, ``ts_ns``, ``value``, ``dp_kind``, ``count``,
    ``attrs_json``. Empty DataFrame with correct schema if the file is
    missing or contains no data points yet.
    """
    path = _file_for(config, "metrics.jsonl")
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=_METRIC_SCHEMA)

    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rm in rec.get("resource_metrics", []):
            for sm in rm.get("scope_metrics", []):
                for m in sm.get("metrics", []):
                    name = m.get("name")
                    for dp in (m.get("data") or {}).get("data_points", []):
                        kind, value, count = _point_value(dp)
                        rows.append(
                            {
                                "metric": name,
                                "ts_ns": dp.get("time_unix_nano"),
                                "value": value,
                                "dp_kind": kind,
                                "count": count,
                                "attrs_json": json.dumps(
                                    dp.get("attributes") or {}, sort_keys=True
                                ),
                            }
                        )

    if not rows:
        return pl.DataFrame(schema=_METRIC_SCHEMA)
    return pl.DataFrame(rows, schema=_METRIC_SCHEMA)


def load_traces(config: Mapping[str, Any] | str | Path) -> pl.DataFrame:
    """Return one row per span from ``traces.jsonl``.

    Columns: ``name``, ``status_code``, ``start_ts_ns``, ``end_ts_ns``,
    ``attrs_json``. Empty DataFrame if no spans have closed yet.
    """
    path = _file_for(config, "traces.jsonl")
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=_TRACE_SCHEMA)

    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            span = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "name": span.get("name"),
                "status_code": (span.get("status") or {}).get("status_code"),
                "start_ts_ns": span.get("start_time_unix_nano") or span.get("start_time"),
                "end_ts_ns": span.get("end_time_unix_nano") or span.get("end_time"),
                "attrs_json": json.dumps(span.get("attributes") or {}, sort_keys=True),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_TRACE_SCHEMA)
    df = pl.DataFrame(rows)
    for col in ("start_ts_ns", "end_ts_ns"):
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
    return df.select(list(_TRACE_SCHEMA.keys()))
