"""WaDi (Water Distribution) dataset adapter — skeleton.

iTrust SUTD water distribution testbed. Wide-format CSV with ~127
sensor/actuator columns at 1-second intervals; 14 days of normal
operation + a separate attack CSV.

Sensor naming convention (from SCADA tags):
  ``{stage}_{type}_{number}_{suffix}``
  e.g. ``1_AIT_001_PV`` → stage 1, Analog Input Transmitter #001, Process Value

Sensor types: AIT (analog), FIT (flow), LT (level), MV (valve), P (pump),
DPIT (differential pressure), FIC (flow controller), LS (level switch).

Status: skeleton. Implementation requires:
  1. Defining the per-domain :class:`GraphSchema` (node/edge Polars
     exprs, column orders, attack taxonomy from
     ``attack_description.xlsx``).
  2. Implementing :meth:`WaDiDataset._read_raw` — scan CSV (skip 4-line
     header), strip the SCADA path prefix from column names via
     :func:`_clean_column_names`, parse timestamp from Date+Time, attach
     the attack column, melt wide → long via :func:`_wide_to_long`.
  3. Implementing :meth:`WaDiSource._scan_vocab` — sorted unique
     ``sensor_name`` across every CSV the catalog declares.

Once those three are in place the base classes (``BaseGraphDataset`` /
``BaseGraphSource`` in ``_base.py``) handle splits, scaler fit/apply,
metadata merge, mmap load, and per-test-subdir tensor layout. WaDi's
single-CSV-plus-attack-CSV split layout doesn't fit ``BaseGraphSource``'s
catalog-subdir-grid assumption — :meth:`WaDiSource.build` will likely
need to override the parent.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from graphids.core.data.datasets._base import (
    BaseGraphDataset,
    BaseGraphSource,
    GraphSchema,
)


def _clean_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Strip SCADA path prefix from WaDi column names.

    Raw:   ``\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_001_PV``
    Clean: ``1_AIT_001_PV``
    """
    rename_map = {}
    for name in lf.collect_schema().names():
        if "SUTD_WADI" in name:
            rename_map[name] = name.rsplit("\\", 1)[-1]
    return lf.rename(rename_map) if rename_map else lf


def _wide_to_long(lf: pl.LazyFrame, sensor_cols: list[str]) -> pl.LazyFrame:
    """Melt wide sensor DataFrame to long format for the graph pipeline.

    Wide:  [timestamp, sensor_1, sensor_2, ..., attack]
    Long:  [timestamp, sensor_name, value, attack]
    """
    return lf.unpivot(
        index=["timestamp", "attack"],
        on=sensor_cols,
        variable_name="sensor_name",
        value_name="value",
    )


# TODO: populate when the WaDi schema is finalized.
WADI_SCHEMA: GraphSchema | None = None


class WaDiDataset(BaseGraphDataset):
    """WaDi water distribution graph dataset (skeleton — not yet implemented)."""

    SCHEMA: ClassVar[GraphSchema]  # set when WADI_SCHEMA is populated

    def _read_raw(self) -> pl.DataFrame:
        raise NotImplementedError(
            "WaDiDataset._read_raw is not yet implemented. See module docstring."
        )


class WaDiSource(BaseGraphSource):
    """WaDi source (skeleton — not yet implemented).

    NOTE: WaDi splits are CSV-pair shaped (one big normal CSV + one
    attack CSV), not the catalog-subdir grid that
    :meth:`BaseGraphSource.build` assumes. Override ``build()`` rather
    than relying on the parent.
    """

    KIND: ClassVar[str] = "wadi"
    DATASET_CLS: ClassVar[type[BaseGraphDataset]] = WaDiDataset

    def _scan_vocab(self, raw_dir, source_dirs):
        raise NotImplementedError(
            "WaDiSource._scan_vocab is not yet implemented. See module docstring."
        )
