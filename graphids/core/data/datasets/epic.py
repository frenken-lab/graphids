"""EPIC (Electric Power and Intelligent Control) dataset adapter — skeleton.

iTrust SUTD power grid testbed: sensor + actuator CSVs across 8 normal
operation scenarios (30 min each); the Oct 2021 release adds attack
scenarios. Network traffic in pcap files is not used here.

Status: skeleton. Implementation requires:
  1. Defining the per-domain :class:`GraphSchema` (node/edge Polars
     exprs, column orders, attack taxonomy if any).
  2. Implementing :meth:`EPICDataset._read_raw` — scan CSV(s) for the
     selected scenario(s), normalize the timestamp, attach an ``attack``
     column (0 for normal scenarios), melt wide → long so each row is
     ``(timestamp, sensor_name, value, attack)``.
  3. Implementing :meth:`EPICSource._scan_vocab` — sorted unique
     ``sensor_name`` across every CSV the catalog declares.

Once those three are in place the base classes (``BaseGraphDataset`` /
``BaseGraphSource`` in ``_base.py``) handle splits, scaler fit/apply,
metadata merge, mmap load, and per-test-subdir tensor layout — see
``can_bus.py`` for the reference subclass.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from graphids.core.data.datasets._base import (
    BaseGraphDataset,
    BaseGraphSource,
    GraphSchema,
)


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


# TODO: populate when the EPIC schema is finalized.
EPIC_SCHEMA: GraphSchema | None = None


class EPICDataset(BaseGraphDataset):
    """EPIC power grid graph dataset (skeleton — not yet implemented)."""

    SCHEMA: ClassVar[GraphSchema]  # set when EPIC_SCHEMA is populated

    def _read_raw(self) -> pl.DataFrame:
        raise NotImplementedError(
            "EPICDataset._read_raw is not yet implemented. See module docstring."
        )


class EPICSource(BaseGraphSource):
    """EPIC source (skeleton — not yet implemented)."""

    KIND: ClassVar[str] = "epic"
    DATASET_CLS: ClassVar[type[BaseGraphDataset]] = EPICDataset

    def _scan_vocab(self, raw_dir, source_dirs):
        raise NotImplementedError(
            "EPICSource._scan_vocab is not yet implemented. See module docstring."
        )
