"""EPIC (Electric Power and Intelligent Control) dataset adapter.

iTrust SUTD power grid testbed. CSV sensor + actuator data across 8 normal
operation scenarios (30 min each). Network traffic in pcap files (not used here).

Scenarios:
  1. Sync without load (G1-G2 angle sweep)
  2. Sync with 10kW resistive load
  3. G1 & G2 running, 10kW load
  4. G1 & G2 + PV system, 10kW load
  5. G1 & G2 + PV system, 7kW load
  6. G1-G3 running, 14kW load
  7. G1 & G2 powering SWaT testbed
  8. G1 & G2 powering SWaT + WaDi testbeds

Raw data location: see dataset_registry.json and data/README.md.
See datasets/README.md for the adapter pattern.
"""

from __future__ import annotations

import polars as pl

from graphids.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TODO: Sensor registry
# ---------------------------------------------------------------------------

# EPIC CSV columns are sensor/actuator readings from the power grid testbed.
# Inspect the extracted CSVs to identify column names and types.
# Generators (G1, G2, G3), PV system, loads, breakers, etc.
#
# SENSOR_COLS: list[str] = [...]  # TODO: populate after inspecting CSVs


# ---------------------------------------------------------------------------
# TODO: Attack-type taxonomy
# ---------------------------------------------------------------------------

# EPIC normal scenarios (1-8) have no attacks. The Oct 2021 dataset may
# contain attack scenarios. Define codes once attack data is inspected.
#
# ATTACK_TYPE_CODES: dict[str, int] = {
#     "normal": 0,
#     ...  # TODO: from Oct 2021 dataset description
# }


# ---------------------------------------------------------------------------
# TODO: Feature schema — column layouts
# ---------------------------------------------------------------------------

# Power grid nodes might be: generators, buses, loads, breakers, PV inverter.
# Per-node features within a window:
#   - voltage_mean, voltage_std, voltage_range
#   - current_mean, current_std
#   - power_mean, frequency_mean
#   - clustering_coeff, in_degree, out_degree (filled by pipeline)
#
# NODE_COL_ORDER: list[str] = [...]
# N_NODE_FEATURES = len(NODE_COL_ORDER)
#
# EDGE_COL_ORDER: tuple[str, ...] = ("iat", "bidir", "edge_freq")
# N_EDGE_FEATURES = len(EDGE_COL_ORDER)


# ---------------------------------------------------------------------------
# TODO: Polars expressions
# ---------------------------------------------------------------------------

# NODE_STAT_EXPRS: list[pl.Expr] = [
#     pl.col("value").mean().alias("voltage_mean"),
#     ...
#     pl.lit(0.0).alias("clustering_coeff"),
#     pl.lit(0.0).alias("in_degree"),
#     pl.lit(0.0).alias("out_degree"),
# ]
#
# EDGE_STAT_EXPRS: list[pl.Expr] = [
#     pl.col("timestamp").diff().over("_wid").cast(pl.Float32).alias("iat"),
# ]
#
# LABEL_EXPRS: list[pl.Expr] = [
#     (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
# ]
#
# EDGE_BASE_COLS: list[str] = []


# ---------------------------------------------------------------------------
# TODO: Raw data I/O
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TODO: EPICDataset — InMemoryDataset adapter
# ---------------------------------------------------------------------------

# class EPICDataset(InMemoryDataset):
#     """EPIC power grid graph dataset.
#
#     Each graph is one sliding window of sensor readings. Nodes are
#     generators/buses/loads/breakers, edges are temporal adjacency.
#     """
#
#     def __init__(self, root, raw_dir, split="train", ...):
#         ...
#
#     def _read_raw(self) -> pl.DataFrame:
#         # 1. scan CSV(s) for the selected scenario(s)
#         # 2. parse/normalize timestamp column
#         # 3. add attack label column (0 for normal scenarios)
#         # 4. select sensor columns
#         # 5. melt wide -> long via _wide_to_long()
#         # 6. build vocab: sensor_name -> node_id
#         ...
#
#     def _build_graphs(self) -> tuple[Data, dict, int, int]:
#         df = self._read_raw()
#         vocab, oov = vocab_from_column(df["sensor_name"])
#         num_sensors = len(vocab) + 1
#         df = df.with_columns(
#             pl.col("sensor_name")
#             .replace_strict(vocab, default=oov)
#             .cast(pl.Int64)
#             .alias("node_id")
#         )
#         data, slices, num_graphs = sliding_window_graphs(
#             df, self.window_size, self.stride,
#             node_stat_exprs=NODE_STAT_EXPRS,
#             edge_stat_exprs=EDGE_STAT_EXPRS,
#             node_col_order=NODE_COL_ORDER,
#             edge_col_order=EDGE_COL_ORDER,
#             label_exprs=LABEL_EXPRS,
#             edge_base_cols=EDGE_BASE_COLS,
#         )
#         return data, slices, num_sensors, num_graphs
