"""WaDi (Water Distribution) dataset adapter.

iTrust SUTD water distribution testbed. Wide-format CSV with ~127 sensor/actuator
columns at 1-second intervals. Normal data (14 days) + attack data (separate CSV).

Sensor naming convention (from SCADA tags):
  {stage}_{type}_{number}_{suffix}
  e.g. 1_AIT_001_PV  ->  stage 1, Analog Input Transmitter #001, Process Value
       2_MV_001_STATUS -> stage 2, Motorized Valve #001, Status

Sensor types: AIT (analog), FIT (flow), LT (level), MV (valve), P (pump),
              DPIT (differential pressure), FIC (flow controller), LS (level switch)

Raw data location: see dataset_registry.json and data/README.md.
See datasets/README.md for the adapter pattern.
"""

from __future__ import annotations

import polars as pl

from structlog import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TODO: Sensor registry — which columns to keep, how to group into nodes
# ---------------------------------------------------------------------------

# Full SCADA prefix stripped from raw CSV headers. Raw columns look like:
#   \\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV
# After cleaning they become: 1_AIT_001_PV
#
# Decide which sensors to include. Options:
#   - All sensors (full graph, ~127 nodes)
#   - Filter by type (e.g. only analog inputs + flow transmitters)
#   - Filter by stage (e.g. stage 1 = raw water supply, stage 2 = treatment)
#
# SENSOR_COLS: list[str] = [...]  # TODO: populate after inspecting cleaned headers


# ---------------------------------------------------------------------------
# TODO: Attack-type taxonomy
# ---------------------------------------------------------------------------

# WaDi attacks are described in attack_description.xlsx inside each zip.
# Map attack scenario names/IDs to integer codes.
#
# ATTACK_TYPE_CODES: dict[str, int] = {
#     "normal": 0,
#     ...  # TODO: from attack_description.xlsx
# }


# ---------------------------------------------------------------------------
# TODO: Feature schema — column layouts
# ---------------------------------------------------------------------------

# NODE_COL_ORDER defines the tensor layout (x.shape[1]).
# For ICS sensors, typical per-node features within a window:
#   - value_mean, value_std, value_range (basic stats of the sensor reading)
#   - value_trend (slope of linear fit or last - first)
#   - value_min, value_max
#   - clustering_coeff, in_degree, out_degree (filled by pipeline)
#
# NODE_COL_ORDER: list[str] = [
#     "value_mean", "value_std", "value_range",
#     ...
#     "clustering_coeff",  # placeholder — filled by pipeline
#     "in_degree",         # placeholder — filled by pipeline
#     "out_degree",        # placeholder — filled by pipeline
# ]
#
# N_NODE_FEATURES = len(NODE_COL_ORDER)

# EDGE_COL_ORDER defines edge feature layout (edge_attr.shape[1]).
# For ICS, edges might carry: inter-arrival time, cross-correlation, etc.
#
# EDGE_COL_ORDER: tuple[str, ...] = ("iat", "bidir", "edge_freq")
# N_EDGE_FEATURES = len(EDGE_COL_ORDER)


# ---------------------------------------------------------------------------
# TODO: Polars expressions
# ---------------------------------------------------------------------------

# NODE_STAT_EXPRS: list[pl.Expr] = [
#     pl.col("value").mean().alias("value_mean"),
#     pl.col("value").std().alias("value_std"),
#     (pl.col("value").max() - pl.col("value").min()).alias("value_range"),
#     ...
#     pl.lit(0.0).alias("clustering_coeff"),  # filled by pipeline
#     pl.lit(0.0).alias("in_degree"),          # filled by pipeline
#     pl.lit(0.0).alias("out_degree"),         # filled by pipeline
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
# EDGE_BASE_COLS: list[str] = []  # extra columns needed for edge computation


# ---------------------------------------------------------------------------
# TODO: Raw data I/O
# ---------------------------------------------------------------------------


def _clean_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Strip SCADA path prefix from WaDi column names.

    Raw:  \\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_001_PV
    Clean: 1_AIT_001_PV
    """
    rename_map = {}
    for name in lf.collect_schema().names():
        if "SUTD_WADI" in name:
            clean = name.rsplit("\\", 1)[-1]
            rename_map[name] = clean
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


# ---------------------------------------------------------------------------
# TODO: WaDiDataset — InMemoryDataset adapter
# ---------------------------------------------------------------------------

# class WaDiDataset(InMemoryDataset):
#     """WaDi water distribution graph dataset.
#
#     Each graph is one sliding window of sensor readings. Nodes are
#     sensors/actuators, edges are temporal adjacency.
#     """
#
#     def __init__(self, root, raw_dir, split="train", ...):
#         ...
#
#     def _read_raw(self) -> pl.DataFrame:
#         # 1. scan CSV (skip 4-line header), clean column names
#         # 2. parse timestamp from Date + Time columns
#         # 3. select sensor columns + attack label
#         # 4. melt wide -> long via _wide_to_long()
#         # 5. build vocab: sensor_name -> node_id via vocab_from_column()
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
#         data, slices, num_graphs = GraphPipeline(...).run(
#             df, self.window_size, self.stride,
#             node_stat_exprs=NODE_STAT_EXPRS,
#             edge_stat_exprs=EDGE_STAT_EXPRS,
#             node_col_order=NODE_COL_ORDER,
#             edge_col_order=EDGE_COL_ORDER,
#             label_exprs=LABEL_EXPRS,
#             edge_base_cols=EDGE_BASE_COLS,
#         )
#         return data, slices, num_sensors, num_graphs
