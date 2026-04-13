"""Dataset-agnostic sliding-window → graph pipeline.

Converts a timestamped message DataFrame into a collection of PyG ``Data``
graphs, one per sliding window. Domain-specific adapters (e.g.
``datasets/can_bus.py``) supply their own Polars expressions and column
layouts; this class handles windowing, graph construction, and tensor
packing with no knowledge of the underlying protocol.

Pipeline steps (methods on ``GraphPipeline``):

1. ``_aggregate`` — ``group_by_dynamic`` for node stats, edge adjacency,
   and per-window labels in a single scan of the data.
2. ``_add_bidir_flag`` — mark edges whose reverse also exists.
3. ``_compute_graph_structure`` — clustering coefficient (triangle counting)
   and in/out degree via Polars joins + ``DataFrame.update``.
4. ``_map_labels`` — per-window ``y`` (binary) and auxiliary labels.
5. ``_build_pyg_data`` — local ID assignment via ``cum_count().over()``,
   Polars pre-sort, bulk ``to_torch``, and ``torch.cumsum`` for slices.
"""

from __future__ import annotations

import polars as pl
import torch
from torch_geometric.data import Data

from graphids._otel import get_logger

log = get_logger(__name__)


class GraphPipeline:
    """Dataset-agnostic sliding-window → graph pipeline.

    Parameters describing the dataset schema:
        ``node_stat_exprs``: Polars aggregations for per-node features.
        ``edge_stat_exprs``: Polars expressions for edge features.
            Must NOT include ``.over()`` — the window context is provided
            by ``group_by_dynamic``.
        ``node_col_order``: final column order for the node feature tensor.
        ``edge_col_order``: final column order for the edge feature tensor.
        ``label_exprs``: per-window aggregations yielding label columns. The
            first expression must be aliased ``y``.
        ``edge_base_cols``: extra columns required for edge feature
            computation (e.g. byte_0..7 for CAN byte diffs).
    """

    def __init__(
        self,
        *,
        node_stat_exprs: list[pl.Expr],
        edge_stat_exprs: list[pl.Expr],
        node_col_order: list[str],
        edge_col_order: tuple[str, ...],
        label_exprs: list[pl.Expr],
        edge_base_cols: list[str],
    ):
        self.node_stat_exprs = node_stat_exprs
        self.edge_stat_exprs = edge_stat_exprs
        self.node_col_order = node_col_order
        self.edge_col_order = edge_col_order
        self.label_exprs = label_exprs
        self.edge_base_cols = edge_base_cols
        self.label_names = [e.meta.output_name() for e in label_exprs]
        assert self.label_names[0] == "y", (
            f"first label expr must be aliased 'y', got {self.label_names[0]!r}"
        )

    def run(self, df: pl.DataFrame, window_size: int, stride: int) -> tuple[Data, dict, int]:
        """Execute the full pipeline. Returns (Data, slices, num_graphs)."""
        df = df.with_row_index("_row").with_columns(pl.col("_row").cast(pl.Int64))
        n_rows = len(df)
        n_windows = max(0, (n_rows - window_size) // stride + 1)
        if n_windows == 0:
            log.warning("no_complete_windows", n_rows=n_rows, window_size=window_size)
            return Data(), {}, 0

        log.info("windowing", n_windows=n_windows, window_size=window_size, stride=stride)
        half = window_size // 2
        every, period = f"{stride}i", f"{window_size}i"
        max_wid = (n_windows - 1) * stride

        # _first_half for split_half_ratio (exact when stride >= window_size)
        df = df.with_columns((pl.col("_row") % window_size < half).alias("_first_half"))
        lf = df.lazy().sort("_row")

        node_stats, edge_df, labels = self._aggregate(lf, every, period)
        del df, lf

        # Drop incomplete trailing windows
        node_stats = node_stats.filter(pl.col("_wid") <= max_wid)
        edge_df = edge_df.filter(pl.col("_wid") <= max_wid)
        labels = labels.filter(pl.col("_wid") <= max_wid)

        edge_df = self._add_bidir_flag(edge_df)
        node_stats = self._compute_graph_structure(node_stats, edge_df)
        label_maps = self._map_labels(labels)

        return self._build_pyg_data(node_stats, edge_df, label_maps)

    # -- Step 1: Aggregate via group_by_dynamic --------------------------------

    def _aggregate(
        self, lf: pl.LazyFrame, every: str, period: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Node stats, edges, and labels via ``group_by_dynamic``."""
        dyn = dict(every=every, period=period, closed="left")

        node_stats = (
            lf.group_by_dynamic("_row", **dyn, group_by="node_id")
            .agg(*self.node_stat_exprs)
            .fill_null(0).fill_nan(0)
            .rename({"_row": "_wid"})
            .collect()
        )

        labels = (
            lf.group_by_dynamic("_row", **dyn)
            .agg(*self.label_exprs)
            .rename({"_row": "_wid"})
            .collect()
        )

        # Edges: shift-1 temporal adjacency within each window
        edge_agg = [
            pl.col("node_id").alias("src"),
            pl.col("node_id").shift(-1).alias("dst"),
            *self.edge_stat_exprs,
        ]
        edge_list_cols = ["src", "dst"] + [e.meta.output_name() for e in self.edge_stat_exprs]
        edge_df = (
            lf.select("_row", "node_id", "timestamp", *self.edge_base_cols)
            .group_by_dynamic("_row", **dyn)
            .agg(*edge_agg)
            .rename({"_row": "_wid"})
            .explode(edge_list_cols)
            .filter(pl.col("dst").is_not_null() & pl.col("iat").is_not_null())
            .with_columns(
                pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq"),
            )
            .collect()
        )

        log.info("features_computed", stat_rows=len(node_stats), edge_rows=len(edge_df))
        return node_stats, edge_df, labels

    # -- Step 2: Bidirectional edge flag ---------------------------------------

    @staticmethod
    def _add_bidir_flag(edge_df: pl.DataFrame) -> pl.DataFrame:
        """bidir=1.0 if the reverse edge exists in the same window."""
        edge_pairs = edge_df.select("_wid", "src", "dst").unique()
        return (
            edge_df.join(
                edge_pairs.with_columns(pl.lit(True).alias("_rev")),
                left_on=["_wid", "dst", "src"],
                right_on=["_wid", "src", "dst"],
                how="left",
            )
            .with_columns(pl.col("_rev").fill_null(False).cast(pl.Float32).alias("bidir"))
            .drop("_rev")
        )

    # -- Step 3: Graph structure (clustering + degree) -------------------------

    @staticmethod
    def _compute_graph_structure(
        node_stats: pl.DataFrame, edge_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute clustering coefficient + in/out degree, update placeholders."""
        in_deg = (
            edge_df.group_by(["_wid", "dst"])
            .agg(pl.len().cast(pl.Float32).alias("in_degree"))
            .rename({"dst": "node_id"})
        )
        out_deg = (
            edge_df.group_by(["_wid", "src"])
            .agg(pl.len().cast(pl.Float32).alias("out_degree"))
            .rename({"src": "node_id"})
        )

        # Triangle counting on undirected edges
        edge_pairs = pl.concat([
            edge_df.select("_wid", pl.col("src").alias("u"), pl.col("dst").alias("v")),
            edge_df.select("_wid", pl.col("dst").alias("u"), pl.col("src").alias("v")),
        ]).unique(["_wid", "u", "v"])

        two_paths = edge_pairs.join(
            edge_pairs.select("_wid", pl.col("u").alias("_mid"), pl.col("v").alias("w")),
            left_on=["_wid", "v"], right_on=["_wid", "_mid"], how="inner",
        ).filter(pl.col("u") != pl.col("w"))

        tri = two_paths.join(
            edge_pairs, left_on=["_wid", "u", "w"],
            right_on=["_wid", "u", "v"], how="semi",
        )
        del two_paths

        tri_per_node = (
            tri.group_by(["_wid", "u"])
            .agg((pl.len() / 2).cast(pl.Float32).alias("_tri"))
            .rename({"u": "node_id"})
        )
        del tri

        undirected_deg = (
            edge_pairs.group_by(["_wid", "u"])
            .agg(pl.len().cast(pl.Float32).alias("_undeg"))
            .rename({"u": "node_id"})
        )
        del edge_pairs

        cc = (
            tri_per_node.join(undirected_deg, on=["_wid", "node_id"], how="right")
            .fill_null(0)
            .with_columns(
                pl.when(pl.col("_undeg") > 1)
                .then(2.0 * pl.col("_tri") / (pl.col("_undeg") * (pl.col("_undeg") - 1)))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias("clustering_coeff")
            )
            .select("_wid", "node_id", "clustering_coeff")
        )
        del tri_per_node, undirected_deg

        node_stats = (
            node_stats
            .update(cc, on=["_wid", "node_id"])
            .update(in_deg, on=["_wid", "node_id"])
            .update(out_deg, on=["_wid", "node_id"])
        )
        log.info("graph_structure_features_computed")
        return node_stats

    # -- Step 4: Label mapping -------------------------------------------------

    def _map_labels(self, labels: pl.DataFrame) -> dict[str, dict[int, int]]:
        """Build per-window label lookup dicts."""
        return {
            name: dict(zip(labels["_wid"].to_list(), labels[name].to_list()))
            for name in self.label_names
        }

    # -- Step 5: Build PyG Data ------------------------------------------------

    def _build_pyg_data(
        self,
        node_stats: pl.DataFrame,
        edge_df: pl.DataFrame,
        label_maps: dict[str, dict[int, int]],
    ) -> tuple[Data, dict, int]:
        """Local IDs, pre-sort by window size, convert to (Data, slices)."""
        # Keep only windows that have edges
        edge_wids = edge_df["_wid"].unique()
        node_stats = node_stats.filter(pl.col("_wid").is_in(edge_wids))

        if len(node_stats) == 0:
            log.warning("no_graphs_with_edges")
            return Data(), {}, 0

        # Pre-sort windows by node count (sequential page-faults for sampler)
        wid_sizes = node_stats.group_by("_wid").agg(pl.len().alias("_n"))
        node_stats = node_stats.join(wid_sizes, on="_wid").sort(["_n", "_wid"])
        edge_df = edge_df.join(wid_sizes, on="_wid").sort(["_n", "_wid"])

        # Local IDs per window via cum_count
        node_stats = node_stats.with_columns(
            (pl.cum_count("node_id").over("_wid") - 1).cast(pl.Int64).alias("_local_id")
        )
        id_map = node_stats.select("_wid", "node_id", "_local_id")
        edge_df = (
            edge_df
            .join(id_map.rename({"node_id": "src", "_local_id": "src_local"}),
                  on=["_wid", "src"], how="left")
            .join(id_map.rename({"node_id": "dst", "_local_id": "dst_local"}),
                  on=["_wid", "dst"], how="left")
        )

        # Polars → torch (data is already contiguous by window order)
        cat_x = (
            node_stats.select(self.node_col_order)
            .fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
        )
        cat_node_id = torch.from_numpy(
            node_stats["node_id"].cast(pl.Int64).to_numpy().copy()
        )
        cat_edge_index = torch.stack([
            torch.from_numpy(edge_df["src_local"].cast(pl.Int64).to_numpy().copy()),
            torch.from_numpy(edge_df["dst_local"].cast(pl.Int64).to_numpy().copy()),
        ])
        cat_edge_attr = (
            edge_df.select(list(self.edge_col_order))
            .fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
        )

        # Slice boundaries via Polars group_by + torch.cumsum
        kept_wids = node_stats.group_by("_wid", maintain_order=True).first()["_wid"].to_list()
        num_graphs = len(kept_wids)

        node_counts = torch.from_numpy(
            node_stats.group_by("_wid", maintain_order=True).len()["len"].to_numpy().copy()
        )
        node_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long), node_counts.cumsum(0).to(torch.long),
        ])

        edge_counts = torch.from_numpy(
            edge_df.group_by("_wid", maintain_order=True).len()["len"].to_numpy().copy()
        )
        edge_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long), edge_counts.cumsum(0).to(torch.long),
        ])

        graph_idx = torch.arange(num_graphs + 1, dtype=torch.long)
        label_tensors = {
            name: torch.tensor(
                [label_maps[name].get(w, 0) for w in kept_wids], dtype=torch.long,
            )
            for name in self.label_names
        }

        data = Data(
            x=cat_x, edge_index=cat_edge_index, edge_attr=cat_edge_attr,
            node_id=cat_node_id, **label_tensors,
        )
        slices = {
            "x": node_cumsum, "edge_index": edge_cumsum,
            "edge_attr": edge_cumsum, "node_id": node_cumsum,
            **{name: graph_idx for name in self.label_names},
        }
        log.info("graphs_built", count=num_graphs)
        return data, slices, num_graphs
