"""KD-GAT Experiment Dashboard — unified Streamlit app for experiments + sweeps."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from data_loader import filter_df, get_hp_columns, load_experiments, load_sweeps

st.set_page_config(page_title="KD-GAT Dashboard", layout="wide", page_icon="🔬")

# --- Navigation ---
page = st.sidebar.radio(
    "Navigation",
    ["Experiments", "Sweeps"],
    index=0,
)

# =====================================================================
# EXPERIMENTS SECTION
# =====================================================================
if page == "Experiments":
    st.title("🔬 KD-GAT Experiments")

    experiments = load_experiments()

    if experiments.empty:
        st.warning(
            "No experiment data available yet. "
            "Run training with MLflow enabled, then push to HF Dataset."
        )
        st.stop()

    # --- Sidebar filters ---
    with st.sidebar:
        st.divider()
        st.subheader("Filters")

        if st.button("Reload Data", key="reload_exp"):
            st.cache_data.clear()
            st.rerun()

        filter_cols = {}
        for col in ["dataset", "model_type", "scale", "stage", "status"]:
            if col in experiments.columns:
                options = sorted(experiments[col].dropna().unique().tolist())
                selected = st.multiselect(col.replace("_", " ").title(), options, default=options)
                filter_cols[col] = selected

    filtered = filter_df(experiments, filter_cols)

    # --- Metric cards ---
    total = len(filtered)
    success = (
        filtered[filtered.get("status", pd.Series()) == "success"]
        if "status" in filtered.columns
        else filtered
    )
    best_loss = None
    for col in ["best_val_loss", "val_loss"]:
        if col in filtered.columns:
            best_loss = filtered[col].min()
            break

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Runs", f"{total:,}")
    m2.metric("Successful", f"{len(success):,}")
    avg_dur = (
        filtered["duration_seconds"].mean() if "duration_seconds" in filtered.columns else None
    )
    m3.metric("Avg Duration", f"{avg_dur:.0f}s" if avg_dur else "N/A")
    m4.metric("Best Val Loss", f"{best_loss:.6f}" if best_loss is not None else "N/A")

    st.divider()

    # --- Tabs ---
    tab_board, tab_seeds, tab_kd, tab_compare, tab_data = st.tabs(
        ["Leaderboard", "Seed Aggregation", "KD Transfer", "Model Comparison", "Raw Data"]
    )

    with tab_board:
        st.subheader("Leaderboard")
        sort_col = None
        for candidate in ["best_val_loss", "val_loss", "f1", "accuracy"]:
            if candidate in filtered.columns:
                sort_col = candidate
                break

        if sort_col:
            ascending = sort_col in ("best_val_loss", "val_loss")
            display = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
        else:
            display = filtered

        show_cols = [
            c
            for c in [
                "run_name",
                "dataset",
                "model_type",
                "scale",
                "stage",
                "has_kd",
                "best_val_loss",
                "val_loss",
                "f1",
                "accuracy",
                "duration_seconds",
                "peak_gpu_mb",
                "status",
            ]
            if c in display.columns
        ]
        if show_cols:
            st.dataframe(
                display[show_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No displayable columns found.")

    with tab_seeds:
        st.subheader("Seed Aggregation (Mean ± Std)")
        if "run_group" not in filtered.columns:
            st.info("No run_group data available. Run with --seeds to generate multi-seed data.")
        else:
            import pandas as pd

            groups = filtered.dropna(subset=["run_group"]).groupby("run_group")
            agg_rows = []
            metric_candidates = ["best_val_loss", "val_loss", "f1", "accuracy"]
            agg_metrics = [c for c in metric_candidates if c in filtered.columns]

            for group_name, group_df in groups:
                row = {"run_group": group_name, "n_seeds": len(group_df)}
                for m in agg_metrics:
                    vals = group_df[m].dropna()
                    if len(vals) > 0:
                        row[f"{m}_mean"] = vals.mean()
                        row[f"{m}_std"] = vals.std()
                        row[f"{m}_display"] = f"{vals.mean():.4f} ± {vals.std():.4f}"
                agg_rows.append(row)

            if agg_rows:
                agg_df = pd.DataFrame(agg_rows)
                display_cols = ["run_group", "n_seeds"] + [
                    f"{m}_display" for m in agg_metrics if f"{m}_display" in agg_df.columns
                ]
                st.dataframe(
                    agg_df[display_cols].sort_values("run_group"),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart of mean ± std for best metric
                if agg_metrics and f"{agg_metrics[0]}_mean" in agg_df.columns:
                    best_m = agg_metrics[0]
                    chart_df = agg_df.dropna(subset=[f"{best_m}_mean"])
                    if not chart_df.empty:
                        fig = px.bar(
                            chart_df,
                            x="run_group",
                            y=f"{best_m}_mean",
                            error_y=f"{best_m}_std",
                            title=f"{best_m} by Run Group (mean ± std)",
                            labels={
                                f"{best_m}_mean": best_m.replace("_", " ").title(),
                                "run_group": "Run Group",
                            },
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No multi-seed run groups found.")

    with tab_kd:
        st.subheader("Knowledge Distillation Transfer")
        if "has_kd" not in filtered.columns:
            st.info("No KD data available.")
        else:
            kd_runs = filtered[filtered["has_kd"] == True]  # noqa: E712
            nokd_runs = filtered[filtered["has_kd"] == False]  # noqa: E712

            if kd_runs.empty or nokd_runs.empty:
                st.info("Need both KD and non-KD runs for comparison.")
            else:
                metric_col = None
                for c in ["best_val_loss", "val_loss"]:
                    if c in filtered.columns:
                        metric_col = c
                        break

                if metric_col:
                    compare_data = []
                    for _, row in filtered.iterrows():
                        compare_data.append(
                            {
                                "Dataset": row.get("dataset", ""),
                                "Model": row.get("model_type", ""),
                                "Scale": row.get("scale", ""),
                                "KD": "With KD" if row.get("has_kd") else "Without KD",
                                metric_col: row.get(metric_col),
                            }
                        )

                    import pandas as pd

                    compare_df = pd.DataFrame(compare_data).dropna(subset=[metric_col])
                    if not compare_df.empty:
                        fig = px.bar(
                            compare_df,
                            x="Dataset",
                            y=metric_col,
                            color="KD",
                            barmode="group",
                            facet_col="Scale",
                            title="KD vs Non-KD Performance",
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab_compare:
        st.subheader("Model Comparison")
        if "model_type" in filtered.columns and "duration_seconds" in filtered.columns:
            metric_col = None
            for c in ["best_val_loss", "val_loss"]:
                if c in filtered.columns:
                    metric_col = c
                    break

            if metric_col:
                fig = px.scatter(
                    filtered.dropna(subset=[metric_col, "duration_seconds"]),
                    x="duration_seconds",
                    y=metric_col,
                    color="model_type",
                    symbol="scale",
                    hover_data=["dataset", "stage"],
                    title="Performance vs Training Time",
                    labels={
                        "duration_seconds": "Training Duration (s)",
                        metric_col: metric_col.replace("_", " ").title(),
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for comparison chart.")

        if "peak_gpu_mb" in filtered.columns:
            fig = px.box(
                filtered.dropna(subset=["peak_gpu_mb"]),
                x="model_type" if "model_type" in filtered.columns else None,
                y="peak_gpu_mb",
                color="scale" if "scale" in filtered.columns else None,
                title="GPU Memory Usage by Model",
                labels={"peak_gpu_mb": "Peak GPU (MB)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.subheader(f"All Runs ({len(filtered):,} rows)")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, "kd_gat_experiments.csv", "text/csv")

# =====================================================================
# SWEEPS SECTION
# =====================================================================
elif page == "Sweeps":
    st.title("📊 KD-GAT Hyperparameter Sweeps")

    sweeps = load_sweeps()

    if sweeps.empty:
        st.warning("No sweep data available. Run a tune sweep first.")
        st.stop()

    # --- Sidebar filters ---
    with st.sidebar:
        st.divider()
        st.subheader("Filters")

        if st.button("Reload Data", key="reload_sweep"):
            st.cache_data.clear()
            st.rerun()

        filter_cols = {}
        for col in ["stage", "dataset", "scale", "status"]:
            if col in sweeps.columns:
                options = sorted(sweeps[col].dropna().unique().tolist())
                selected = st.multiselect(col.title(), options, default=options, key=f"sw_{col}")
                filter_cols[col] = selected

    filtered = filter_df(sweeps, filter_cols)
    hp_cols = get_hp_columns(filtered)

    # --- Metric cards ---
    total_trials = len(filtered)
    completed = filtered[filtered["status"] == "TERMINATED"]
    error_trials = filtered[filtered["status"] == "ERROR"]
    completion_rate = len(completed) / total_trials * 100 if total_trials > 0 else 0
    best_val_loss = completed["val_loss"].min() if not completed.empty else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trials", f"{total_trials:,}")
    m2.metric("Completed", f"{len(completed):,}")
    m3.metric("Completion Rate", f"{completion_rate:.0f}%")
    m4.metric("Best val_loss", f"{best_val_loss:.6f}" if best_val_loss is not None else "N/A")

    st.divider()

    tab_overview, tab_parallel, tab_sensitivity, tab_data = st.tabs(
        ["Overview", "Parallel Coords", "HP Sensitivity", "Raw Trials"]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            status_counts = filtered["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = px.pie(
                status_counts,
                names="status",
                values="count",
                title="Trial Outcomes",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            if not completed.empty:
                best_per_sweep = (
                    completed.sort_values("val_loss").groupby("sweep_id").first().reset_index()
                )
                display_cols = ["sweep_id", "stage", "dataset", "scale", "val_loss", "duration_s"]
                display_cols = [c for c in display_cols if c in best_per_sweep.columns]
                st.subheader("Best Trial per Sweep")
                st.dataframe(
                    best_per_sweep[display_cols].sort_values("val_loss"),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "val_loss": st.column_config.NumberColumn("Val Loss", format="%.6f"),
                        "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
                    },
                )
            else:
                st.info("No completed trials.")

        if not completed.empty and "duration_s" in completed.columns:
            fig = px.histogram(
                completed,
                x="duration_s",
                color="stage",
                title="Trial Duration Distribution",
                nbins=20,
                labels={"duration_s": "Duration (seconds)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_parallel:
        if completed.empty or not hp_cols:
            st.info("No completed trials with HP data.")
        else:
            numeric_hp = [
                c for c in hp_cols if completed[c].dtype in ("float64", "int64", "float32")
            ]
            if numeric_hp:
                pc_cols = numeric_hp + ["val_loss"]
                pc_df = completed[pc_cols].dropna()
                if not pc_df.empty:
                    fig = px.parallel_coordinates(
                        pc_df,
                        color="val_loss",
                        dimensions=pc_cols,
                        color_continuous_scale=px.colors.diverging.Tealrose,
                        color_continuous_midpoint=pc_df["val_loss"].median(),
                        title="Hyperparameter Parallel Coordinates",
                        labels={c: c.removeprefix("hp_") for c in pc_cols},
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid data for parallel coordinates.")
            else:
                st.info("No numeric HP columns found.")

    with tab_sensitivity:
        if completed.empty or not hp_cols:
            st.info("No completed trials with HP data.")
        else:
            numeric_hp = [
                c for c in hp_cols if completed[c].dtype in ("float64", "int64", "float32")
            ]
            if not numeric_hp:
                st.info("No numeric HP columns for sensitivity analysis.")
            else:
                n_plots = min(len(numeric_hp), 8)
                for i in range(0, n_plots, 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx >= n_plots:
                            break
                        hp = numeric_hp[idx]
                        with col:
                            fig = px.scatter(
                                completed,
                                x=hp,
                                y="val_loss",
                                color="stage",
                                title=hp.removeprefix("hp_"),
                                labels={hp: hp.removeprefix("hp_"), "val_loss": "Val Loss"},
                                opacity=0.7,
                            )
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.subheader(f"All Trials ({len(filtered):,} rows)")
        display_cols = [
            "sweep_id",
            "trial_id",
            "stage",
            "dataset",
            "scale",
            "status",
            "val_loss",
            "duration_s",
            "timestamp",
        ] + hp_cols
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].sort_values("val_loss", na_position="last"),
            use_container_width=True,
            hide_index=True,
            column_config={
                "val_loss": st.column_config.NumberColumn("Val Loss", format="%.6f"),
                "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
            },
        )
        csv = filtered[display_cols].to_csv(index=False)
        st.download_button("Download CSV", csv, "kd_gat_sweeps.csv", "text/csv")

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("KD-GAT | CAN Bus Intrusion Detection")
