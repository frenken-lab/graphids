"""Load experiment and sweep data from HF Datasets with Streamlit caching."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

EXPERIMENTS_REPO = "buckeyeguy/kd-gat-experiments"
SWEEPS_REPO = "buckeyeguy/kd-gat-sweeps"


@st.cache_data(ttl=300)
def load_experiments() -> pd.DataFrame:
    """Download experiments.parquet from private HF Dataset. Cached for 5 min."""
    try:
        path = hf_hub_download(
            repo_id=EXPERIMENTS_REPO, filename="experiments.parquet", repo_type="dataset"
        )
        df = pd.read_parquet(path)
        # Flatten MLflow prefixed columns to short names
        rename = {}
        for col in df.columns:
            for prefix in ("params.", "metrics.", "tags."):
                if col.startswith(prefix):
                    short = col[len(prefix) :]
                    # Only rename if short name doesn't already exist
                    if short not in df.columns and short not in rename.values():
                        rename[col] = short
        if rename:
            df = df.rename(columns=rename)
        for col in ["duration_seconds", "peak_gpu_mb", "val_loss", "best_val_loss"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sweeps() -> pd.DataFrame:
    """Download sweeps.parquet from private HF Dataset. Cached for 5 min."""
    try:
        path = hf_hub_download(repo_id=SWEEPS_REPO, filename="sweeps.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        for col in ["val_loss", "duration_s"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def get_hp_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are hyperparameters (hp_ prefix)."""
    return sorted([c for c in df.columns if c.startswith("hp_")])


def filter_df(
    df: pd.DataFrame,
    col_filters: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply column-level multiselect filters."""
    filtered = df.copy()
    for col, values in col_filters.items():
        if values and col in filtered.columns:
            filtered = filtered[filtered[col].isin(values)]
    return filtered
