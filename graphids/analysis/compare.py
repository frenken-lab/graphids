"""Cross-variant comparison over MLflow's run catalog.

Four public functions drive off ``mlflow.search_runs`` (see `MLflow search
syntax <https://mlflow.org/docs/latest/ml/search/search-runs/>`_) and
``scipy.stats.bootstrap`` (see `scipy docs
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html>`_):

- :func:`leaderboard` — mean ± 95 % bootstrap BCa CI per variant.
- :func:`tie_candidates` — pairs whose single-seed gap falls within
  ``tol``; inputs to a promote-to-N=3 re-run.
- :func:`effect_size` — pairwise Cohen's d + mean-difference bootstrap CI.
  **No p-values**: N ≤ 3 has insufficient power for significance claims
  (Bouthillier et al. 2021, §5 — see ``plans/bouthillier-2021-section-5.md``).
- :func:`expected_max` — Dodge 2019 compute-budget curve, truncated at N.

All four consume TEST-phase MLflow rows (``tags."graphids.phase" = 'test'``)
and expect the classifier-flavor unified metrics (``f1_macro``,
``f1_per_class/attack``, …) written by
:func:`graphids.core.models.base.classification_test_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

DEFAULT_METRIC = "f1_macro"
DEFAULT_TOL = 0.005
DEFAULT_N_RESAMPLES = 9999
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class _Aggregate:
    """Per-variant aggregate with seed values and bootstrap CI."""

    variant: str
    values: np.ndarray
    mean: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# MLflow query surface
# ---------------------------------------------------------------------------


def _fetch_test_runs(group: str, dataset: str, metric: str) -> pd.DataFrame:
    """Return one DataFrame row per test-phase child run for ``(group, dataset)``.

    Columns: ``variant``, ``seed``, ``value``. Tag-filter across all
    experiments — works uniformly on the old ``graphids/{group}/{variant}``
    layout and the new ``graphids/{dataset}/{group}`` layout without a
    dual path. Test phase is always-fresh (one row per attempt), so we
    dedup to the latest FINISHED row per ``(variant, seed)``.
    """
    import mlflow

    from graphids._mlflow import build_search_filter, ensure_tracking_uri

    uri = ensure_tracking_uri()
    if not uri:
        raise RuntimeError(
            "MLflow tracking URI not set (GRAPHIDS_LAKE_ROOT or MLFLOW_TRACKING_URI)"
        )
    mlflow.set_tracking_uri(uri)

    df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=build_search_filter(
            dataset=dataset, group=group, phase="test", status="FINISHED"
        ),
        order_by=["attributes.start_time DESC"],
        output_format="pandas",
    )
    if df.empty:
        return pd.DataFrame(columns=["variant", "seed", "value"])

    metric_col = f"metrics.{metric}"
    if metric_col not in df.columns:
        return pd.DataFrame(columns=["variant", "seed", "value"])

    out = pd.DataFrame(
        {
            "variant": df["tags.graphids.variant"],
            "seed": pd.to_numeric(df["tags.graphids.seed"], errors="coerce"),
            "value": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()
    out["seed"] = out["seed"].astype(int)
    # Keep the most recent FINISHED attempt per (variant, seed) — order_by
    # DESC above, so ``keep='first'`` picks the newest.
    return out.drop_duplicates(subset=["variant", "seed"], keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Bootstrap BCa CI on the mean. N<3 is degenerate → return (NaN, NaN).

    N=3 borderline — scipy warns on small samples but still produces a CI.
    """
    if values.size < 3:
        return (float("nan"), float("nan"))
    res = bootstrap(
        (values,),
        np.mean,
        method="BCa",
        confidence_level=confidence,
        n_resamples=DEFAULT_N_RESAMPLES,
        random_state=0,  # deterministic CIs across invocations
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _aggregate_per_variant(df: pd.DataFrame) -> list[_Aggregate]:
    """Group a (variant, seed, value) frame into per-variant aggregates."""
    out: list[_Aggregate] = []
    for variant, sub in df.groupby("variant"):
        values = sub["value"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(values)
        out.append(
            _Aggregate(
                variant=str(variant),
                values=values,
                mean=float(values.mean()),
                ci_low=lo,
                ci_high=hi,
            )
        )
    return sorted(out, key=lambda a: -a.mean)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def leaderboard(
    group: str,
    dataset: str,
    *,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Per-variant leaderboard: mean + 95 % BCa CI, sorted by mean (desc).

    Columns: ``variant``, ``n_seeds``, ``mean``, ``ci_low``, ``ci_high``.
    ``ci_low``/``ci_high`` are NaN when ``n_seeds < 3``.
    """
    raw = _fetch_test_runs(group, dataset, metric)
    aggs = _aggregate_per_variant(raw)
    return pd.DataFrame(
        [
            {
                "variant": a.variant,
                "n_seeds": int(a.values.size),
                "mean": a.mean,
                "ci_low": a.ci_low,
                "ci_high": a.ci_high,
            }
            for a in aggs
        ]
    )


def tie_candidates(
    group: str,
    dataset: str,
    *,
    metric: str = DEFAULT_METRIC,
    tol: float = DEFAULT_TOL,
) -> pd.DataFrame:
    """Variant pairs whose mean gap is within ``tol`` — candidates for N=3 promotion.

    Columns: ``variant_a``, ``variant_b``, ``mean_a``, ``mean_b``, ``gap``,
    ``n_seeds_a``, ``n_seeds_b``. Only pairs where at least one variant has
    n_seeds == 1 (actionable ties in the screening phase).
    """
    aggs = _aggregate_per_variant(_fetch_test_runs(group, dataset, metric))
    rows = []
    for i, a in enumerate(aggs):
        for b in aggs[i + 1 :]:
            gap = abs(a.mean - b.mean)
            if gap >= tol:
                continue
            if a.values.size > 1 and b.values.size > 1:
                continue  # both already promoted — re-running won't help
            rows.append(
                {
                    "variant_a": a.variant,
                    "variant_b": b.variant,
                    "mean_a": a.mean,
                    "mean_b": b.mean,
                    "gap": gap,
                    "n_seeds_a": int(a.values.size),
                    "n_seeds_b": int(b.values.size),
                }
            )
    return pd.DataFrame(rows)


def effect_size(
    group: str,
    dataset: str,
    *,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Pairwise Cohen's d + mean-difference bootstrap CI (no p-values).

    Columns: ``variant_a``, ``variant_b``, ``mean_diff``, ``cohens_d``,
    ``diff_ci_low``, ``diff_ci_high``, ``n_seeds_a``, ``n_seeds_b``.
    ``cohens_d`` is NaN when either variant has n_seeds < 2 (undefined std);
    ``diff_ci_*`` are NaN when either variant has n_seeds < 3.
    """
    aggs = _aggregate_per_variant(_fetch_test_runs(group, dataset, metric))
    rows = []
    for i, a in enumerate(aggs):
        for b in aggs[i + 1 :]:
            mean_diff = a.mean - b.mean
            if a.values.size >= 2 and b.values.size >= 2:
                # Pooled std with (N_a - 1) + (N_b - 1) dof
                pooled = np.sqrt(
                    (
                        (a.values.size - 1) * a.values.var(ddof=1)
                        + (b.values.size - 1) * b.values.var(ddof=1)
                    )
                    / (a.values.size + b.values.size - 2)
                )
                d = float(mean_diff / pooled) if pooled > 0 else float("nan")
            else:
                d = float("nan")
            # Mean-diff CI via bootstrap on pooled seeds — independent two-sample.
            if a.values.size >= 3 and b.values.size >= 3:
                res = bootstrap(
                    (a.values, b.values),
                    lambda x, y, axis=-1: x.mean(axis=axis) - y.mean(axis=axis),
                    method="BCa",
                    confidence_level=DEFAULT_CONFIDENCE,
                    n_resamples=DEFAULT_N_RESAMPLES,
                    random_state=0,
                )
                ci_lo = float(res.confidence_interval.low)
                ci_hi = float(res.confidence_interval.high)
            else:
                ci_lo, ci_hi = float("nan"), float("nan")
            rows.append(
                {
                    "variant_a": a.variant,
                    "variant_b": b.variant,
                    "mean_diff": mean_diff,
                    "cohens_d": d,
                    "diff_ci_low": ci_lo,
                    "diff_ci_high": ci_hi,
                    "n_seeds_a": int(a.values.size),
                    "n_seeds_b": int(b.values.size),
                }
            )
    return pd.DataFrame(rows)


def expected_max(
    group: str,
    dataset: str,
    *,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Dodge 2019 expected max validation performance as a function of N.

    For each variant and each ``n`` in ``[1, N_variant]`` compute
    ``E[max(X_{i_1}, ..., X_{i_n})]`` over all ``C(N, n)`` seed subsets —
    honest empirical curve rather than the parametric i.i.d. approximation.
    Truncated at the per-variant seed count.

    Columns: ``variant``, ``n_seeds``, ``expected_max``.
    """
    from itertools import combinations

    aggs = _aggregate_per_variant(_fetch_test_runs(group, dataset, metric))
    rows = []
    for a in aggs:
        N = int(a.values.size)
        for n in range(1, N + 1):
            subset_maxes = [max(a.values[list(c)]) for c in combinations(range(N), n)]
            rows.append(
                {
                    "variant": a.variant,
                    "n_seeds": n,
                    "expected_max": float(np.mean(subset_maxes)),
                }
            )
    return pd.DataFrame(rows)
