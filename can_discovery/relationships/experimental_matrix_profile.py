"""STUMPY baselines for decoded or candidate-field histories.

Input series must be aligned within one recording. Do not join windows across
recordings or treat a normalized motif distance as an attack score by itself.
"""

from __future__ import annotations

import numpy as np


def univariate_profile(values: np.ndarray, window: int) -> dict:
    """Find repeated subsequences and discords with ``stumpy.stump``.

    ``stumpy.motifs`` returns motif distances and locations; high matrix-profile
    values are discord candidates. Both use an exclusion zone for trivial matches.
    """
    import stumpy

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or window < 3 or len(series) < 2 * window:
        raise ValueError("expected a 1D series with at least two windows")
    profile = stumpy.stump(series, m=window)
    distances = np.asarray(profile[:, 0], dtype=float)
    motif_distances, motif_indices = stumpy.motifs(series, profile[:, 0], max_motifs=3)
    return {
        "distances": distances,
        "neighbors": np.asarray(profile[:, 1], dtype=np.int64),
        "motif_distances": motif_distances,
        "motif_indices": motif_indices,
        "discord_index": int(np.argmax(distances)),
    }


def group_profile(values: np.ndarray, window: int) -> dict:
    """Find recurring subdimensional episodes with ``stumpy.mstump``.

    Rows are signals and columns are aligned samples. ``mmotifs`` chooses
    motif locations and contributing signals; it does not infer graph edges.
    """
    import stumpy

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 2 or window < 3 or series.shape[1] < 2 * window:
        raise ValueError("expected [signals, samples] with at least two windows")
    profile, neighbors = stumpy.mstump(series, m=window)
    k = min(2, series.shape[0]) - 1
    distances, indices, subspaces, _mdls = stumpy.mmotifs(
        series, profile, neighbors, k=k, max_motifs=1, max_matches=2,
        cutoffs=np.inf, max_distance=np.inf)
    if not len(indices) or len(indices[0]) < 2 or indices[0, 0] < 0 or indices[0, 1] < 0:
        raise ValueError("no multidimensional motif with two valid matches")
    index, neighbor = int(indices[0, 0]), int(indices[0, 1])
    dimensions = subspaces[0]
    return {"profile": profile, "neighbors": neighbors, "motif_index": index,
            "neighbor_index": neighbor, "dimensions": np.asarray(dimensions, dtype=int),
            "motif_distance": float(distances[0, 1])}


def query_matches(reference: np.ndarray, target: np.ndarray, *, window: int,
                  query_index: int, max_matches: int = 10) -> list[tuple[float, int]]:
    """Locate a training motif in a separate recording via ``stumpy.match``."""
    import stumpy

    source = np.asarray(reference, dtype=np.float64)
    other = np.asarray(target, dtype=np.float64)
    if query_index < 0 or query_index + window > len(source):
        raise ValueError("query window lies outside reference")
    matches = stumpy.match(source[query_index:query_index + window], other,
                           max_matches=max_matches)
    return [(float(distance), int(index)) for distance, index in matches]
