"""Optional experimental catch22, STUMPY, and Tigramite analyses."""
from __future__ import annotations
import math
import numpy as np
from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from .field_analysis import FieldEdge, field_key, _regular_history, align

def fingerprints(capture: Capture, fields: tuple[FieldCandidate, ...]) -> dict[str, dict]:
    """catch22 plus raw update and increment statistics for suitable fields."""
    import pycatch22

    output = {}
    for field in fields:
        values, step, total = _regular_history(capture, field)
        if len(values) < 32 or len(np.unique(values)) < 6:
            continue
        descriptor = pycatch22.catch22_all(values.tolist())
        features = {name: float(value) for name, value in
                    zip(descriptor["names"], descriptor["values"], strict=True)
                    if np.isfinite(value)}
        output[field_key(field)] = {
            "samples": len(values), "total_observations": total,
            "sample_step_seconds": step,
            "median_abs_change": float(np.median(np.abs(np.diff(values)))),
            "range": [float(np.min(values)), float(np.max(values))],
            "catch22": features,
        }
    return output


def shape_motifs(capture: Capture, fields: tuple[FieldCandidate, ...], *,
                 windows: tuple[int, ...] = (16, 32, 64), max_fields: int = 16) -> dict:
    """STUMPY matrix profiles on observed field values within one recording."""
    from .experimental_matrix_profile import univariate_profile

    output = {}
    for field in fields[:max_fields]:
        values, step, total = _regular_history(capture, field)
        if len(values) < 128 or len(np.unique(values)) < 8:
            continue
        results = []
        for window in windows:
            if len(values) < 2 * window:
                continue
            try:
                profile = univariate_profile(values, window)
            except ValueError:
                continue
            results.append({"window_samples": window,
                            "window_seconds": window * step,
                            "total_observations": total,
                            "motif_indices": profile["motif_indices"].tolist(),
                            "discord_index": profile["discord_index"]})
        if results:
            output[field_key(field)] = results
    return output


def predictive_edges(captures: list[Capture], fields: tuple[FieldCandidate, ...], *,
                     step: float = .1, max_age: float = .5, max_lag: float = .5,
                     alpha: float = .05, max_fields: int = 12) -> tuple[FieldEdge, ...]:
    """Screen conditional lagged links with Tigramite, requiring both references.

    The output is predictive association. Observational CAN traffic does not
    establish physical causality or rule out an unobserved common driver.
    """
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    selected = tuple(field for field in fields if field.role == "smooth")[:max_fields]
    if len(selected) < 2 or len(captures) < 2:
        return ()
    per_capture = []
    for capture in captures:
        values, valid, _, actual_step = align(capture, selected, step=step, max_age=max_age)
        if len(values) < 100 or np.any(valid.sum(axis=0) < 50):
            return ()
        varying = [np.std(values[valid[:, i], i]) > 0 for i in range(len(selected))]
        if not all(varying):
            return ()
        frame = DataFrame(values, mask=~valid, var_names=[field_key(f) for f in selected])
        pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(mask_type="xyz"), verbosity=0)
        tau_max = max(1, math.ceil(max_lag / actual_step))
        result = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=.05,
                                 fdr_method="fdr_bh")
        q = pcmci.get_corrected_pvalues(result["p_matrix"], fdr_method="fdr_bh",
                                         tau_min=1, tau_max=tau_max)
        per_capture.append((result["val_matrix"], q, valid, actual_step))
    edges = []
    for source in range(len(selected)):
        for target in range(len(selected)):
            if source == target:
                continue
            candidates = []
            for effects, q, valid, actual_step in per_capture:
                lags = np.flatnonzero(q[source, target, 1:] < alpha) + 1
                if not len(lags):
                    break
                tau = int(lags[np.argmax(np.abs(effects[source, target, lags]))])
                candidates.append((tau * actual_step, float(effects[source, target, tau]),
                                   float(q[source, target, tau]),
                                   int(np.count_nonzero(valid[:, source] & valid[:, target]))))
            if len(candidates) != len(per_capture):
                continue
            if len({np.sign(item[1]) for item in candidates}) != 1:
                continue
            lags_seconds = [item[0] for item in candidates]
            if max(lags_seconds) - min(lags_seconds) > max(step, max_lag / 2):
                continue
            edges.append(FieldEdge(field_key(selected[source]), field_key(selected[target]),
                                   "predictive_value", float(np.median(lags_seconds)),
                                   float(np.mean([item[1] for item in candidates])),
                                   max(item[2] for item in candidates),
                                   tuple(item[3] for item in candidates), actual_step,
                                   ("tigramite_pcmci", "fdr_bh", "independent_references")))
    return tuple(edges)
