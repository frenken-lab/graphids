"""Joint, abstaining physical-role hypotheses over overlapping CAN layouts."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..decoding.field_values import extract
from ..decoding.signals import FieldCandidate, _bit_positions, propose_fields
from ..preprocessing.capture import Capture
from .field_analysis import align
from .physics_factors import FACTOR_ROLES, FactorEvidence, evaluate_factor
from .observational_edges import GRAPH_EDGES, duplicate_signal_groups, evaluate_graph_edge
from .vehicle_models import MODEL_ROLES, evaluate_vehicle_model

ROLES = ("speed", "rpm", "gear", "acceleration", "steering", "brake", "throttle",
         "yaw", "lateral_acceleration")
RUNTIME_ROLES = frozenset(("speed", "rpm", "gear", "acceleration", "steering",
                           "brake", "throttle"))


@dataclass(frozen=True, slots=True)
class RoleHypothesis:
    candidate: FieldCandidate
    role: str
    unary_score: float
    support: tuple[int, ...]
    evidence: tuple[str, ...]
    equivalence_group: int = -1
    update_limit: float = 0.0
    allowed_values: tuple[int, ...] = ()
    calibration: str = "physics_only"


@dataclass(frozen=True, slots=True)
class CandidateCombination:
    """A mutually compatible role assignment; a missing role is the null choice."""
    assignments: tuple[int | None, ...]
    weight: float
    score: float
    relation_evidence: tuple[tuple[str, float], ...]
    factors: tuple[FactorEvidence, ...] = ()
    runtime_weight: float | None = None


@dataclass(frozen=True, slots=True)
class PhysicalDiscovery:
    hypotheses: tuple[RoleHypothesis, ...]
    combinations: tuple[CandidateCombination, ...]
    protocol_fields: tuple[FieldCandidate, ...]
    rejected: tuple[dict, ...]
    factor_evidence: tuple[FactorEvidence, ...] = ()


from .source_calibration import (SourceRoleCalibration, SourceFactorValidation,
                                 fit_source_calibration, fit_source_factor_validation,
                                 _dbc_role)


from .unary_roles import (_index_capture, _sample_values, _series_metrics,
                          _role_score, _field_id)


def _equivalence_groups(hypotheses: list[RoleHypothesis], captures: list[Capture],
                        indices: list[dict[tuple[str, int], list]] | None = None) -> list[int]:
    """Group affine-equivalent histories within one message family."""
    signatures = {}
    output = []
    for hypothesis in hypotheses:
        field = hypothesis.candidate
        traces = []
        for number, capture in enumerate(captures):
            _, values = _sample_values(capture, field, cap=512,
                                        index=indices[number] if indices is not None else None)
            if not len(values):
                continue
            centered = values - values[0]
            scale = np.max(np.abs(centered))
            traces.append(tuple(np.round(centered / scale, 8)) if scale else ("constant",))
        signature = (field.bus, field.arb_id, field.payload_length, field.mux_start,
                     field.mux_value, tuple(traces))
        output.append(signatures.setdefault(signature, len(signatures)))
    return output


def _relation_score(captures: list[Capture], left: FieldCandidate,
                    right: FieldCandidate, relation: str,
                    indices: list[dict[tuple[str, int], list]] | None = None) -> float:
    """Signed predictive gain over target history and a time-shift control."""
    effects = []
    for number, capture in enumerate(captures):
        if indices is not None:
            index = indices[number]
            left_frames = index.get((left.arb_id, left.payload_length), ())
            right_frames = index.get((right.arb_id, right.payload_length), ())
            selected = (left_frames if (left.arb_id, left.payload_length) ==
                        (right.arb_id, right.payload_length) else
                        sorted((*left_frames, *right_frames),
                               key=lambda frame: (frame.timestamp, frame.ordinal)))
            capture = Capture(capture.source, capture.sha256, tuple(selected),
                              capture.start, capture.end, 0, 0, {}, {})
        try:
            values, valid, _, step = align(capture, (left, right), step=.1, max_age=.3)
        except ValueError:
            return 0.0
        if len(values) < 100:
            return 0.0
        x, y = values[:, 0], values[:, 1]
        mask = valid[:, 0] & valid[:, 1]
        if relation in ('speed_acceleration', 'speed_brake', 'speed_throttle'):
            target = np.gradient(x, step)
            driver = y
            sign = -1 if relation == 'speed_brake' else 1
        elif relation == 'rpm_throttle':
            target, driver, sign = np.gradient(x, step), y, 1
        elif relation in ('speed_rpm', 'wheel_speed'):
            target, driver, sign = x, y, 1
        else:
            return 0.0
        lag = 2 if relation in ('speed_brake', 'speed_throttle', 'rpm_throttle') else 1
        mask[:lag] = False
        mask &= np.roll(valid[:, 1], lag)
        if mask.sum() < 100 or min(np.std(target[mask]), np.std(driver[mask])) <= 0:
            return 0.0
        shifted_driver = np.roll(driver, lag)
        baseline = np.roll(target, lag)
        shift = max(10, len(y) // 3)
        control_driver = np.roll(shifted_driver, shift)
        control_mask = mask & np.roll(valid[:, 1], shift)
        control_mask[:shift + lag] = False
        if control_mask.sum() < 100:
            return 0.0
        def gain(driver_values: np.ndarray, keep: np.ndarray,
                 target=target, baseline=baseline, sign=sign) -> float:
            t = target[keep]
            base = baseline[keep]
            feature = driver_values[keep]
            base_matrix = np.column_stack((np.ones(len(t)), base))
            base_error = np.mean((t - base_matrix @ np.linalg.lstsq(base_matrix, t, rcond=None)[0]) ** 2)
            if base_error <= 1e-12:
                return 0.0
            residual_driver = feature - base_matrix @ np.linalg.lstsq(base_matrix, feature, rcond=None)[0]
            residual_target = t - base_matrix @ np.linalg.lstsq(base_matrix, t, rcond=None)[0]
            if np.std(residual_driver) <= 1e-12:
                return 0.0
            slope = float(np.dot(residual_driver, residual_target) /
                          np.dot(residual_driver, residual_driver))
            if slope * sign <= 0:
                return 0.0
            return float(np.clip(1 - np.mean((residual_target - slope * residual_driver) ** 2) /
                                 base_error, 0, 1))
        effects.append(max(0.0, gain(shifted_driver, control_mask) -
                           gain(control_driver, control_mask)))
    return min(effects, default=0.0)


def fit_physical_roles(*args, **kwargs) -> PhysicalDiscovery:
    """Compatibility entry point for the bounded physical combination ranker."""
    from .combination_ranker import fit_physical_roles as rank
    return rank(*args, **kwargs, _functions=globals())
