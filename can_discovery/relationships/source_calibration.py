"""Source-vehicle calibration and third-trip factor validation."""
from __future__ import annotations
import re
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..preprocessing.capture import Capture
from .evidence import FactorEvidence
from .physics_factors import FACTOR_ROLES, evaluate_factor
from .observational_edges import GRAPH_EDGES, evaluate_graph_edge
from .vehicle_models import MODEL_ROLES, evaluate_vehicle_model

@dataclass(frozen=True, slots=True)
class SourceRoleCalibration:
    role: str
    positive_mean: float
    negative_mean: float
    scale: float
    source_vehicles: tuple[str, ...]
    positive_count: int
    negative_count: int
    inclusion_threshold: float = 0.0
    leave_one_vehicle_out_recall: tuple[tuple[str, float], ...] = ()

    def adjustment(self, score: float) -> float:
        midpoint = (self.positive_mean + self.negative_mean) / 2
        direction = 1 if self.positive_mean >= self.negative_mean else -1
        return float(np.clip(direction * (score - midpoint) / self.scale, -2, 2) * .15)


@dataclass(frozen=True, slots=True)
class SourceFactorValidation:
    name: str
    supporting_vehicles: tuple[str, ...]
    weight: float
    signatures: tuple[tuple[str, int, int, float], ...]


def fit_source_factor_validation(sources: list[tuple[str, list[Capture], Path]], *,
                                 target_vehicle: str) -> tuple[SourceFactorValidation, ...]:
    """Validate source DBC factors on third trips, excluding the target vehicle."""
    from ..decoding.dbc import load_dbc_fields, rank_dbc
    from .physical_roles import _sample_values

    reports: dict[str, list[tuple[str, FactorEvidence]]] = {name: [] for name in FACTOR_ROLES}
    for vehicle, trips, dbc in sources:
        if vehicle.casefold() == target_vehicle.casefold():
            raise ValueError('target-vehicle DBC cannot validate its own factors')
        if len(trips) != 3 or len({trip.sha256 for trip in trips}) != 3:
            raise ValueError('factor validation requires three independent trips')
        if any(any(frame.attack for frame in trip.frames) for trip in trips):
            raise ValueError('factor validation received attack labels')
        if any(rank_dbc(trip, dbc)['coverage'] < .5 for trip in trips):
            raise ValueError(f'source DBC fails coverage preflight: {vehicle}')
        fields = load_dbc_fields(dbc)
        by_role = {}
        for field in fields:
            role = _dbc_role(field.label_hypotheses[0][0])
            if role:
                histories = [_sample_values(trip, field, cap=512)[1] for trip in trips[:2]]
                activity = min((len(np.unique(values)) for values in histories), default=0)
                if activity > by_role.get(role, (0, None))[0]:
                    by_role[role] = (activity, field)
        for name, roles in FACTOR_ROLES.items():
            if any(role not in by_role for role in roles):
                continue
            chosen = tuple(by_role[role][1] for role in roles)
            fitted = evaluate_factor(trips[:2], chosen, name)
            held = evaluate_factor(trips[2:], chosen, name, forced_lag=fitted.lag_bins)
            if fitted.status == held.status == 'supported' and fitted.direction == held.direction:
                reports[name].append((vehicle, held))
    validated = []
    for name, entries in reports.items():
        compatible = max(([(vehicle, item) for vehicle, item in entries
                           if item.direction == anchor.direction and
                           abs(item.lag_bins - anchor.lag_bins) <= 3]
                          for _, anchor in entries), key=len, default=[])
        if len(compatible) < 2:
            continue
        validated.append(SourceFactorValidation(name,
            tuple(sorted(vehicle for vehicle, _ in compatible)),
            float(min(.35, np.mean([item.effect for _, item in compatible]))),
            tuple((vehicle, item.direction, item.lag_bins, item.residual_quality)
                  for vehicle, item in compatible)))
    graph_reports: dict[str, list[tuple[str, float, float]]] = {
        name: [] for name in GRAPH_EDGES}
    model_reports: dict[str, list[tuple[str, FactorEvidence]]] = {
        name: [] for name in MODEL_ROLES if name != 'model:dynamic_bicycle'}
    for vehicle, trips, dbc in sources:
        fields = load_dbc_fields(dbc)
        by_role = {}
        for field in fields:
            role = _dbc_role(field.label_hypotheses[0][0])
            if role not in ({r for edge in GRAPH_EDGES.values() for r in edge[:2]} |
                            {r for roles in MODEL_ROLES.values() for r in roles}):
                continue
            activity = min(len(np.unique(_sample_values(trip,field,cap=512)[1]))
                           for trip in trips[:2])
            if activity > by_role.get(role,(0,None))[0]:
                by_role[role] = (activity,field)
        for name,(source_role,target_role,_) in GRAPH_EDGES.items():
            if source_role not in by_role or target_role not in by_role:
                continue
            result = evaluate_graph_edge(trips[:2],by_role[source_role][1],
                                         by_role[target_role][1],name)
            if result.status != 'supported':
                continue
            held = evaluate_graph_edge(trips[2:],by_role[source_role][1],
                                       by_role[target_role][1],name,
                                       forced_lag_seconds=result.lag_seconds)
            if held.status == 'supported':
                graph_reports[name].append((vehicle,held.gain,held.lag_seconds))
        for name,roles in model_reports.items():
            required = MODEL_ROLES[name]
            if any(role not in by_role for role in required):
                continue
            chosen = tuple(by_role[role][1] for role in required)
            fitted = evaluate_vehicle_model(trips[:2],chosen,name)
            if fitted.status != 'supported':
                continue
            held = evaluate_vehicle_model(trips[2:],chosen,name,
                                          forced_lag=fitted.lag_bins)
            if held.status == 'supported' and held.direction == fitted.direction:
                model_reports[name].append((vehicle,held))
    for name,entries in graph_reports.items():
        if len({vehicle for vehicle,_,_ in entries}) < 2:
            continue
        validated.append(SourceFactorValidation('graph:'+name,
            tuple(sorted(vehicle for vehicle,_,_ in entries)),
            min(.15,float(np.mean([gain for _,gain,_ in entries]))),
            tuple((vehicle,1,int(round(lag/.02)),gain)
                  for vehicle,gain,lag in entries)))
    for name,entries in model_reports.items():
        if len({vehicle for vehicle,_ in entries}) < 2:
            continue
        validated.append(SourceFactorValidation(name,
            tuple(sorted(vehicle for vehicle,_ in entries)),
            min(.15,float(np.mean([item.effect for _,item in entries]))),
            tuple((vehicle,item.direction,item.lag_bins,item.effect)
                  for vehicle,item in entries)))
    return tuple(validated)


def _dbc_role(name: str) -> str | None:
    """Conservative source-label mapping; convenience and cruise setpoints abstain."""
    label = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    if any(token in label for token in ("CRUISE_SET", "LIMIT", "WARNING", "ERROR", "FAULT")):
        return None
    if label in ("SPEED", "VEHICLE_SPEED", "CAR_SPEED", "ROUGH_CAR_SPEED",
                 "CF_CLU_VEHICLESPEED", "CF_CLU_VEHICLESPEED2", "WHEEL_SPEED",
                 "WHEEL_SPEED_FL", "WHEEL_SPEED_FR", "WHEEL_SPEED_RL", "WHEEL_SPEED_RR"):
        return "speed"
    if label in ("RPM", "ENGINE_RPM", "ENGINE_SPEED", "CLUSTER_ENGINE_RPM"):
        return "rpm"
    if label in ("GEAR", "CVT_GEAR", "CURRENT_GEAR", "GEAR_SHIFTER",
                 "CF_LVR_GEAR", "CF_CLU_GEAR"):
        return "gear"
    if label in ("LONG_ACCEL", "LONGITUDINAL_ACCELERATION", "ACCELERATION_X", "ACCEL_X"):
        return "acceleration"
    if label in ("STEERING_ANGLE", "STEER_ANGLE"):
        return "steering"
    if label in ("YAW_RATE", "VEHICLE_YAW_RATE"):
        return "yaw"
    if label in ("LATERAL_ACCELERATION", "LAT_ACCEL", "ACCELERATION_Y", "ACCEL_Y"):
        return "lateral_acceleration"
    if label in ("BRAKE_PEDAL", "BRAKE_PRESSURE", "BRAKE_PEDAL_POS",
                 "BRAKE_PRESSURE1", "BRAKE_PRESSURE2", "BRAKE"):
        return "brake"
    if label in ("THROTTLE_PEDAL", "ACCEL_PEDAL", "ACCEL_PEDAL_POS",
                 "ACCELERATOR_PEDAL", "GAS_PEDAL", "PEDAL_GAS", "THROTTLE"):
        return "throttle"
    return None


def fit_source_calibration(sources: list[tuple[str, list[Capture], Path]], *,
                           target_vehicle: str) -> tuple[SourceRoleCalibration, ...]:
    """Calibrate unary rankings from DBC-backed source vehicles only."""
    from ..decoding.dbc import load_dbc_fields, rank_dbc
    from .physical_roles import _sample_values, _series_metrics, _role_score, ROLES

    examples: dict[str, dict[str, list[float]]] = {
        role: {"positive": [], "negative": []} for role in ROLES}
    vehicles: dict[str, set[str]] = {role: set() for role in ROLES}
    positives_by_vehicle: dict[str, dict[str, list[float]]] = {role: {} for role in ROLES}
    for vehicle, captures, dbc_path in sources:
        if vehicle.casefold() == target_vehicle.casefold():
            raise ValueError("target-vehicle DBC cannot calibrate its own role scores")
        if len(captures) < 2 or len({capture.sha256 for capture in captures}) != len(captures):
            raise ValueError("source calibration requires distinct normal references")
        if any(any(frame.attack for frame in capture.frames) for capture in captures):
            raise ValueError("source calibration received labeled attack frames")
        if any(rank_dbc(capture, dbc_path)["coverage"] < .5 for capture in captures):
            raise ValueError(f"source DBC fails coverage preflight: {vehicle}")
        for field in load_dbc_fields(dbc_path):
            truth_role = _dbc_role(field.label_hypotheses[0][0])
            if truth_role is None:
                continue
            series = [_sample_values(capture, field, cap=2048) for capture in captures]
            metrics = _series_metrics(series)
            if metrics is None:
                continue
            for role in ROLES:
                score, _ = _role_score(role, series, field, metrics)
                if not math.isfinite(score):
                    continue
                bucket = "positive" if role == truth_role else "negative"
                examples[role][bucket].append(score)
                if bucket == "positive":
                    vehicles[role].add(vehicle)
                    positives_by_vehicle[role].setdefault(vehicle, []).append(score)
    calibrated = []
    for role in ROLES:
        positive, negative = examples[role]["positive"], examples[role]["negative"]
        if len(positive) < 3 or len(negative) < 10 or len(vehicles[role]) < 2:
            continue
        scale = max(.05, float(np.std(positive + negative)))
        checks = []
        for held, values in positives_by_vehicle[role].items():
            fitting = [value for vehicle, scores in positives_by_vehicle[role].items()
                       if vehicle != held for value in scores]
            if not fitting:
                break
            cut = float(np.quantile(fitting, .05)) - .02
            checks.append((held, float(np.mean(np.asarray(values) >= cut))))
        if len(checks) != len(vehicles[role]) or min(value for _, value in checks) < .95:
            continue
        threshold = float(np.quantile(positive, .05)) - .02
        calibrated.append(SourceRoleCalibration(
            role, float(np.mean(positive)), float(np.mean(negative)), scale,
            tuple(sorted(vehicles[role])), len(positive), len(negative), threshold,
            tuple(sorted(checks))))
    return tuple(calibrated)
