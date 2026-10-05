"""Held-out, scale-tolerant tests derived from elementary vehicle models.

Model evidence ranks role combinations; it never certifies CAN bit layouts.
"""
from __future__ import annotations

import numpy as np

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from .field_analysis import align
from .physics_factors import FactorEvidence

MODEL_ROLES = {
    'model:double_integrator': ('speed','acceleration'),
    'model:kinematic_yaw': ('speed','steering','yaw'),
    'model:kinematic_lateral': ('speed','yaw','lateral_acceleration'),
    'model:dynamic_bicycle': ('speed','steering','yaw','lateral_acceleration'),
}


def _series(capture: Capture, fields: tuple[FieldCandidate,...], name: str,
            lag: int, *, step: float = .05):
    try:
        values, valid, _, actual_step = align(capture,fields,step=step,
                                               max_age=.15,max_bins=12000)
    except ValueError:
        return None
    horizon = max(2,round(.25/actual_step)) if name == 'model:double_integrator' else max(1,round(.1/actual_step))
    shift = max(10,round(2./actual_step))
    n = len(values)
    if n <= shift + lag + 2*horizon + 120:
        return None
    keep = np.all(valid,axis=1)
    for offset in (-horizon,horizon,lag,lag+shift):
        keep &= np.roll(np.all(valid,axis=1),offset)
    keep[:shift+lag+horizon] = False
    keep[-horizon:] = False
    if keep.sum() < 160:
        return None
    if name == 'model:double_integrator':
        speed,accel = values.T
        y = np.roll(speed,-horizon)-speed
        history = speed-np.roll(speed,horizon)
        feature = np.roll(accel,lag)*horizon*actual_step
        control = np.roll(accel,lag+shift)*horizon*actual_step
        motion = np.std(y[keep]) > 1e-8 and np.std(feature[keep]) > 1e-8
    elif name == 'model:kinematic_yaw':
        speed,steering,yaw = values.T
        y = np.roll(yaw,-horizon)
        history = yaw
        # Small-angle bicycle: yaw rate is proportional to speed × road-wheel
        # steering angle. Scale and steering ratio remain nuisance parameters.
        feature = speed*np.roll(steering,lag)
        control = speed*np.roll(steering,lag+shift)
        motion = (np.std(speed[keep]) > 1e-8 and np.std(steering[keep]) > 1e-8
                  and np.count_nonzero(np.diff(np.sign(steering[keep]))) >= 2)
    else:
        speed,yaw,lateral = values.T
        y = np.roll(lateral,-horizon)
        history = lateral
        # Kinematic centripetal acceleration a_lat = v × yaw_rate.
        feature = speed*np.roll(yaw,lag)
        control = speed*np.roll(yaw,lag+shift)
        motion = (np.std(speed[keep]) > 1e-8 and np.std(yaw[keep]) > 1e-8
                  and np.count_nonzero(np.diff(np.sign(yaw[keep]))) >= 2)
    if not motion or np.std(y[keep]) < 1e-8:
        return None
    return y[keep],history[keep],feature[keep],control[keep],int(keep.sum())


def _fit_and_score(train, test):
    yt,bt,xt,ct,_ = train
    y,b,x,c,_ = test
    if min(np.std(yt),np.std(xt),np.std(y)) < 1e-8:
        return None
    def error(feature_train,feature_test):
        design_train = np.column_stack((np.ones(len(yt)),bt,feature_train))
        design_test = np.column_stack((np.ones(len(y)),b,feature_test))
        coef = np.linalg.lstsq(design_train,yt,rcond=None)[0]
        return float(np.mean((y-design_test@coef)**2)),float(coef[-1])
    baseline = np.linalg.lstsq(np.column_stack((np.ones(len(yt)),bt)),yt,rcond=None)[0]
    base_error = float(np.mean((y-(baseline[0]+baseline[1]*b))**2))
    if base_error < 1e-10:
        return None
    actual,slope = error(xt,x)
    shifted,_ = error(ct,c)
    gain = float(np.clip(1-actual/base_error,0,1))
    control_gain = float(np.clip(1-shifted/base_error,0,1))
    return gain,control_gain,int(np.sign(slope))


def evaluate_vehicle_model(captures: list[Capture], fields: tuple[FieldCandidate,...],
                           name: str, *, forced_lag: int | None = None) -> FactorEvidence:
    roles = MODEL_ROLES[name]
    if len(fields) != len(roles):
        raise ValueError('model field count does not match roles')
    if name == 'model:dynamic_bicycle':
        return FactorEvidence(name,roles,'unavailable',reason='lateral_velocity_not_observed')
    if not captures:
        return FactorEvidence(name,roles,'unavailable',reason='no_trips')
    lags = (forced_lag,) if forced_lag is not None else (0,1,2,4)
    best = None
    for lag in lags:
        first = _series(captures[0],fields,name,lag)
        if first is None:
            continue
        midpoint = len(first[0])//2
        training = tuple(value[:midpoint] if isinstance(value,np.ndarray) else midpoint
                         for value in first)
        testing = tuple(value[midpoint:] if isinstance(value,np.ndarray) else len(first[0])-midpoint
                        for value in first)
        result = _fit_and_score(training,testing)
        if result is None:
            continue
        net = result[0]-result[1]
        if best is None or net > best[0]:
            best = (net,lag,training,result,first[4])
    if best is None:
        return FactorEvidence(name,roles,'unavailable',reason='insufficient_motion_or_samples')
    _,lag,training,first_result,first_support = best
    results = [first_result]
    supports = [first_support]
    for capture in captures[1:]:
        series = _series(capture,fields,name,lag)
        if series is None:
            return FactorEvidence(name,roles,'unavailable',reason='insufficient_independent_trip')
        result = _fit_and_score(training,series)
        if result is None:
            return FactorEvidence(name,roles,'unavailable',reason='unidentifiable_scale')
        results.append(result)
        supports.append(series[4])
    if len(set(item[2] for item in results)) != 1:
        return FactorEvidence(name,roles,'failed',lag_bins=lag,
                              valid_samples=min(supports),reason='inconsistent_direction')
    net = min(item[0]-item[1] for item in results)
    weakest = min(results,key=lambda item:item[0]-item[1])
    return FactorEvidence(name,roles,'supported' if net > .01 else 'failed',
                          max(0.,float(net)),float(weakest[1]),lag,
                          weakest[2],min(supports),
                          residual_quality=float(weakest[0]))
