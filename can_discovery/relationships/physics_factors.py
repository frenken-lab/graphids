"""Masked, predictive vehicle-dynamics factors for provisional CAN roles."""
from __future__ import annotations

import numpy as np

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from .field_analysis import align
from .evidence import FactorEvidence

FACTOR_ROLES = {
    'speed_acceleration': ('speed', 'acceleration'),
    'speed_brake': ('speed', 'brake'),
    'speed_throttle': ('speed', 'throttle'),
    'rpm_throttle': ('rpm', 'throttle'),
    'speed_rpm_gear': ('speed', 'rpm', 'gear'),
    'steering_yaw': ('speed', 'steering', 'yaw'),
    'steering_lateral': ('speed', 'steering', 'lateral_acceleration'),
    'wheel_speed': ('speed', 'wheel_speed'),
}


def _gain(target: np.ndarray, base: np.ndarray, feature: np.ndarray,
          mask: np.ndarray, sign: int = 0) -> tuple[float, int, float]:
    if mask.sum() < 80:
        return 0., 0, 0.
    y, b, x = target[mask], base[mask], feature[mask]
    if min(np.std(y), np.std(x)) < 1e-8:
        return 0., 0, 0.
    cut = len(y) // 2
    baseline = np.column_stack((np.ones(len(y)), b))
    extended = np.column_stack((baseline, x))
    base_coef = np.linalg.lstsq(baseline[:cut], y[:cut], rcond=None)[0]
    full_coef = np.linalg.lstsq(extended[:cut], y[:cut], rcond=None)[0]
    base_error = float(np.mean((y[cut:] - baseline[cut:] @ base_coef) ** 2))
    if base_error < 1e-10:
        return 0., 0, 0.
    slope = float(full_coef[-1])
    direction = int(np.sign(slope))
    if sign and direction != sign:
        return 0., direction, 0.
    residual = y[cut:] - extended[cut:] @ full_coef
    full_error = float(np.mean(residual ** 2))
    return float(np.clip(1 - full_error / base_error, 0, 1)), direction, float(
        np.clip(1 - full_error / max(np.var(y[cut:]), 1e-10), 0, 1))


def evaluate_factor(captures: list[Capture], fields: tuple[FieldCandidate, ...],
                    name: str, indices: list[dict[tuple[str, int], list]] | None = None,
                    forced_lag: int | None = None) -> FactorEvidence:
    """Minimum independent-trip gain; unavailable data never contributes zero evidence."""
    roles = FACTOR_ROLES[name]
    if len(fields) != len(roles):
        raise ValueError('factor field count does not match its roles')
    if forced_lag is None and len(captures) > 1 and name not in ('speed_rpm_gear', 'wheel_speed'):
        first = evaluate_factor(captures[:1], fields, name,
                                indices[:1] if indices is not None else None)
        if first.status != 'supported':
            return first
        return evaluate_factor(captures, fields, name, indices, forced_lag=first.lag_bins)
    trip_results = []
    for number, capture in enumerate(captures):
        if indices is not None:
            families = {(field.arb_id, field.payload_length) for field in fields}
            frames = sorted((frame for family in families
                             for frame in indices[number].get(family, ())),
                            key=lambda frame: (frame.timestamp, frame.ordinal))
            capture = Capture(capture.source, capture.sha256, tuple(frames),
                              capture.start, capture.end, 0, 0, {}, {})
        try:
            values, valid, _, step = align(capture, fields, step=.1, max_age=.3)
        except ValueError:
            return FactorEvidence(name, roles, 'unavailable', reason='invalid_time_grid')
        n = len(values)
        if n < 110:
            return FactorEvidence(name, roles, 'unavailable', reason='short_trip')
        mask = np.all(valid, axis=1)
        if mask.sum() < 80:
            return FactorEvidence(name, roles, 'unavailable', reason='insufficient_joint_samples')
        if name == 'wheel_speed':
            target, driver = values.T
            keep = np.flatnonzero(mask)
            if len(keep) < 100 or min(np.std(target[mask]), np.std(driver[mask])) < 1e-8:
                return FactorEvidence(name, roles, 'unavailable', reason='insufficient_motion')
            train, test = keep[:len(keep) // 2], keep[len(keep) // 2:]
            def error(feature, train=train, test=test, target=target):
                design = np.column_stack((np.ones(len(train)), feature[train]))
                intercept, slope = np.linalg.lstsq(design, target[train], rcond=None)[0]
                predicted = intercept + slope * feature[test]
                return (float(np.mean((target[test] - predicted) ** 2) /
                              max(np.var(target[test]), 1e-10)), int(np.sign(slope)))
            actual_error, direction = error(driver)
            control_error, _ = error(np.roll(driver, max(10, n // 3)))
            effect = float(np.clip(control_error - actual_error, 0, 1)) if direction > 0 else 0.
            trip_results.append((effect, 0., 0, direction,
                                 float(np.clip(1 - actual_error, 0, 1)), len(test)))
            continue
        if name == 'speed_rpm_gear':
            speed, rpm, gear = values.T
            previous_rpm = np.roll(rpm, 1)
            mask &= np.roll(valid[:, 1], 1)
            mask[:1] = False
            states, counts = np.unique(gear[mask], return_counts=True)
            active = states[counts >= 30]
            if len(active) < 2 or np.count_nonzero(np.diff(gear[mask])) < 2:
                return FactorEvidence(name, roles, 'unavailable', reason='no_shifts')
            mask &= np.isin(gear, active) & (speed > np.quantile(speed[mask], .2))
            if mask.sum() < 100 or np.std(speed[mask]) < 1e-8:
                return FactorEvidence(name, roles, 'unavailable', reason='insufficient_motion')
            # Compare a separate affine RPM/speed law in each state with one
            # shared law. Shuffle complete gear runs by circularly shifting.
            x = np.column_stack((np.ones(mask.sum()), speed[mask], previous_rpm[mask]))
            y = rpm[mask]
            cut = len(y) // 2
            common_coef = np.linalg.lstsq(x[:cut], y[:cut], rcond=None)[0]
            common = np.mean((y[cut:] - x[cut:] @ common_coef) ** 2)
            if common < 1e-10:
                return FactorEvidence(name, roles, 'unavailable', reason='no_baseline_error')
            def conditional(labels, active=active, common=common, x=x, y=y, cut=cut):
                error = 0.
                for state in active:
                    train = labels[:cut] == state
                    test = labels[cut:] == state
                    if train.sum() < 20 or test.sum() < 20:
                        return common
                    coef = np.linalg.lstsq(x[:cut][train], y[:cut][train], rcond=None)[0]
                    error += float(np.sum((y[cut:][test] - x[cut:][test] @ coef) ** 2))
                return error / len(y[cut:])
            actual = float(np.clip(1 - conditional(gear[mask]) / common, 0, 1))
            control = float(np.clip(1 - conditional(np.roll(gear[mask], max(1, mask.sum() // 3))) / common, 0, 1))
            quality = float(np.clip(1 - conditional(gear[mask]) / max(np.var(y), 1e-10), 0, 1))
            trip_results.append((actual, control, 0, 0, quality, int(mask.sum())))
            continue
        if name.startswith('steering_'):
            speed, steering, response = values.T
            if np.std(steering[mask]) < 1e-8 or np.count_nonzero(np.diff(np.sign(steering[mask]))) < 2:
                return FactorEvidence(name, roles, 'unavailable', reason='no_turns')
            motion = np.abs(speed) > np.quantile(np.abs(speed[mask]), .2)
            mask &= motion
            target = response
            feature = speed * steering
            sign = 0  # CAN sign convention and steering ratio are unknown.
        else:
            first, second = values.T
            target = np.gradient(first, step)
            feature = (np.maximum(np.gradient(second, step), 0.)
                       if name in ('speed_brake', 'speed_throttle') else second)
            sign = -1 if name == 'speed_brake' else 1
            if np.std(target[mask]) < 1e-8 or np.std(feature[mask]) < 1e-8:
                return FactorEvidence(name, roles, 'unavailable', reason='no_excitation')
        best = (0., 0., 0, 0, 0., 0)
        for lag in ((forced_lag,) if forced_lag is not None else (1, 2, 3, 5)):
            keep = mask & np.roll(mask, lag)
            keep[:lag] = False
            shifted = np.roll(feature, lag)
            baseline = np.roll(target, lag)
            control_shift = max(10, n // 3)
            control_keep = keep & np.roll(mask, control_shift)
            control_keep[:control_shift + lag] = False
            effect, direction, quality = _gain(target, baseline, shifted, control_keep, sign)
            control, _, _ = _gain(target, baseline, np.roll(shifted, control_shift), control_keep, sign)
            if effect - control > best[0] - best[1]:
                best = (effect, control, lag, direction, quality, int(control_keep.sum()))
        if best[-1] < 80:
            return FactorEvidence(name, roles, 'unavailable', reason='insufficient_joint_samples')
        trip_results.append(best)
    if not trip_results:
        return FactorEvidence(name, roles, 'unavailable', reason='no_trips')
    net = min(item[0] - item[1] for item in trip_results)
    worst = min(trip_results, key=lambda item: item[0] - item[1])
    return FactorEvidence(name, roles, 'supported' if net > .01 else 'failed',
                          max(0., float(net)), float(worst[1]), int(worst[2]),
                          int(worst[3]), int(worst[5]),
                          residual_quality=float(worst[4]))
