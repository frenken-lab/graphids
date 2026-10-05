"""Conservative, PicaCAN-inspired evidence for physical-role graphs.

These observational edges rank candidate assignments; they do not validate
signal boundaries or establish intervention-level causality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from .field_analysis import align


@dataclass(frozen=True, slots=True)
class GraphEdgeEvidence:
    source_role: str
    target_role: str
    status: str
    gain: float = 0.
    control_gain: float = 0.
    lag_seconds: float = 0.
    support: tuple[int, ...] = ()
    reason: str = ''


# Expected sign applies to the response in motion, not to raw CAN polarity.
GRAPH_EDGES = {
    'throttle_rpm': ('throttle', 'rpm', 1),
    'throttle_speed': ('throttle', 'speed', 1),
    'brake_speed': ('brake', 'speed', -1),
}


def duplicate_signal_groups(hypotheses, captures: list[Capture], base_groups: list[int],
                            *, max_per_role: int = 8) -> tuple[list[int], set[int]]:
    """Group replicated affine copies across IDs and identify the earlier copy."""
    parent = list(range(len(hypotheses)))
    leads = set(range(len(hypotheses)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    prior = {}
    for index,group in enumerate(base_groups):
        key = (hypotheses[index].role,group)
        if key in prior:
            parent[root(index)] = root(prior[key])
        else:
            prior[key] = index
    for role in {item.role for item in hypotheses}:
        indices = sorted((i for i,item in enumerate(hypotheses) if item.role == role),
                         key=lambda i:-hypotheses[i].unary_score)[:max_per_role]
        for offset,left in enumerate(indices):
            for right in indices[offset+1:]:
                a,b = hypotheses[left].candidate,hypotheses[right].candidate
                if (a.arb_id,a.payload_length) == (b.arb_id,b.payload_length):
                    continue
                trip_lags = []
                for capture in captures:
                    try:
                        values,valid,_,_ = align(capture,(a,b),step=.02,
                                                 max_age=.1,max_bins=4096)
                    except ValueError:
                        break
                    best = (0.,0)
                    for lag in range(-5,6):
                        if lag < 0:
                            x,y = values[-lag:,0],values[:lag,1]
                            keep = valid[-lag:,0]&valid[:lag,1]
                        elif lag > 0:
                            x,y = values[:-lag,0],values[lag:,1]
                            keep = valid[:-lag,0]&valid[lag:,1]
                        else:
                            x,y = values[:,0],values[:,1]
                            keep = valid[:,0]&valid[:,1]
                        if keep.sum() < 150:
                            continue
                        x,y = x[keep],y[keep]
                        if min(np.std(x),np.std(y)) < 1e-8:
                            continue
                        level = float(np.corrcoef(x,y)[0,1])
                        dx,dy = np.diff(x),np.diff(y)
                        if min(np.std(dx),np.std(dy)) < 1e-8:
                            continue
                        change = float(np.corrcoef(dx,dy)[0,1])
                        score = min(level,change)
                        if score > best[0]:
                            best = (score,lag)
                    if best[0] < .98:
                        break
                    trip_lags.append(best[1])
                if len(trip_lags) != len(captures) or len(set(np.sign(trip_lags))) > 1:
                    continue
                parent[root(right)] = root(left)
                if trip_lags[0] > 0:
                    leads.discard(right)
                elif trip_lags[0] < 0:
                    leads.discard(left)
    grouped = {}
    result = []
    for index in range(len(base_groups)):
        key = (hypotheses[index].role,root(index))
        result.append(grouped.setdefault(key,len(grouped)))
    return result,leads


def _predictive_gain(target, history, driver, mask, sign):
    indices = np.flatnonzero(mask)
    if len(indices) < 100:
        return 0., 0
    split = len(indices)//2
    train, test = indices[:split], indices[split:]
    base = np.column_stack((np.ones(len(indices)), history[indices]))
    full = np.column_stack((base, driver[indices]))
    if np.std(driver[train]) < 1e-9 or np.std(target[train]) < 1e-9:
        return 0., 0
    base_coef = np.linalg.lstsq(base[:split],target[train],rcond=None)[0]
    full_coef = np.linalg.lstsq(full[:split],target[train],rcond=None)[0]
    direction = int(np.sign(full_coef[-1]))
    if sign and direction != sign:
        return 0., direction
    baseline = np.mean((target[test]-base[split:]@base_coef)**2)
    if baseline < 1e-10:
        return 0., direction
    residual = np.mean((target[test]-full[split:]@full_coef)**2)
    return float(np.clip(1-residual/baseline,0,1)), direction


def evaluate_graph_edge(captures: list[Capture], source: FieldCandidate,
                        target: FieldCandidate, name: str, *,
                        step: float = .02, max_bins: int = 12000,
                        forced_lag_seconds: float | None = None) -> GraphEdgeEvidence:
    """Require activity, forward prediction, a shifted control, and trip replication."""
    source_role,target_role,sign = GRAPH_EDGES[name]
    if not captures:
        return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='no_trips')
    trip_data = []
    for capture in captures:
        try:
            values,valid,_,actual_step = align(capture,(source,target),step=step,
                                               max_age=.12,max_bins=max_bins)
        except ValueError:
            return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='invalid_time_grid')
        mask = valid.all(axis=1)
        if mask.sum() < 220:
            return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='insufficient_joint_samples')
        driver,response = values.T
        if min(np.std(driver[mask]),np.std(response[mask])) < 1e-8:
            return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='no_excitation')
        # Response changes remove a shared speed trend; brake and throttle
        # must precede subsequent changes rather than merely track speed.
        y = np.gradient(response,actual_step)
        if np.std(y[mask]) < 1e-8:
            return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='no_response_motion')
        lags = ([max(1,round(forced_lag_seconds/actual_step))] if forced_lag_seconds
                is not None else [max(1,round(seconds/actual_step))
                for seconds in (.04,.08,.12,.2,.32)])
        outcomes = []
        for lag in sorted(set(lags)):
            shift = max(lag+1,round(1.5/actual_step))
            if len(y) <= shift+lag+100:
                continue
            keep = mask & np.roll(mask,lag) & np.roll(mask,shift+lag)
            keep[:shift+lag] = False
            if keep.sum() < 100:
                continue
            history = np.roll(y,lag)
            actual,_ = _predictive_gain(y,history,np.roll(driver,lag),keep,sign)
            control,_ = _predictive_gain(y,history,np.roll(driver,lag+shift),keep,sign)
            outcomes.append((actual-control,actual,control,lag*actual_step,int(keep.sum())))
        if not outcomes:
            return GraphEdgeEvidence(source_role,target_role,'unavailable',reason='insufficient_control_samples')
        trip_data.append(max(outcomes,key=lambda item:item[0]))
    if forced_lag_seconds is None and len(captures) > 1:
        # Choose delay on the first trip; evaluate it independently elsewhere.
        return evaluate_graph_edge(captures,source,target,name,step=step,max_bins=max_bins,
                                   forced_lag_seconds=trip_data[0][3])
    net = min(item[0] for item in trip_data)
    weakest = min(trip_data,key=lambda item:item[0])
    return GraphEdgeEvidence(source_role,target_role,
             'supported' if net > .01 else 'failed',max(0.,float(net)),
             float(weakest[2]),float(weakest[3]),
             tuple(item[4] for item in trip_data))
