"""Cheap, DBC-free message-family priority from repeated motion activity."""
from __future__ import annotations

import numpy as np

from ..prepare import PreparedFamily
from ..preprocessing.capture import Capture


def motion_id_priority(families: tuple[PreparedFamily, ...],
                       captures: list[Capture], *, bins: int = 64) -> list[dict]:
    """Rank families by correlated non-protocol bit changes across captures.

    This is an execution priority, not a candidate filter or a role label.
    Correlation must recur in every fitting capture; missing activity scores zero.
    """
    keys = [(item.trace.bus, item.trace.arb_id.upper(), item.trace.payload_length)
            for item in families]
    count = len(keys)
    if not count:
        return []
    index = {key: i for i, key in enumerate(keys)}
    correlations = []
    within_correlations = []
    for capture in captures:
        activity = np.zeros((count, bins), dtype=float)
        lane_activity = np.zeros((count, 4, bins), dtype=float)
        observations = np.zeros((count, bins), dtype=float)
        previous = {}
        masks = [((1 << (item.trace.payload_length * 8)) - 1) ^
                 sum(1 << bit for bit in item.protocol_bits)
                 for item in families]
        duration = max(capture.end - capture.start, 1e-9)
        for frame in capture.frames:
            if frame.payload is None:
                continue
            key = (frame.bus or capture.known_bus, frame.arb_id.upper(), len(frame.payload))
            i = index.get(key)
            if i is None:
                continue
            value = int.from_bytes(frame.payload, 'little')
            old = previous.get(i)
            previous[i] = value
            if old is None:
                continue
            slot = min(bins - 1, max(0, int((frame.timestamp - capture.start) / duration * bins)))
            changed = (old ^ value) & masks[i]
            activity[i, slot] += changed.bit_count()
            for lane in range((len(frame.payload) + 1) // 2):
                lane_activity[i, lane, slot] += ((changed >> (lane * 16)) & 0xffff).bit_count()
            observations[i, slot] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            activity = np.divide(activity, observations, out=np.zeros_like(activity),
                                 where=observations > 0)
        activity -= activity.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(activity, axis=1)
        activity = np.divide(activity, norms[:, None], out=np.zeros_like(activity),
                             where=norms[:, None] > 0)
        correlations.append(activity @ activity.T)
        lane_activity = np.divide(lane_activity, observations[:, None, :],
                                  out=np.zeros_like(lane_activity),
                                  where=observations[:, None, :] > 0)
        lane_activity -= lane_activity.mean(axis=2, keepdims=True)
        lane_norms = np.linalg.norm(lane_activity, axis=2, keepdims=True)
        lane_activity = np.divide(lane_activity, lane_norms,
                                  out=np.zeros_like(lane_activity),
                                  where=lane_norms > 0)
        within_correlations.append(np.einsum('ilb,imb->ilm', lane_activity,
                                             lane_activity))
    # A peer must show the same positive association in every fitting capture.
    peer = np.minimum.reduce(correlations)
    np.fill_diagonal(peer, 0)
    peer = np.maximum(peer, 0)
    cross_scores = np.sort(peer, axis=1)[:, -min(3, count):].mean(axis=1)
    within = np.minimum.reduce(within_correlations)
    for i, item in enumerate(families):
        lanes = (item.trace.payload_length + 1) // 2
        within[i, np.tril_indices(4)[0], np.tril_indices(4)[1]] = 0
        within[i, lanes:, :] = 0
        within[i, :, lanes:] = 0
    within_scores = np.maximum(within, 0).max(axis=(1, 2))
    scores = np.maximum(cross_scores, within_scores)
    order = sorted(range(count), key=lambda i: (-scores[i], repr(keys[i])))
    return [{'bus': keys[i][0], 'arb_id': keys[i][1],
             'payload_length': keys[i][2], 'priority': float(scores[i]),
             'cross_id_score': float(cross_scores[i]),
             'within_id_score': float(within_scores[i]),
             'rank': rank} for rank, i in enumerate(order, 1)]
