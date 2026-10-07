"""DBC-free, hand-specified physical-role ranking of field layouts."""
from __future__ import annotations

import math

import numpy as np

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from ..relationships.physical.unary_roles import _role_score
from .composition import score_record, unary_evidence

ROLES = ('accelerator_pedal', 'brake_input', 'throttle_plate',
         'engine_rpm', 'vehicle_speed')
UNARY_ROLE = {'accelerator_pedal': 'throttle', 'brake_input': 'brake',
              'throttle_plate': 'throttle', 'engine_rpm': 'rpm',
              'vehicle_speed': 'speed'}
def _series(fitting: list[Capture], family: tuple, cap: int = 128):
    out = []
    for capture in fitting:
        frames = [f for f in capture.frames if f.payload is not None and
                  (f.bus or capture.known_bus, f.arb_id.upper(), len(f.payload)) == family]
        if len(frames) > cap:
            frames = [frames[int(i)] for i in np.linspace(0, len(frames)-1, cap, dtype=int)]
        payloads = [f.payload for f in frames]
        out.append((np.asarray([f.timestamp for f in frames]),
                    np.asarray([int.from_bytes(p, 'little') for p in payloads], dtype=np.uint64),
                    np.asarray([int.from_bytes(p, 'big') for p in payloads], dtype=np.uint64)))
    return out


def _score(field: FieldCandidate, samples: list, role: str):
    shift = (field.payload_length * 8 - field.start - field.width
             if field.endian == 'big' else field.start)
    series = []
    for times, little, big in samples:
        values = ((big if field.endian == 'big' else little) >> shift) & ((1 << field.width) - 1)
        values = values.astype(np.float64)
        if field.signed:
            values = np.where(values >= 1 << (field.width - 1), values - (1 << field.width), values)
        series.append((times, values))
    score, evidence = _role_score(UNARY_ROLE[role], series, field)
    return score, evidence



def score_family(group: dict, layouts, role: str, fitting) -> dict:
    """Score supplied field layouts without generating more candidates."""
    family = group['bus'], group['arb_id'], group['payload_length']
    samples = _series(fitting, family)
    ordered = []
    for layout in layouts:
        field = FieldCandidate(group['arb_id'], layout.start, layout.width,
            layout.endian, layout.signed, 'unknown', 0., .5, (),
            (('unknown', 1.),), ('exhaustive_tokenization',),
            group['payload_length'], bus=group['bus'])
        score, tags = _score(field, samples, role)
        if math.isfinite(score):
            ordered.append({'field': field, **score_record((unary_evidence(float(score), tags),))})
    ordered.sort(key=lambda r: (-r['score'], repr((r['field'].arb_id, r['field'].start, r['field'].width, r['field'].endian, r['field'].signed))))
    return {'pool_size': len(ordered), 'ordered': ordered}
