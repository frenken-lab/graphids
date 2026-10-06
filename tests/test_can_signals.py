from __future__ import annotations

import numpy as np
import pytest

from can_discovery.decoding.dbc import (
    evaluate_discovery,
    load_dbc_fields,
    transfer_label_candidates,
)
from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import (
    FieldCandidate,
    _bit_positions,
    _detect_mux,
    _verified_checksums,
    _verified_counters,
    discover_fields,
    fit_fields,
    propose_fields,
    unresolved_spans,
)
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.preprocessing.message_types import (
    classify_messages,
)
from can_discovery.relationships.shared.signal_time import align
from can_discovery.relationships.physical import (
    fit_physical_roles,
    fit_source_calibration,
    fit_source_factor_validation,
)


def _capture(seed: int) -> Capture:
    frames = []
    for i in range(320):
        payload = ((i + seed) * 3).to_bytes(2, "little") + bytes([i % 16, 0, 0, 0, 0, 0])
        frames.append(Frame(i * .01, i, "123", payload))
    return Capture(str(seed), str(seed), tuple(frames), 0, 3.19, 0, 0,
                   {8: len(frames)}, {16: len(frames)})


def test_tokenized_candidate_pool_and_weighted_role_sets():
    refs = [_capture(0), _capture(11)]
    pool = propose_fields(refs, max_ids=1, max_candidates_per_id=8192)
    layouts = {(item.start, item.width, item.endian, item.signed)
               for item in pool if item.role == "unknown"}
    assert layouts
    assert len(layouts) < 64
    found = fit_physical_roles(refs, max_ids=1, max_role_candidates=3,
                               max_combinations=4)
    assert len(found.combinations) > 1
    assert sum(item.weight for item in found.combinations) == pytest.approx(1)
    assert all(len(item.assignments) == 9 for item in found.combinations)
    assert any(None in item.assignments for item in found.combinations)


def test_registry_selected_layout_comes_from_its_candidate_pool():
    registry = discover_fields([_capture(0), _capture(11)], max_ids=1,
                               min_changes=10)
    keys = {(field.arb_id, field.payload_length, field.start, field.width,
             field.endian, field.signed) for field in registry.candidates}
    assert registry.selected
    assert all((field.arb_id, field.payload_length, field.start, field.width,
                field.endian, field.signed) in keys for field in registry.selected)


def test_target_dbc_cannot_train_its_own_roles(tmp_path):
    refs = [_capture(0), _capture(11)]
    with pytest.raises(ValueError, match='target-vehicle DBC'):
        fit_source_calibration([('target', refs, tmp_path / 'target.dbc')],
                               target_vehicle='target')
    with pytest.raises(ValueError, match='target-vehicle DBC'):
        fit_source_factor_validation([('target', [*refs, _capture(23)],
                                       tmp_path / 'target.dbc')], target_vehicle='target')


def test_nonoverlapping_layout_recovers_ramp_and_counter(tmp_path):
    dbc = tmp_path / "known.dbc"
    dbc.write_text('''VERSION ""
NS_ :
BS_:
BU_: ECU
BO_ 291 Example: 8 ECU
 SG_ Ramp : 0|16@1+ (1,0) [0|65535] "" ECU
 SG_ Counter : 16|4@1+ (1,0) [0|15] "" ECU
''')
    refs = [_capture(0), _capture(11)]
    fields = fit_fields(refs, max_ids=1, min_changes=10)
    actual = {(field.start, field.width, field.endian, field.role) for field in fields}
    assert actual
    assert all(not (_bit_positions(a.start, a.width, a.endian) &
                    _bit_positions(b.start, b.width, b.endian))
               for i, a in enumerate(fields) for b in fields[i + 1:])
    result = evaluate_discovery(fields, load_dbc_fields(dbc), _capture(23))
    assert result["dbc_signal_count"] == 2
    transferred = transfer_label_candidates(fields, load_dbc_fields(dbc))
    assert len(transferred) == len(fields)
    unresolved = unresolved_spans(fields, refs)
    assert sum(span.width for span in unresolved) > 0
    assert all(span.arb_id == "123" and span.payload_length == 8 for span in unresolved)


def test_transport_frames_are_classified_before_field_fitting():
    captures = []
    for seed in (0, 1):
        frames = []
        for i in range(40):
            frames.extend((Frame(i, 4 * i, "700", bytes([0x10, 0x14, i, 0, 0, 0, 0, 0])),
                           Frame(i + .01, 4 * i + 1, "700", bytes([0x21, i, 1, 2, 3, 4, 5, 6])),
                           Frame(i + .02, 4 * i + 2, "700", bytes([0x22, i, 1, 2, 3, 4, 5, 6])),
                           Frame(i + .03, 4 * i + 3, "18ECFF00", bytes([i, 0, 0, 0, 0, 0, 0, 0]))))
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0, 39.02,
                                0, 0, {8: len(frames)}, {16: len(frames)}))
    types = {item.arb_id: item for item in classify_messages(captures)}
    assert types["700"].kind == "iso_tp"
    assert types["18ECFF00"].kind == "j1939_transport"
    assert types["18ECFF00"].pgn == 0xEC00
    assert fit_fields(captures, min_changes=10) == ()


def test_counter_wrap_and_dropped_frames_use_unsampled_transmissions():
    captures = []
    for seed in (0, 5):
        frames = [Frame(i * .01, i, "123", bytes([((i + i // 31 + seed) % 16), 0]))
                  for i in range(320)]
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0, 3.19,
                                0, 0, {2: 320}, {4: 320}))
    assert (0, 4) in _verified_counters(captures, "123", 2)


def test_checksum_requires_held_out_calculation():
    captures = []
    for seed in (0, 1):
        frames = []
        for i in range(160):
            data = bytes([(i * 17 + seed) % 256, (i * 29) % 256])
            checksum = (sum(data) + 37) & 255
            frames.append(Frame(i * .01, i, "123", bytes([checksum]) + data))
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0, 1.59,
                                0, 0, {3: 160}, {6: 160}))
    assert (0, "sum8") in _verified_checksums(captures, "123", 3, set())
    altered = []
    for capture in captures:
        frames = tuple(Frame(frame.timestamp, frame.ordinal, frame.arb_id,
                             bytes([frame.payload[0] ^ 1]) + frame.payload[1:]
                             if frame.ordinal >= 80 else frame.payload)
                       for frame in capture.frames)
        altered.append(Capture(capture.source, capture.sha256, frames,
                               capture.start, capture.end, 0, 0, {}, {}))
    assert (0, "sum8") not in _verified_checksums(altered, "123", 3, set())


def test_byte_order_signedness_and_mux_conditioning():
    payload = bytes([0xFE, 0x01])
    assert extract(payload, 0, 8, "little", True) == -2
    assert extract(payload, 0, 16, "big", False) == 0xFE01
    assert extract(payload, 0, 16, "little", False) == 0x01FE
    recordings = []
    for seed in (0, 1):
        recordings.append([bytes([i % 2, ((i + seed) % 50) if i % 2 else 0,
                                  ((i + seed) % 50) if not i % 2 else 0])
                           for i in range(200)])
    assert _detect_mux(recordings) is None
