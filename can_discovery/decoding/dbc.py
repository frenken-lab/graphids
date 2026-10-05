"""DBC ground truth and cautious transfer into the field registry."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..preprocessing.capture import Capture
from .field_values import extract
from .signals import FieldCandidate, _bit_positions


def _dbc_start(start: int, byte_order: str) -> int:
    # cantools uses sawtooth DBC bit numbering for Motorola signals.
    return start if byte_order == "little_endian" else (start // 8) * 8 + 7 - start % 8


def load_dbc_fields(path: Path) -> tuple[FieldCandidate, ...]:
    """Read known signal layouts and semantics through cantools."""
    import cantools

    db = cantools.database.load_file(str(path), strict=False)
    fields = []
    for message in db.messages:
        arb_id = format(message.frame_id, "X")
        for signal in message.signals:
            fields.append(FieldCandidate(
                arb_id, _dbc_start(signal.start, signal.byte_order), signal.length,
                "little" if signal.byte_order == "little_endian" else "big",
                signal.is_signed, "known", 0.0, 1.0, (),
                ((signal.name, 1.0),), ("dbc", str(path)), message.length,
                _dbc_start(message.get_signal_by_name(signal.multiplexer_signal).start,
                           message.get_signal_by_name(signal.multiplexer_signal).byte_order)
                if signal.multiplexer_signal else None,
                signal.multiplexer_ids[0] if signal.multiplexer_ids and
                len(signal.multiplexer_ids) == 1 else None))
    return tuple(fields)


def rank_dbc(capture: Capture, path: Path) -> dict:
    """Rank a DBC by message coverage and successful raw decoding.

    This is a candidate score, never enough alone to assign physical labels.
    """
    import cantools

    db = cantools.database.load_file(str(path), strict=False)
    by_id = {message.frame_id: message for message in db.messages}
    counts: Counter[str] = Counter()
    for frame in capture.frames:
        if frame.payload is None:
            continue
        try:
            numeric_id = int(frame.arb_id, 16)
        except ValueError:
            continue
        message = by_id.get(numeric_id)
        if message is None:
            continue
        counts["id"] += 1
        if len(frame.payload) != message.length:
            continue
        counts["length"] += 1
        try:
            db.decode_message(numeric_id, frame.payload, decode_choices=False,
                              scaling=False, allow_truncated=False)
        except (ValueError, KeyError, cantools.database.errors.DecodeError):
            continue
        counts["decoded"] += 1
    return {"path": str(path), "matched_id_frames": counts["id"],
            "matched_length_frames": counts["length"],
            "decoded_frames": counts["decoded"],
            "coverage": counts["decoded"] / max(1, len(capture.frames))}


def evaluate_discovery(inferred: tuple[FieldCandidate, ...],
                       truth: tuple[FieldCandidate, ...], capture: Capture) -> dict:
    """Exact structural matches and raw-value error for active held-out signals."""
    inferred = tuple(field for field in inferred if field.signed is not None
                     and field.endian in ('big', 'little'))
    payloads: dict[tuple[str, int], list[bytes]] = {}
    for frame in capture.frames:
        if frame.payload is not None:
            payloads.setdefault((frame.arb_id.upper(), len(frame.payload)), []).append(frame.payload)
    known = {(field.arb_id.upper(), field.payload_length,
              _bit_positions(field.start, field.width, field.endian), field.endian): field
             for field in truth if field.mux_start is None and
             len({extract(payload, field.start, field.width, field.endian, field.signed)
                  for payload in payloads.get((field.arb_id.upper(), field.payload_length), [])}) > 1}
    matched = []
    raw_value_errors = []
    for key, reference in known.items():
        candidates = [field for field in inferred if field.arb_id.upper() == key[0]
                      and field.payload_length == key[1]]
        if not candidates:
            raw_value_errors.append({"id": reference.arb_id,
                                     "signal": reference.label_hypotheses[0][0],
                                     "candidate": None, "mean_absolute_raw_error": None})
            continue
        reference_bits = key[2]
        best = max(candidates, key=lambda field: (
            len(_bit_positions(field.start, field.width, field.endian) & reference_bits),
            -abs(field.width - reference.width)))
        if not (_bit_positions(best.start, best.width, best.endian) & reference_bits):
            raw_value_errors.append({"id": reference.arb_id,
                                     "signal": reference.label_hypotheses[0][0],
                                     "candidate": None, "mean_absolute_raw_error": None})
            continue
        values = payloads.get((key[0], key[1]), [])
        errors = [abs(extract(payload, best.start, best.width, best.endian, best.signed) -
                      extract(payload, reference.start, reference.width,
                              reference.endian, reference.signed)) for payload in values]
        raw_value_errors.append({"id": reference.arb_id,
                                 "signal": reference.label_hypotheses[0][0],
                                 "candidate": {"start": best.start, "width": best.width,
                                               "endian": best.endian, "signed": best.signed},
                                 "mean_absolute_raw_error": sum(errors) / len(errors) if errors else None})
    for field in inferred:
        key = (field.arb_id.upper(), field.payload_length,
               _bit_positions(field.start, field.width, field.endian), field.endian)
        reference = known.get(key)
        if reference is None or field.signed != reference.signed:
            continue
        errors = []
        for frame in capture.frames:
            if frame.arb_id.upper() != field.arb_id.upper() or frame.payload is None:
                continue
            if len(frame.payload) != field.payload_length:
                continue
            predicted = extract(frame.payload, field.start, field.width,
                                field.endian, field.signed)
            expected = extract(frame.payload, reference.start, reference.width,
                               reference.endian, reference.signed)
            errors.append(abs(predicted - expected))
        matched.append({"id": field.arb_id, "start": field.start, "width": field.width,
                        "endian": field.endian, "signed": field.signed,
                        "mean_raw_error": sum(errors) / len(errors) if errors else None})
    categories = {}
    for name, needle in (("speed", "speed"), ("rpm", "rpm"),
                         ("gear", "gear"), ("counter", "counter"),
                         ("checksum", "checksum")):
        selected = [field for field in known.values()
                    if needle in field.label_hypotheses[0][0].lower()]
        hits = [match for match in matched if any(
            match["id"].upper() == field.arb_id.upper() and
            match["start"] == field.start and match["width"] == field.width
            for field in selected)]
        categories[name] = {"active_signals": len(selected), "exact": len(hits),
                            "mean_raw_error": (sum(m["mean_raw_error"] for m in hits) / len(hits)
                                               if hits else None)}
    return {"candidate_count": len(inferred), "dbc_signal_count": len(known),
            "exact_count": len(matched),
            "precision": len(matched) / max(1, len(inferred)),
            "recall": len(matched) / max(1, len(known)),
            "coverage": len(matched) / max(1, len(known)),
            "categories": categories,
            "raw_value_errors": raw_value_errors,
            "matches": matched}


def transfer_label_candidates(inferred: tuple[FieldCandidate, ...],
                              known: tuple[FieldCandidate, ...], *,
                              min_message_agreement: int = 2) -> tuple[FieldCandidate, ...]:
    """Offer DBC names only when several independent layouts agree per message."""
    lookup = {(item.arb_id.upper(), item.payload_length,
               _bit_positions(item.start, item.width, item.endian),
               item.endian, item.signed): item
              for item in known if item.mux_start is None}
    matches = {}
    for item in inferred:
        key = (item.arb_id.upper(), item.payload_length,
               _bit_positions(item.start, item.width, item.endian),
               item.endian, item.signed)
        if key in lookup:
            matches[key] = lookup[key]
    agreements = Counter((key[0], key[1]) for key in matches)
    output = []
    for item in inferred:
        key = (item.arb_id.upper(), item.payload_length,
               _bit_positions(item.start, item.width, item.endian),
               item.endian, item.signed)
        known_item = matches.get(key)
        if known_item is None or agreements[(key[0], key[1])] < min_message_agreement:
            output.append(item)
            continue
        output.append(replace(item,
                              label_hypotheses=((known_item.label_hypotheses[0][0], .8),
                                                ("unknown", .2)),
                              evidence=item.evidence + ("dbc_transfer_candidate",)))
    return tuple(output)
