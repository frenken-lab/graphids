"""Explicit, content-checked recording splits; usable without torch."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Role = Literal[
    "train_reference",
    "train_query",
    "val_reference",
    "val_query",
    "target_reference",
    "target_calibration",
    "target_normal_test",
    "target_attack_test",
]
ROLES = set(Role.__args__)


@dataclass(frozen=True)
class Recording:
    key: str
    dataset: str
    vehicle: str
    session: str
    path: str
    sha256: str
    role: Role
    verified_normal: bool = False
    timestamp_unit: Literal["seconds", "milliseconds", "microseconds"] = "seconds"
    payload_length_verified: bool = False
    labels_verified: bool = False
    attack_intervals: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        if self.verified_normal and self.attack_intervals:
            raise ValueError(
                "verified-normal recordings cannot contain attack intervals"
            )
        if self.role not in ROLES:
            raise ValueError(f"unsupported recording role: {self.role}")
        if not all(
            (self.key, self.dataset, self.vehicle, self.session, self.sha256, self.path)
        ):
            raise ValueError(
                "recording identity, session, path and hash must be explicit"
            )
        if self.timestamp_unit not in ("seconds", "milliseconds", "microseconds"):
            raise ValueError("timestamp units must be declared")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("recording requires a SHA256 content hash")
        if self.role.endswith("reference") or self.role in (
            "target_calibration",
            "target_normal_test",
        ):
            if not self.verified_normal:
                raise ValueError(f"{self.role} requires verified normal provenance")
        for start, end in self.attack_intervals:
            if not math.isfinite(start) or not math.isfinite(end) or start >= end:
                raise ValueError(
                    "attack intervals must be finite, nonempty, in seconds"
                )


@dataclass(frozen=True)
class CANRecordingManifest:
    records: tuple[Recording, ...]
    version: int = 1
    provenance_reviewed: bool = False

    @classmethod
    def load(
        cls, path: str | Path, *, verify_files: bool = True
    ) -> CANRecordingManifest:
        path = Path(path)
        raw = json.loads(path.read_text())
        records = []
        for item in raw["records"]:
            item = dict(item)
            item["path"] = str((path.parent / item["path"]).resolve())
            item["attack_intervals"] = tuple(
                tuple(x) for x in item.get("attack_intervals", ())
            )
            records.append(Recording(**item))
        manifest = cls(
            tuple(records), raw.get("version", 1), raw.get("provenance_reviewed", False)
        )
        manifest.validate(verify_files=verify_files)
        return manifest

    def validate(self, *, verify_files: bool = True):
        if self.version != 1 or not self.provenance_reviewed:
            raise ValueError(
                "event fitting requires a reviewed version-1 recording manifest; run the CAN audit first"
            )
        keys: set[str] = set()
        hashes: dict[str, str] = {}
        sessions: dict[tuple[str, str], str] = {}
        vehicle_domains: dict[str, set[str]] = {}
        for rec in self.records:
            if rec.key in keys:
                raise ValueError(f"duplicate recording key: {rec.key}")
            keys.add(rec.key)
            if rec.sha256 in hashes:
                raise ValueError(
                    f"duplicate recording content: {rec.key} and {hashes[rec.sha256]}; keep one canonical record"
                )
            hashes[rec.sha256] = rec.key
            session = (rec.vehicle, rec.session)
            if session in sessions and sessions[session] != rec.role:
                raise ValueError(
                    f"session crosses reference/query or split boundaries: {session}"
                )
            sessions[session] = rec.role
            vehicle_domains.setdefault(rec.vehicle, set()).add(
                "target" if rec.role.startswith("target_") else "source"
            )
            if verify_files and file_hash(Path(rec.path)) != rec.sha256:
                raise ValueError(f"recording hash changed: {rec.key}")
        if any(len(domains) > 1 for domains in vehicle_domains.values()):
            raise ValueError(
                "a held-out target vehicle appears in source training/validation"
            )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def iter_rows(rec: Recording):
    """Yield every row in physical order; validate the entire recording on exhaustion."""
    count, first = 0, None
    last = -math.inf
    factor = {"seconds": 1.0, "milliseconds": 0.001, "microseconds": 0.000001}[
        rec.timestamp_unit
    ]
    with Path(rec.path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line, raw in enumerate(reader, 2):
            try:
                timestamp = float(raw["timestamp"]) * factor
                arb_id = raw.get("arbitration_id", raw.get("arb_id"))
                payload = raw.get("data_field", raw.get("payload"))
                if not arb_id or payload is None:
                    raise ValueError("missing ID or payload")
                payload = payload.strip()
                if len(payload) % 2 or len(payload) > 16:
                    raise ValueError("invalid classic-CAN hex payload length")
                data = bytes.fromhex(payload)
                if len(data) * 2 != len(payload):
                    raise ValueError("whitespace inside payload")
                if not math.isfinite(timestamp) or timestamp < last:
                    raise ValueError("non-finite or decreasing clock")
                label = None
                if rec.labels_verified and raw.get("attack") not in (None, ""):
                    value = int(raw["attack"])
                    if value not in (0, 1):
                        raise ValueError("attack label must be binary")
                    label = value
                if rec.verified_normal:
                    if label == 1:
                        raise ValueError("positive label in verified-normal recording")
                    label = 0
                count += 1
                if first is None:
                    first = timestamp
                yield {
                    "time": timestamp,
                    "id": arb_id.strip(),
                    "data": data,
                    "label": label,
                }
                last = timestamp
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{rec.key}:{line}: {exc}") from exc
    if count < 2 or last <= first:
        raise ValueError(f"{rec.key}: insufficient time exposure")


def read_rows(rec: Recording) -> list[dict]:
    """Read one ordered recording; never infer normality or labels from names."""
    return list(iter_rows(rec))


def audit_catalog(raw_root: str | Path, catalog: dict) -> dict:
    """Inventory all requested datasets without assigning trusted split roles."""
    result = {"version": 1, "datasets": {}, "duplicates": [], "fit_ready": False}
    hashes: dict[str, str] = {}
    for dataset in catalog:
        root = Path(raw_root) / dataset
        files = []
        errors = []
        for path in sorted(root.rglob("*.csv")):
            digest = file_hash(path)
            key = f"{dataset}/{path.relative_to(root)}"
            if digest in hashes:
                result["duplicates"].append([hashes[digest], key])
            else:
                hashes[digest] = key
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                count, bad_clock, bad_payload, positives = 0, 0, 0, 0
                first, last = None, None
                for raw in reader:
                    count += 1
                    try:
                        t = float(raw["timestamp"])
                        if not math.isfinite(t) or (last is not None and t < last):
                            bad_clock += 1
                        first = t if first is None else first
                        last = t
                    except (KeyError, ValueError):
                        bad_clock += 1
                    try:
                        payload = raw.get("data_field", raw.get("payload", ""))
                        parsed = bytes.fromhex(payload)
                        if (
                            len(payload) % 2
                            or len(parsed) > 8
                            or len(parsed) * 2 != len(payload)
                        ):
                            bad_payload += 1
                    except ValueError:
                        bad_payload += 1
                    positives += raw.get("attack") == "1"
            files.append(
                {
                    "path": key,
                    "sha256": digest,
                    "columns": fields,
                    "rows": count,
                    "first_timestamp": first,
                    "last_timestamp": last,
                    "bad_clock": bad_clock,
                    "bad_payload": bad_payload,
                    "positive_rows": positives,
                }
            )
        if not files:
            errors.append("raw files unavailable")
        errors.append(
            "review vehicle/session independence, clock units, payload validity and label provenance before assigning roles"
        )
        result["datasets"][dataset] = {"files": files, "blocked_reasons": errors}
    return result
