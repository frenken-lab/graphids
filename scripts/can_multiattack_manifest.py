"""Freeze reviewed mixed attacks, with bounded, explicitly audited clock repair.

Run inside a CPU Slurm allocation. Raw lake files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from decimal import Decimal
from pathlib import Path

from scripts.can_pilot_manifest import compare_log, sha256


def repair_clock(source, destination, bound=Decimal("0.001")):
    """Prefix-max time only; preserve frame order and all other field values."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit = destination.with_suffix(".clock.jsonl")
    maximum = Decimal("-Infinity")
    changed, largest = 0, Decimal(0)
    try:
        with (
            source.open(newline="") as inp,
            destination.open("w", newline="") as out,
            audit.open("w") as log,
        ):
            reader = csv.DictReader(inp)
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for index, row in enumerate(reader, 2):
                original = Decimal(row["timestamp"])
                if not original.is_finite():
                    raise ValueError("nonfinite clock")
                effective = max(maximum, original)
                delta = effective - original
                if delta > bound:
                    raise ValueError(
                        f"clock correction exceeds {bound}s at row {index}: {source}"
                    )
                if delta:
                    changed += 1
                    largest = max(largest, delta)
                    log.write(
                        json.dumps(
                            dict(
                                csv_line=index,
                                original=row["timestamp"],
                                effective=str(effective),
                                delta_seconds=str(delta),
                            )
                        )
                        + "\n"
                    )
                    row["timestamp"] = str(effective)
                maximum = effective
                writer.writerow(row)
    except Exception:
        destination.unlink(missing_ok=True)
        audit.unlink(missing_ok=True)
        raise
    return dict(
        method="prefix maximum; no row reordering",
        changed_rows=changed,
        maximum_correction_seconds=str(largest),
        bound_seconds=str(bound),
        raw_sha256=sha256(source),
        derived_sha256=sha256(destination),
        audit=str(audit),
        audit_sha256=sha256(audit),
    )


def repair_replay_clock(source, log_path, destination, bound=Decimal("0.001")):
    """Verify row/interface alignment, repair each stream, then stably merge.

    Merged timestamps determine bin cutoffs. Within-interface frame order and all
    frame fields survive; movements across interfaces are explicit in the audit.
    """
    from itertools import zip_longest

    destination.parent.mkdir(parents=True, exist_ok=True)
    audit = destination.with_suffix(".clock.jsonl")
    rows, previous = [], {}
    changed, largest = 0, Decimal(0)
    try:
        with source.open(newline="") as inp, log_path.open() as log:
            reader = csv.DictReader(inp)
            fields = reader.fieldnames
            offset = None
            for line, pair in enumerate(zip_longest(reader, log), 2):
                row, original = pair
                if row is None or original is None:
                    raise ValueError("replay row count differs")
                timestamp, interface, frame = original.split()
                timestamp = float(timestamp.strip("()"))
                arb, payload = frame.split("#")
                if offset is None:
                    offset = timestamp - float(row["timestamp"])
                if (
                    round(timestamp - offset, 6),
                    arb,
                    payload,
                    int(interface in {"vcan1", "vcan3", "vcan5", "vcan7"}),
                ) != (
                    round(float(row["timestamp"]), 6),
                    row["arbitration_id"],
                    row["data_field"],
                    int(row["attack"]),
                ):
                    raise ValueError(
                        f"replay row/interface alignment differs at line {line}"
                    )
                raw_time = Decimal(row["timestamp"])
                if not raw_time.is_finite():
                    raise ValueError("nonfinite replay clock")
                effective = max(previous.get(interface, raw_time), raw_time)
                delta = effective - raw_time
                if delta > bound:
                    raise ValueError(
                        f"per-interface clock correction exceeds {bound}s at row {line}"
                    )
                old_token = row["timestamp"]
                if delta:
                    changed += 1
                    largest = max(largest, delta)
                    row["timestamp"] = str(effective)
                previous[interface] = effective
                rows.append((effective, line, interface, old_token, delta, row))
        # Stable ties retain original arrival order; each interface is monotonic.
        rows.sort(key=lambda item: item[0])
        moved = 0
        with destination.open("w", newline="") as out, audit.open("w") as sidecar:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for line, (_, old_line, interface, old_token, delta, row) in enumerate(
                rows, 2
            ):
                moved += old_line != line
                if old_line != line or delta:
                    sidecar.write(
                        json.dumps(
                            dict(
                                original_csv_line=old_line,
                                derived_csv_line=line,
                                interface=interface,
                                original=old_token,
                                effective=row["timestamp"],
                                delta_seconds=str(delta),
                            )
                        )
                        + "\n"
                    )
                writer.writerow(row)
    except Exception:
        destination.unlink(missing_ok=True)
        audit.unlink(missing_ok=True)
        raise
    return dict(
        method="per-interface prefix maximum, then stable chronological merge; within-interface order preserved",
        changed_rows=changed,
        moved_rows=moved,
        rows=len(rows),
        maximum_correction_seconds=str(largest),
        bound_seconds=str(bound),
        raw_sha256=sha256(source),
        derived_sha256=sha256(destination),
        audit=str(audit),
        audit_sha256=sha256(audit),
    )


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def check_groups(assignments, canonical, groups):
    roles = {}
    for path, item in assignments.items():
        group = groups.find(canonical[path])
        role = item["role"]
        if group in roles and roles[group] != role:
            raise ValueError(
                f"reused capture crosses roles: {path}, {roles[group]}, {role}"
            )
        roles[group] = role


def acquisition_overlap(left, right):
    if left["vehicle"] != right["vehicle"] or max(
        left["acquisition_start"], right["acquisition_start"]
    ) > min(left["acquisition_end"], right["acquisition_end"]):
        return False
    virtual_left = left.get("clock_domain") == "virtual_replay"
    virtual_right = right.get("clock_domain") == "virtual_replay"
    if virtual_left and virtual_right:
        return bool(set(left["interfaces"]) & set(right["interfaces"]))
    if virtual_left != virtual_right:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.review_root
    plan = json.loads((root / "plan.json").read_text())
    review = json.loads((root / "recording-review.json").read_text())
    sources = json.loads((root / "upstream/sources.json").read_text())
    inventory = {r["path"]: r for r in review["recordings"]}
    canonical = {p: g[0] for g in review["aliases"] for p in g}
    assignments = plan["assignments"]
    if len({canonical[p] for p in assignments}) != len(assignments):
        raise ValueError("exact alias selected twice")
    groups = Components()
    for pair in review["candidate_reuse"]:
        groups.union(*pair["paths"])
    check_groups(assignments, canonical, groups)
    records, evidence, families = [], [], {}
    progress_path = root / "correspondence-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    verifier = (
        inspect.getsource(compare_log)
        + inspect.getsource(repair_clock)
        + inspect.getsource(repair_replay_clock)
    )
    prior_records, prior_proof = {}, {}
    prior_root = plan.get("reuse_review_root")
    if prior_root:
        prior_root = Path(prior_root)
        prior_records = {
            r["key"]: r
            for r in json.loads(
                (prior_root / "can-recording-manifest-v1.json").read_text()
            )["records"]
        }
        prior_proof = {
            r["key"]: r
            for r in json.loads((prior_root / "pilot-provenance.json").read_text())[
                "recordings"
            ]
        }
    for path, item in assignments.items():
        rec = inventory[canonical[path]]
        raw = args.raw_root / path
        if rec["bad_payload"] or rec["bad_labels"] or sha256(raw) != rec["sha256"]:
            raise ValueError(f"invalid/changed capture: {path}")
        upstream = [s for s in sources if s["local"] == path]
        for s in upstream:
            if sha256(s["download"]) != s["sha256"]:
                raise ValueError(f"changed upstream evidence: {path}")
        log = next(s for s in upstream if s["source"].endswith(".log"))
        if (
            log["source"] != item["source_log"]
            or log["source"].split("/")[0] != item["vehicle"]
        ):
            raise ValueError(f"wrong upstream vehicle/path: {path}")
        normal = item["attack_family"] == "normal"
        if normal:
            if "/train_01_attack_free/" not in path or rec["positive_rows"]:
                raise ValueError(f"unverified normal acquisition: {path}")
        else:
            label = next(s for s in upstream if s["source"].endswith(".csv"))
            if (
                label["source"] != item["source_labels"]
                or label["sha256"] != rec["sha256"]
            ):
                raise ValueError(f"curator labels differ: {path}")
        fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    raw=rec["sha256"],
                    upstream=[(u["url"], u["sha256"]) for u in upstream],
                    interface_labels=item.get("verify_replay_interface_labels", False),
                    verifier=verifier,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = progress.get(path, {})
        if (
            cached.get("fingerprint") == fingerprint
            and Path(cached["path"]).is_file()
            and sha256(cached["path"]) == cached["sha256"]
            and (
                cached.get("repair") is None
                or (
                    Path(cached["repair"]["audit"]).is_file()
                    and sha256(cached["repair"]["audit"])
                    == cached["repair"]["audit_sha256"]
                )
            )
        ):
            match, repair, local = (
                cached["match"],
                cached["repair"],
                Path(cached["path"]),
            )
        else:
            record_key = f"{item['vehicle']}/{raw.stem}"
            old_record, old_proof = (
                prior_records.get(record_key, {}),
                prior_proof.get(record_key, {}),
            )
            old_logs = [
                u for u in old_proof.get("upstream", []) if u["source"].endswith(".log")
            ]
            if (
                not item.get("verify_replay_interface_labels", False)
                and old_record.get("sha256") == rec["sha256"]
                and old_record.get("verified_normal") == normal
                and any(
                    u["sha256"] == log["sha256"] and u["url"] == log["url"]
                    for u in old_logs
                )
            ):
                match = {
                    k: old_proof[k]
                    for k in (
                        "rows",
                        "acquisition_start",
                        "acquisition_end",
                        "original_decreasing_clock_rows",
                        "original_largest_backstep_seconds",
                        "correspondence",
                        "ordering_provenance",
                    )
                }
                match["interfaces"] = ["reviewed-physical-original"]
                match["reused_correspondence_review"] = dict(
                    root=str(prior_root),
                    manifest_sha256=sha256(
                        prior_root / "can-recording-manifest-v1.json"
                    ),
                    proof_sha256=sha256(prior_root / "pilot-provenance.json"),
                )
            else:
                match = compare_log(
                    raw,
                    Path(log["download"]),
                    interface_labels=item.get("verify_replay_interface_labels", False),
                )
            local, repair = raw, None
            if rec["bad_clock"]:
                local = root / "derived" / path
                repair = (
                    repair_replay_clock(raw, Path(log["download"]), local)
                    if item.get("source_log_repo") == "can-ml"
                    else repair_clock(raw, local)
                )
            progress[path] = dict(
                fingerprint=fingerprint,
                path=str(local),
                sha256=sha256(local),
                match=match,
                repair=repair,
            )
            temporary = progress_path.with_suffix(".partial")
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(progress_path)
        match["clock_domain"] = (
            "virtual_replay"
            if all(i.startswith("vcan") for i in match["interfaces"])
            else "physical"
        )
        vehicle = item["vehicle"]
        key = f"{vehicle}/{raw.stem}"
        records.append(
            dict(
                key=key,
                dataset=path.split("/")[0],
                vehicle=vehicle,
                session="pending",
                path=str(local.resolve()),
                sha256=sha256(local),
                role=item["role"],
                verified_normal=normal,
                timestamp_unit="seconds",
                payload_length_verified=True,
                labels_verified=True,
                attack_intervals=[],
            )
        )
        families[key] = item["attack_family"]
        evidence.append(
            dict(
                path=path,
                key=key,
                vehicle=vehicle,
                role=item["role"],
                attack_family=item["attack_family"],
                upstream=upstream,
                clock_repair=repair,
                **match,
            )
        )
        print(
            f"verified {path}; repair={repair and repair['changed_rows']}", flush=True
        )
    for i, left in enumerate(evidence):
        for right in evidence[i + 1 :]:
            if acquisition_overlap(left, right):
                groups.union(canonical[left["path"]], canonical[right["path"]])
    check_groups(assignments, canonical, groups)
    for record, proof in zip(records, evidence, strict=True):
        record["session"] = "capture-group:" + groups.find(canonical[proof["path"]])
    from graphids.core.data.recordings import CANRecordingManifest, Recording

    CANRecordingManifest(tuple(Recording(**r) for r in records), 1, True).validate()
    manifest = dict(
        version=1, provenance_reviewed=True, records=records, attack_families=families
    )
    (root / "multiattack-provenance.json").write_text(
        json.dumps(
            dict(
                plan=plan,
                recordings=evidence,
                grouping="Global sampled-reuse connected components plus overlapping captures within the same physical/replay bus domain. Concurrent disjoint virtual buses do not establish shared capture identity. Related variants are not independent acquisitions.",
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "can-recording-manifest-v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"frozen {len(records)} recordings, {len({r['session'] for r in records})} capture groups",
        flush=True,
    )


if __name__ == "__main__":
    main()
