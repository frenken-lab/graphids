"""Recheck raw hashes and inventory sampled sequence reuse without approving splits.

Run in a CPU Slurm allocation. Fingerprints deliberately omit timestamps and
labels, and filter injected rows, to find candidate reused background traffic.
Matches are leads for provenance review, not proof of a shared acquisition.
No matches likewise cannot certify independence. Raw files are never changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WINDOW = 64
MASK = (1 << 64) - 1
BASE = 1000003


def inspect_recording(task):
    root, expected = task
    path = Path(root) / expected["path"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"raw hash changed: {path}")
    fingerprints = {}
    queue = deque()
    rolling = 0
    power = pow(BASE, WINDOW - 1, 1 << 64)
    ids = Counter()
    bad_clock = []
    bad_payload = []
    bad_labels = []
    positives = ties = rows = 0
    first = last = None
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            rows += 1
            timestamp = float(row["timestamp"])
            if not math.isfinite(timestamp) or (last is not None and timestamp < last):
                bad_clock.append(line)
            ties += last == timestamp
            first = timestamp if first is None else first
            last = timestamp
            payload = row["data_field"]
            try:
                data = bytes.fromhex(payload)
                if len(data) > 8 or len(data) * 2 != len(payload):
                    raise ValueError("invalid payload")
            except ValueError:
                bad_payload.append(line)
            label = row["attack"]
            if label not in ("0", "1"):
                bad_labels.append(line)
            positives += label == "1"
            ids[row["arbitration_id"]] += 1
            # Zero event labels are used only for reuse detection, never normality.
            if label != "0":
                continue
            token = int.from_bytes(
                hashlib.blake2b(
                    (row["arbitration_id"] + "#" + payload).encode(), digest_size=8
                ).digest(),
                "little",
            )
            if len(queue) == WINDOW:
                old, _ = queue.popleft()
                rolling = (rolling - old * power) & MASK
            rolling = (rolling * BASE + token) & MASK
            queue.append((token, line))
            # Content-defined sampling is invariant to offsets in the file.
            if len(queue) == WINDOW and rolling % 512 == 0:
                exact = hashlib.sha256(
                    b"".join(x.to_bytes(8, "little") for x, _ in queue)
                ).hexdigest()
                fingerprints.setdefault(exact, [queue[0][1], line])
    report = dict(expected)
    report.update(
        actual_sha256=digest,
        rows=rows,
        first_timestamp=first,
        last_timestamp=last,
        span_seconds=last - first,
        positive_rows=positives,
        bad_clock=len(bad_clock),
        bad_payload=len(bad_payload),
        bad_labels=len(bad_labels),
        bad_clock_line_examples=bad_clock[:10],
        bad_payload_line_examples=bad_payload[:10],
        equal_timestamp_rows=ties,
        distinct_ids=len(ids),
        sampled_windows=len(fingerprints),
        # Exact estimate from build_snapshots, without allocating dense arrays.
        unanchored_snapshot_bytes_10ms=(int((last - first) / 0.01) + 1)
        * (len(ids) * 110 + 33),
    )
    return report, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    groups = defaultdict(list)
    for dataset in inventory["datasets"].values():
        for item in dataset["files"]:
            groups[item["sha256"]].append(item)
    tasks = [(str(args.raw_root), items[0]) for items in groups.values()]
    report = {
        "version": 1,
        "provenance_reviewed": False,
        "fit_ready": False,
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": {
            "window_frames": WINDOW,
            "sampling_modulus": 512,
            "stream": "ordered zero-event-label ID/payload pairs; timestamps omitted",
            "interpretation": "candidate reuse only; absence is not independence",
        },
        "aliases": [[x["path"] for x in items] for items in groups.values()],
        "recordings": [],
        "candidate_reuse": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    index = defaultdict(list)
    pairs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec, fingerprints in pool.map(inspect_recording, tasks):
            key = rec["path"]
            report["recordings"].append(rec)
            for fingerprint, lines in fingerprints.items():
                for other, other_lines in index[fingerprint]:
                    pair = pairs.setdefault(
                        (other, key), {"matching_windows": 0, "examples": []}
                    )
                    pair["matching_windows"] += 1
                    if len(pair["examples"]) < 3:
                        pair["examples"].append([other_lines, lines])
                index[fingerprint].append((key, lines))
            print(f"inspected {key}: {rec['rows']} rows", flush=True)
            (args.output / "progress.json").write_text(
                json.dumps(
                    {"completed": len(report["recordings"]), "total": len(tasks)}
                )
            )
    report["candidate_reuse"] = [
        {"paths": list(pair), **evidence} for pair, evidence in sorted(pairs.items())
    ]
    (args.output / "recording-review.json").write_text(json.dumps(report, indent=2))
    print(f"finished: {len(tasks)} canonical recordings; {len(pairs)} reuse leads")


if __name__ == "__main__":
    main()
