"""Prepare, submit and inspect the selected per-ID timing-gap candidate.

python -m scripts.event_quality_suite prepare OUTPUT
python -m scripts.event_quality_suite submit OUTPUT
python -m scripts.event_quality_suite status OUTPUT

Preparation freezes runtime source and the reviewed manifest. Submission is
explicit; the fit requires a successful CUDA gate. No target
files are opened, no thresholds are fitted, and no old run is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from graphids.data.recordings import CANRecordingManifest, file_hash
from graphids.exp.config import ExperimentConfig
from graphids.exp.slurm import (
    build_experiment_job,
    dependency_versions,
    snapshot_source,
    submit_script,
    write_job,
)
from graphids.paths import PROJECT_ROOT, env_value
from scripts.event_suite_report import suite_report


def persist(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cohort(base):
    """Count existing lossless source caches and verify reviewed source files."""
    data = base["config"]["data"]
    manifest = CANRecordingManifest.load(data["manifest_path"], verify_files=False)
    rows = []
    for rec in manifest.records:
        if rec.dataset not in data["datasets"] or rec.role not in (
            "train_query",
            "val_query",
        ):
            continue
        if file_hash(Path(rec.path)) != rec.sha256:
            raise ValueError(f"reviewed source changed: {rec.key}")
        digest = hashlib.sha256(
            json.dumps(asdict(rec), sort_keys=True).encode()
        ).hexdigest()
        cache = Path(data["cache_root"]) / f"event-v1-{digest}"
        metadata = json.loads((cache / "metadata.json").read_text())
        if metadata["digest"] != digest:
            raise ValueError(f"source cache identity differs: {rec.key}")
        labels = np.load(cache / "labels.npy", mmap_mode="r", allow_pickle=False)
        ids = np.load(cache / "ids.npy", mmap_mode="r", allow_pickle=False)
        if len(labels) != metadata["count"] or len(ids) != len(labels):
            raise ValueError(f"incomplete source cache: {rec.key}")
        unique_ids = len(np.unique(ids))
        if unique_ids > 512:
            raise ValueError(f"source exceeds declared node capacity: {rec.key}")
        rows.append(
            dict(
                recording=rec.key,
                role=rec.role,
                vehicle=rec.vehicle,
                verified_normal=rec.verified_normal,
                source_digest=digest,
                count=len(labels),
                positives=int((labels == 1).sum()),
                negatives=int((labels == 0).sum()),
                unknown=int((labels < 0).sum()),
                unique_ids=unique_ids,
            )
        )
    train = [r for r in rows if r["role"] == "train_query"]
    validation = [r for r in rows if r["role"] == "val_query"]
    positive = sum(r["positives"] for r in train)
    negative = sum(r["negatives"] for r in train)
    if not train or not validation or not positive or not negative:
        raise ValueError("suite requires training classes and a validation cohort")
    if not any(r["verified_normal"] for r in train):
        raise ValueError("suite requires verified-normal training recordings")
    return dict(
        recordings=rows,
        training_rows=sum(r["count"] for r in train),
        validation_rows=sum(r["count"] for r in validation),
        training_positives=positive,
        training_negatives=negative,
    )


def job_script(root, name, commands, *, gpu=False):
    source = root / "source"
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --clusters=ascend",
        "#SBATCH --account=pas1266",
        "#SBATCH --partition=nextgen",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={8 if gpu else 2}",
        f"#SBATCH --mem={64 if gpu else 8}G",
        f"#SBATCH --time={'00:45:00' if gpu else '00:10:00'}",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --output={root / 'slurm' / (name + '-%j.log')}",
    ]
    if gpu:
        lines.append("#SBATCH --gres=gpu:a100:1")
    lines += [
        "set -euo pipefail",
        "module load python/3.12 cuda/12.8.1",
        f"cd {shlex.quote(str(source))}",
        'export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        shlex.join(
            [
                str(source / ".venv/bin/python"),
                "-m",
                "scripts.event_quality_suite",
                "verify",
                str(root),
            ]
        ),
        *commands,
    ]
    path = root / "slurm/scripts" / f"{name}.sbatch"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def prepare(root, preset, *, seed=42):
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"suite already exists: {root}")
    base = ExperimentConfig.from_yaml(preset).model_dump(mode="json")
    if base["stage"] != "fit" or base["config"].get("ckpt_path"):
        raise ValueError("candidate runs require fresh fits, not checkpoint resumes")
    population = cohort(base)
    # One fresh fit, with the same 25-epoch budget as the selected ablation.
    prediction_bytes = population["validation_rows"] * 6 * 8 * 25
    free = shutil.disk_usage(root.parent).free
    if free < prediction_bytes * 1.2 + 2 * 1024**3:
        raise ValueError("insufficient space for complete candidate score artifacts")
    stat = os.statvfs(root.parent)
    if stat.f_favail and stat.f_favail < 5000:
        raise ValueError("insufficient inode headroom for source and run artifacts")
    root.mkdir()
    for directory in ("comparison", "slurm/scripts", "source"):
        (root / directory).mkdir(parents=True)
    source = root / "source"
    snapshot = snapshot_source(source, source_root=PROJECT_ROOT)
    hashes, git_sha = snapshot.hashes, snapshot.git_sha
    (root / "working-tree.diff").write_text(snapshot.diff)
    # Sanitized runtime settings: credentials and live tracking are not copied.
    settings = dict(
        GRAPHIDS_RUN_ROOT=str(root / "runs"),
        GRAPHIDS_SLURM_LOG_DIR=str(root / "slurm"),
        GRAPHIDS_MLFLOW_MODE="offline",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if lake := env_value("GRAPHIDS_LAKE_ROOT"):
        settings["GRAPHIDS_LAKE_ROOT"] = lake
    (source / ".env").write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in settings.items())
    )
    hashes[".env"] = file_hash(source / ".env")
    persist(root / "source-sha256.json", hashes)
    persist(root / "dependency-versions.json", dependency_versions())
    persist(root / "training-cohort.json", population)
    manifest = root / "training-manifest.json"
    # Resolve paths before relocating the manifest, including reserved metadata.
    original = CANRecordingManifest.load(
        base["config"]["data"]["manifest_path"], verify_files=False
    )
    raw = json.loads(Path(base["config"]["data"]["manifest_path"]).read_text())
    raw["records"] = [asdict(rec) for rec in original.records]
    persist(manifest, raw)
    base["config"]["data"]["manifest_path"] = str(manifest)
    cfg = deepcopy(base)
    cfg["experiment_name"] = f"event_id_gap_seed{seed}"
    cfg["seed"] = seed
    cfg["config"]["model"].update(
        schema_version=2,
        use_id_gap=True,
        positive_weight=1.0,
        payload_weight=1.0,
        max_graph_tokens=8388608,
    )
    cfg["representation_cfg"]["schema_version"] = 2
    cfg["config"]["data"].update(
        representation_cfg=deepcopy(cfg["representation_cfg"]),
        chunk_events=16384,
        stream_batch_size=1,
        effective_batch_events=65536,
        max_microbatch_events=65536,
    )
    cfg["config"]["trainer"].update(max_epochs=25, enable_checkpointing=True)
    jobs = []
    config_hashes = {}
    verify_command = [
        str(source / ".venv/bin/python"),
        "-m",
        "scripts.event_quality_suite",
        "verify",
        str(root),
    ]
    cfg["experiment_name"] = root.name + "_" + cfg["experiment_name"]
    cfg["git_sha"] = git_sha
    path = root / "comparison" / f"{cfg['experiment_name']}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    validated = ExperimentConfig.from_yaml(path)
    job = build_experiment_job(
        validated,
        source_root=source,
        log_dir=root / "slurm",
        preflight=[verify_command],
        modules=["cuda/12.8.1"],
    )
    for artifact in (path, *write_job(job)):
        config_hashes[str(artifact.relative_to(root))] = file_hash(artifact)
    jobs.append(
        dict(
            name=cfg["experiment_name"],
            config=str(path),
            script=str(job.script_path),
            phases=["candidate"],
        )
    )
    python = str(source / ".venv/bin/python")
    gate_command = shlex.join(
        [
            python,
            "scripts/event_model_gate.py",
            "--mode",
            "gpu",
            "--output",
            str(root / "candidate-gate"),
            "--config",
            jobs[0]["config"],
            "--repeats",
            "1",
            "--no-profiles",
            "--chunks",
            "8",
            "--streams",
            "1",
            "--events-per-recording",
            "32",
            "--effective-events",
            "32",
            "--max-microbatch-events",
            "32",
        ]
    )
    gate_scripts = [
        dict(
            phase="candidate",
            script=job_script(root, "candidate-gate", [gate_command], gpu=True),
        )
    ]
    report_command = shlex.join(
        [
            python,
            "scripts/event_suite_report.py",
            str(root / "suite.json"),
            "--phase",
            "candidate",
            "--require-complete",
            "--output",
            str(root / "candidate-results.json"),
        ]
    )
    reports = [
        dict(
            phase="candidate",
            script=job_script(root, "candidate-report", [report_command]),
        )
    ]
    for entry in [*gate_scripts, *reports]:
        path = Path(entry["script"])
        config_hashes[str(path.relative_to(root))] = file_hash(path)
    config_hashes["training-manifest.json"] = file_hash(manifest)
    persist(root / "artifact-sha256.json", config_hashes)
    suite = dict(
        schema_version=2,
        root=str(root),
        prepared_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        seed=seed,
        status="prepared",
        training_jobs=jobs,
        gates=gate_scripts,
        reports=reports,
        expected_prediction_bytes=prediction_bytes,
        free_bytes_at_preparation=free,
        free_inodes_at_preparation=stat.f_favail,
        candidate="per-ID timing gap, ordinary BCE, one stream, joint payload objective",
        interpretation="Exploratory source validation; no target access or threshold fitting",
    )
    persist(root / "suite.json", suite)
    verify(root)
    return suite


def verify(root):
    for filename, base in (
        ("source-sha256.json", root / "source"),
        ("artifact-sha256.json", root),
    ):
        for name, expected in json.loads((root / filename).read_text()).items():
            if file_hash(base / name) != expected:
                raise ValueError(f"frozen suite file changed: {base / name}")
    if dependency_versions() != json.loads(
        (root / "dependency-versions.json").read_text()
    ):
        raise ValueError("suite Python dependency versions changed")


def submit(root):
    verify(root)
    path = root / "suite.json"
    suite = json.loads(path.read_text())
    if suite["status"] != "prepared":
        raise ValueError(
            "suite already submitted or partially submitted; inspect before retrying"
        )
    suite["status"] = "submitting"
    persist(path, suite)

    def queue(entry, dependency=None):
        receipt = submit_script(entry["script"], dependency=dependency)
        job_id = receipt.job_id
        entry.update(job_id=job_id, submit_command=list(receipt.command))
        persist(
            path, suite
        )  # Retain each accepted job even if a later submission fails.
        return job_id

    try:
        gate = queue(suite["gates"][0])
        fit_ids = [queue(job, f"afterok:{gate}") for job in suite["training_jobs"]]
        queue(suite["reports"][0], "afterany:" + ":".join(fit_ids))
        suite["status"] = "submitted"
    except Exception as exc:
        suite.update(status="submission_incomplete", submission_error=repr(exc))
        raise
    finally:
        persist(path, suite)
    return suite


def status(root):
    path = root / "suite.json"
    suite = json.loads(path.read_text())
    job_ids = [
        item["job_id"]
        for item in [*suite["gates"], *suite["training_jobs"], *suite["reports"]]
        if item.get("job_id")
    ]
    if job_ids:
        output = subprocess.check_output(
            [
                "sacct",
                "-M",
                "ascend",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(job_ids),
                "--format=JobIDRaw,State,ExitCode,Elapsed",
            ],
            text=True,
        )
        suite["scheduler_accounting"] = {
            fields[0]: dict(state=fields[1], exit_code=fields[2], elapsed=fields[3])
            for line in output.splitlines()
            if len(fields := line.split("|")) >= 4
        }
    reports = []
    for entry in suite["reports"]:
        phase = entry["phase"]
        report = suite_report(path, phase=phase)
        persist(root / f"{phase}-results.json", report)
        reports.append(report)
    states = [item["state"] for item in suite.get("scheduler_accounting", {}).values()]
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
    if (
        job_ids
        and len(states) == len(job_ids)
        and all(state.split()[0] in terminal for state in states)
    ):
        suite["status"] = (
            "complete"
            if all(r["status"] == "complete" for r in reports)
            and all(state == "COMPLETED" for state in states)
            else "incomplete"
        )
    suite["audited_at"] = datetime.now(UTC).isoformat()
    persist(path, suite)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "submit", "status"))
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--preset",
        type=Path,
        default=(PROJECT_ROOT / "configs/experiments/graph_mamba_events_source.yml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root, args.preset, seed=args.seed)
    elif args.action == "verify":
        verify(root)
        print(f"verified frozen source, configs and dependencies: {root}")
        return
    else:
        result = submit(root) if args.action == "submit" else status(root)
    print(f"{result['status']}: {root / 'suite.json'}")
    for item in [*result["gates"], *result["training_jobs"], *result["reports"]]:
        if item.get("job_id"):
            accounting = result.get("scheduler_accounting", {}).get(item["job_id"], {})
            print(item["job_id"], item.get("name", item.get("phase")), accounting)


if __name__ == "__main__":
    main()
