"""Artifact manifest writer/reader + checksum verification.

Each completed run gets a ``_manifest.json`` with identity fields,
timestamps, git SHA, artifact inventory, and SHA-256 checksums.
Written atomically after all stage artifacts are saved.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

import graphids

log = logging.getLogger(__name__)

# Artifacts that may exist in a run directory (metrics live in manifest, not as separate file)
_KNOWN_ARTIFACTS = [
    "config.json",
    "best_model.pt",
    "embeddings.npz",
    "attention_weights.npz",
    "dqn_policy.json",
    "explanations.npz",
]


class ManifestEntry(BaseModel, frozen=True):
    """One artifact in the manifest."""

    name: str
    size_bytes: int
    sha256: str


class Manifest(BaseModel, frozen=True):
    """Artifact manifest for a completed run.

    The ``metrics`` field is the single source of truth for final run metrics.
    MLflow autolog provides live training observability; the manifest carries
    the authoritative final numbers for catalog/dashboard consumption.
    """

    # Identity
    dataset: str
    model_type: str
    scale: str
    stage: str
    auxiliaries: str = "none"
    seed: int = 42

    # Metadata
    created_at: str
    graphids_version: str
    git_sha: str = ""
    slurm_job_id: str = ""

    # Artifacts
    artifacts: list[ManifestEntry]

    # Metrics (single source of truth — replaces separate metrics.json reads)
    metrics: dict[str, object] = {}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    """Get current git SHA, or empty string if not in a repo."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def write_manifest(
    stage_dir: Path,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    auxiliaries: str = "none",
    seed: int = 42,
    metrics: dict[str, object] | None = None,
) -> Path:
    """Write _manifest.json to a completed run directory.

    Scans for known artifacts, computes checksums, writes atomically.
    Returns the manifest path.
    """
    entries = []
    for name in _KNOWN_ARTIFACTS:
        artifact_path = stage_dir / name
        if artifact_path.exists():
            entries.append(
                ManifestEntry(
                    name=name,
                    size_bytes=artifact_path.stat().st_size,
                    sha256=_sha256_file(artifact_path),
                )
            )

    if metrics is None:
        metrics = {}

    manifest = Manifest(
        dataset=dataset,
        model_type=model_type,
        scale=scale,
        stage=stage,
        auxiliaries=auxiliaries,
        seed=seed,
        created_at=datetime.now(UTC).isoformat(),
        graphids_version=graphids.__version__,
        git_sha=_git_sha(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        artifacts=entries,
        metrics=metrics,
    )

    manifest_path = stage_dir / "_manifest.json"

    from graphids.storage.gateway import StorageGateway

    # Use a dummy gateway just for the atomic write helper
    gw = StorageGateway(
        lake_root=".", dataset="manifest", model_type="manifest", scale="manifest",
    )
    gw.write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode())
    log.info("Wrote manifest: %s (%d artifacts)", manifest_path, len(entries))

    return manifest_path


def read_manifest(stage_dir: Path) -> Manifest | None:
    """Read _manifest.json from a run directory. Returns None if missing."""
    manifest_path = stage_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return Manifest.model_validate_json(manifest_path.read_text())
    except Exception as e:
        log.warning("Failed to read manifest %s: %s", manifest_path, e)
        return None


def verify_manifest(stage_dir: Path) -> tuple[bool, list[str]]:
    """Verify artifact checksums against manifest.

    Returns (all_ok, list_of_error_messages).
    """
    manifest = read_manifest(stage_dir)
    if manifest is None:
        return False, ["No _manifest.json found"]

    errors = []
    for entry in manifest.artifacts:
        artifact_path = stage_dir / entry.name
        if not artifact_path.exists():
            errors.append(f"Missing: {entry.name}")
            continue
        actual_hash = _sha256_file(artifact_path)
        if actual_hash != entry.sha256:
            errors.append(
                f"Checksum mismatch: {entry.name} (expected {entry.sha256[:12]}..., got {actual_hash[:12]}...)"
            )

    return len(errors) == 0, errors
