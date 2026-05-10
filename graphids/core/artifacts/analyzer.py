"""Per-checkpoint artifact generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
from structlog import get_logger

from graphids.core.models.base import eval_mode, safe_load_checkpoint
from graphids.exp.config import AnalyzeConfig

from . import compute, io

log = get_logger(__name__)

MANIFEST_NAME = "analysis_manifest.json"


@dataclass(frozen=True)
class Artifact:
    name: str
    output: str
    applies_to: frozenset[str]
    run: Callable[..., None]


def _run_embeddings(*, model, val_data, device, output_dir, spec: "AnalyzeConfig", **_) -> None:
    r = compute.compute_embeddings(
        model,
        val_data,
        device,
        model_type=spec.model_type,
        max_samples=spec.embedding_max_samples,
        batch_size=spec.batch_size,
    )
    io.save_embeddings(output_dir, r)


def _run_attention(*, model, val_data, device, output_dir, spec: "AnalyzeConfig", **_) -> None:
    r = compute.compute_attention(model, val_data, device, max_samples=spec.attention_max_samples)
    if r is None:
        return
    io.save_attention(output_dir, r)


def _run_cka(*, model, val_data, device, output_dir, spec: "AnalyzeConfig", **_) -> None:
    teacher = io.load_teacher("gat", spec.cka_teacher_ckpt, device)
    try:
        scores = compute.compute_cka(
            model, teacher, val_data, device, max_samples=spec.cka_max_samples
        )
        io.save_cka(output_dir, scores)
    finally:
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_landscape(*, model, val_data, device, output_dir, spec: "AnalyzeConfig", hparams, **_) -> None:
    r = compute.compute_landscape(
        model,
        spec.model_type,
        val_data,
        device,
        hparams,
        resolution=spec.landscape_resolution,
        scale=spec.landscape_scale,
        max_graphs=spec.landscape_max_graphs,
        seed=spec.seed,
        dataset=spec.dataset,
    )
    io.save_landscape(output_dir, r)


def _run_fusion_policy(*, module, device, output_dir, spec: "AnalyzeConfig", **_) -> None:
    td, labels = io.load_fusion_eval(dataset=spec.dataset, seed=spec.seed, device=device)
    r = compute.compute_fusion_policy(module, td, labels)
    io.save_fusion_policy(output_dir, r)


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("embeddings", "embeddings.npz", frozenset({"vgae", "dgi", "gat"}), _run_embeddings),
    Artifact("attention", "attention_weights.npz", frozenset({"gat"}), _run_attention),
    Artifact("cka", "cka.json", frozenset({"gat"}), _run_cka),
    Artifact(
        "landscape",
        "loss_landscape_{model_type}.parquet",
        frozenset({"vgae", "dgi", "gat"}),
        _run_landscape,
    ),
    Artifact("fusion_policy", "dqn_policy.json", frozenset({"fusion"}), _run_fusion_policy),
)


def expected_outputs(spec: AnalyzeConfig) -> tuple[str, ...]:
    out: list[str] = []
    for a in ARTIFACTS:
        if getattr(spec, a.name):
            out.append(a.output.format(model_type=spec.model_type))
    return tuple(out)


class Analyzer:
    """Generate analysis artifacts from a trained checkpoint."""

    def __init__(self, spec: AnalyzeConfig):
        self.spec = spec
        if not Path(spec.ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {spec.ckpt_path}")
        if spec.cka and not Path(spec.cka_teacher_ckpt).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {spec.cka_teacher_ckpt}")

    def run(self) -> None:
        spec = self.spec
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dir = Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "analyzer_start",
            model_type=spec.model_type,
            dataset=spec.dataset,
            output_dir=str(output_dir),
        )

        module = safe_load_checkpoint(spec.model_type, spec.ckpt_path, map_location=device)
        with eval_mode(module):
            val_data = io.load_val_data(
                lake_root=spec.lake_root,
                dataset=spec.dataset,
                vocab_scope=spec.vocab_scope,
                seed=spec.seed,
                representation_cfg=spec.representation_cfg,
            )
            ctx = dict(
                model=module,
                module=module,
                val_data=val_data,
                device=device,
                output_dir=output_dir,
                spec=spec,
                hparams=module.hparams,
            )
            for a in ARTIFACTS:
                if not getattr(spec, a.name):
                    continue
                log.info("artifact_start", artifact=a.name)
                a.run(**ctx)

        self._write_manifest(output_dir)
        log.info("analyzer_done", output_dir=str(output_dir))

    def _write_manifest(self, output_dir: Path) -> None:
        spec = self.spec
        expected = expected_outputs(spec)
        manifest = {
            "contract": "graphids.analyze",
            "version": 1,
            "dataset": spec.dataset,
            "model_type": spec.model_type,
            "checkpoint_path": spec.ckpt_path,
            "output_dir": str(output_dir),
            "expected_outputs": list(expected),
            "existing_outputs": [n for n in expected if (output_dir / n).exists()],
        }
        (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
