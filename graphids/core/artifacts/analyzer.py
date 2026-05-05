"""Per-checkpoint artifact generation.

``Analyzer(spec)`` consumes an :class:`AnalyzeRow` directly — no parallel
30-kwarg constructor to drift against the schema. Artifact dispatch and
expected-output computation come from :mod:`_dispatch`, the single source
of truth for "which artifacts fire for which model type."
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from structlog import get_logger

from graphids.configs.blueprint import AnalyzeRow
from graphids.core.models.base import eval_mode, safe_load_checkpoint

from . import io
from ._dispatch import ARTIFACTS, expected_outputs

log = get_logger(__name__)

MANIFEST_NAME = "analysis_manifest.json"


class Analyzer:
    """Generate analysis artifacts from a trained checkpoint."""

    def __init__(self, spec: AnalyzeRow):
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
                window_size=spec.window_size,
                stride=spec.stride,
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
            "contract": "graphids.analyze_row",
            "version": 1,
            "dataset": spec.dataset,
            "model_type": spec.model_type,
            "checkpoint_path": spec.ckpt_path,
            "output_dir": str(output_dir),
            "expected_outputs": list(expected),
            "existing_outputs": [n for n in expected if (output_dir / n).exists()],
        }
        (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
