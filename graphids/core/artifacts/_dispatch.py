"""Single dispatch table for per-checkpoint artifacts.

One row per artifact: which model types it applies to, what filename it
produces, and the load-compute-save glue. ``Analyzer.run()`` walks this
table once — there is no parallel ``ARTIFACTS_BY_MODEL_TYPE`` /
``expected_outputs`` / if-chain to drift against.

Each ``run`` callable is the only place compute and I/O meet — the
:mod:`compute` module stays pure, :mod:`io` owns all reads/writes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from . import compute, io

if TYPE_CHECKING:
    from graphids.plan.schema import AnalyzeRow


@dataclass(frozen=True)
class Artifact:
    name: str
    output: str  # may include ``{model_type}`` placeholder
    applies_to: frozenset[str]
    run: Callable[..., None]


def _run_embeddings(*, model, val_data, device, output_dir, spec: AnalyzeRow, **_) -> None:
    r = compute.compute_embeddings(
        model,
        val_data,
        device,
        model_type=spec.model_type,
        max_samples=spec.embedding_max_samples,
        batch_size=spec.batch_size,
    )
    io.save_embeddings(output_dir, r)


def _run_attention(*, model, val_data, device, output_dir, spec: AnalyzeRow, **_) -> None:
    r = compute.compute_attention(model, val_data, device, max_samples=spec.attention_max_samples)
    if r is None:  # model has no GAT attention to extract
        return
    io.save_attention(output_dir, r)


def _run_cka(*, model, val_data, device, output_dir, spec: AnalyzeRow, **_) -> None:
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


def _run_landscape(*, model, val_data, device, output_dir, spec: AnalyzeRow, hparams, **_) -> None:
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


def _run_fusion_policy(*, module, device, output_dir, spec: AnalyzeRow, **_) -> None:
    td, labels = io.load_fusion_eval(dataset=spec.dataset, seed=spec.seed, device=device)
    r = compute.compute_fusion_policy(module, td, labels)
    io.save_fusion_policy(output_dir, r)


# Defaults match the prior ``ARTIFACTS_BY_MODEL_TYPE`` matrix:
#   vgae/dgi → embeddings + landscape
#   gat      → embeddings + attention + cka + landscape
#   fusion   → fusion_policy only (no embeddings by default)
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


def default_toggles_for(model_type: str) -> dict[str, bool]:
    return {a.name: model_type in a.applies_to for a in ARTIFACTS}


def expected_outputs(spec: AnalyzeRow) -> tuple[str, ...]:
    out: list[str] = []
    for a in ARTIFACTS:
        if getattr(spec, a.name):
            out.append(a.output.format(model_type=spec.model_type))
    return tuple(out)
