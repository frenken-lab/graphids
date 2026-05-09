"""Row schemas + row builders — the full row contract in one module.

Schema types (``TrainRow``, ``Plan``, etc.) are consumed by orchestrate,
the CLI, and slurm/submit. Row builders (``fit_row``, ``test_row``, etc.)
are the public API for plan authors.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.paths import run_dir as _run_dir
from graphids.plan.primitives import DataCfg, LossFn, ModelCfg

# ── Schema types ──────────────────────────────────────────────────────────────


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Identity(_StrictModel):
    run_name: str
    run_dir: str
    jobname: str


class Meta(_StrictModel):
    group: str
    variant: str
    dataset: str
    seed: int
    model_type: str
    scale: str
    subdir: str = "ablations"


class Upstream(_StrictModel):
    role: str
    ckpt_path: str
    ckpt_tla: str


class Resources(_StrictModel):
    mode: Literal["gpu", "cpu"]
    length: Literal["short", "long"]


class TrainRow(_StrictModel):
    """Fit or test row — carries fully-typed training configs.

    ``plan_module`` + ``git_sha`` are the reproduction contract; together
    with ``plan_id`` and ``meta.dataset`` / ``meta.seed`` they let
    ``git checkout <git_sha> && graphids run <plan_module>`` regenerate
    this row deterministically.
    """

    name: str
    action: Literal["fit", "test"]
    plan_id: str
    plan_module: str
    git_sha: str
    identity: Identity
    meta: Meta
    model: ModelCfg
    loss_fn: LossFn | None = None
    data: DataCfg
    trainer: dict[str, Any]
    callbacks: list[dict[str, Any]]
    seed_everything: int
    upstreams: list[Upstream] = Field(default_factory=list)
    resources: Resources


class CacheRow(_StrictModel):
    name: str
    action: Literal["cache"]
    plan_id: str
    dataset: str
    vocab_scope: Literal["train", "all"] = "train"
    seed: int = 42
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2
    resources: Resources


class ExtractRow(_StrictModel):
    name: str
    action: Literal["extract"]
    plan_id: str
    dataset: str
    extractor_ckpts: dict[str, str]
    output_dir: str
    resources: Resources
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    batch_size: int = 256
    seed: int = 42
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2


class AnalyzeRow(_StrictModel):
    name: str
    action: Literal["analyze"]
    plan_id: str
    resources: Resources

    ckpt_path: str
    dataset: str
    model_type: Literal["vgae", "dgi", "gat", "fusion"]
    output_dir: str
    lake_root: str

    embeddings: bool = True
    attention: bool = False
    cka: bool = False
    landscape: bool = False
    fusion_policy: bool = False

    cka_teacher_ckpt: str = ""
    cka_max_samples: int = 500

    landscape_resolution: int = 51
    landscape_scale: float = 1.0
    landscape_max_graphs: int = 500

    embedding_max_samples: int = 2000
    attention_max_samples: int = 50

    window_size: int = 100
    stride: int = 100
    batch_size: int = 256
    seed: int = 42
    vocab_scope: str = "train"

    vgae_ckpt_path: str = ""
    gat_ckpt_path: str = ""

    @model_validator(mode="after")
    def _validate_conditional_deps(self) -> AnalyzeRow:
        if self.cka and not self.cka_teacher_ckpt:
            raise ValueError("cka=true requires cka_teacher_ckpt")
        if self.cka and self.model_type != "gat":
            raise ValueError(
                f"cka=true only supported for model_type='gat', got {self.model_type!r}"
            )
        if self.fusion_policy and not self.vgae_ckpt_path:
            raise ValueError("fusion_policy=true requires vgae_ckpt_path")
        if self.fusion_policy and not self.gat_ckpt_path:
            raise ValueError("fusion_policy=true requires gat_ckpt_path")
        return self


class HFPushRow(_StrictModel):
    name: str
    action: Literal["hf_push"]
    plan_id: str
    artifact_type: Literal["checkpoints", "cache", "states", "logs", "analysis"]
    repo_id: str
    repo_type: Literal["model", "dataset"] = "model"
    local_path: str
    path_in_repo: str
    private: bool = True
    resources: Resources


Row = Annotated[
    TrainRow | CacheRow | ExtractRow | AnalyzeRow | HFPushRow,
    Field(discriminator="action"),
]


class Plan(_StrictModel):
    plan_id: str
    plan_module: str
    plan_args: dict[str, Any]
    created_at: str
    rows: list[Row]

    def __iter__(self):  # type: ignore[override]
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Row:
        return self.rows[i]


# ── Row builder internals ─────────────────────────────────────────────────────

FUSION_TRAINER: dict[str, Any] = {
    "precision": "32-true",
    "gradient_clip_val": None,
    "max_epochs": 1500,
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 5,
    "reload_dataloaders_every_n_epochs": 1,
}

_TRAINER_DEFAULTS: dict[str, Any] = {
    "accelerator": "auto",
    "devices": "auto",
    "precision": "16-mixed",
    "max_epochs": 300,
    "gradient_clip_val": 1.0,
}


def _make_callbacks(
    *, monitor: str, mode: str, patience: int, run_dir: str, extras: dict | None = None
) -> list[dict[str, Any]]:
    cbs: dict[str, Any] = {
        "checkpoint": {
            "class_path": "graphids.core.callbacks.Sha256ModelCheckpoint",
            "init_args": {
                "monitor": monitor,
                "mode": mode,
                "save_top_k": 1,
                "save_last": True,
                "dirpath": run_dir + "/checkpoints",
                "filename": "best_model",
            },
        },
        "early_stopping": {
            "class_path": "lightning.pytorch.callbacks.EarlyStopping",
            "init_args": {"monitor": monitor, "mode": mode, "patience": patience},
        },
        "learning_rate_monitor": {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
            "init_args": {"logging_interval": "epoch"},
        },
        "mlflow": {
            "class_path": "graphids._mlflow.MLflowTrainingCallback",
            "init_args": {},
        },
    }
    if extras:
        cbs.update(extras)
    return [cbs[k] for k in sorted(cbs)]


# ── Public row builders ───────────────────────────────────────────────────────


def fit_row(
    name: str,
    *,
    model: ModelCfg,
    data: DataCfg,
    meta: dict[str, Any],
    loss: LossFn | None = None,
    monitor: str = "val_auroc",
    mode: str = "max",
    run_mode: str = "gpu",
    trainer_overrides: dict[str, Any] | None = None,
    patience: int = 100,
    upstreams: list[dict[str, Any]] | None = None,
    callback_extras: dict[str, Any] | None = None,
    length: str = "long",
) -> dict[str, Any]:
    rd = _run_dir(
        meta["dataset"],
        meta["group"],
        meta["variant"],
        meta["seed"],
        subdir=meta.get("subdir", "ablations"),
    )
    trainer = {
        **_TRAINER_DEFAULTS,
        "default_root_dir": rd,
        "accelerator": "cpu" if run_mode == "cpu" else "auto",
    }
    if trainer_overrides:
        trainer.update(trainer_overrides)
    return {
        "name": name,
        "action": "fit",
        "identity": {
            "run_name": f"{meta['group']}_{meta['variant']}_{meta['dataset']}_seed{meta['seed']}",
            "run_dir": rd,
            "jobname": f"{meta['model_type']}-{meta['scale']}-{meta['variant']}",
        },
        "meta": dict(meta),
        "model": model.model_dump(),
        "loss_fn": loss.model_dump() if loss is not None else None,
        "data": data.model_dump(),
        "trainer": trainer,
        "callbacks": _make_callbacks(
            monitor=monitor, mode=mode, patience=patience, run_dir=rd, extras=callback_extras
        ),
        "seed_everything": meta["seed"],
        "upstreams": list(upstreams or []),
        "resources": {"mode": run_mode, "length": length},
    }


def test_row(
    name: str,
    *,
    model: ModelCfg,
    data: DataCfg,
    meta: dict[str, Any],
    loss: LossFn | None = None,
    monitor: str = "val_auroc",
    mode: str = "max",
    run_mode: str = "gpu",
    trainer_overrides: dict[str, Any] | None = None,
    patience: int = 100,
    upstreams: list[dict[str, Any]] | None = None,
    callback_extras: dict[str, Any] | None = None,
    length: str = "long",
) -> dict[str, Any]:
    row = fit_row(
        name + "-test",
        model=model,
        data=data,
        meta=meta,
        loss=loss,
        monitor=monitor,
        mode=mode,
        run_mode=run_mode,
        trainer_overrides=trainer_overrides,
        patience=patience,
        upstreams=upstreams,
        callback_extras=callback_extras,
        length=length,
    )
    row["action"] = "test"
    return row


# ── One-shot ops row builders ─────────────────────────────────────────────────


def extract(
    *,
    name: str,
    dataset: str,
    extractor_ckpts: dict[str, str],
    output_dir: str,
    mode: str = "gpu",
    length: str = "short",
    max_samples: int = 150_000,
    max_val_samples: int = 30_000,
    batch_size: int = 256,
    seed: int = 42,
    window_size: int = 100,
    stride: int = 100,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    return {
        "name": name,
        "action": "extract",
        "dataset": dataset,
        "extractor_ckpts": dict(extractor_ckpts),
        "output_dir": output_dir,
        "resources": {"mode": mode, "length": length},
        "max_samples": max_samples,
        "max_val_samples": max_val_samples,
        "batch_size": batch_size,
        "seed": seed,
        "window_size": window_size,
        "stride": stride,
        "val_fraction": val_fraction,
    }


def analyze(
    *,
    name: str,
    ckpt_path: str,
    dataset: str,
    model_type: str,
    output_dir: str,
    lake_root: str,
    embeddings: bool = True,
    attention: bool = False,
    cka: bool = False,
    cka_teacher_ckpt: str = "",
    landscape: bool = False,
    landscape_resolution: int = 51,
    fusion_policy: bool = False,
    vgae_ckpt_path: str = "",
    gat_ckpt_path: str = "",
    mode: str = "gpu",
    length: str = "short",
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "name": name,
        "action": "analyze",
        "ckpt_path": ckpt_path,
        "dataset": dataset,
        "model_type": model_type,
        "output_dir": output_dir,
        "lake_root": lake_root,
        "embeddings": embeddings,
        "attention": attention,
        "cka": cka,
        "cka_teacher_ckpt": cka_teacher_ckpt,
        "landscape": landscape,
        "landscape_resolution": landscape_resolution,
        "fusion_policy": fusion_policy,
        "vgae_ckpt_path": vgae_ckpt_path,
        "gat_ckpt_path": gat_ckpt_path,
        "resources": {"mode": mode, "length": length},
        "seed": seed,
    }


def hf_push(
    *,
    name: str,
    artifact_type: str,
    repo_id: str,
    repo_type: str = "model",
    local_path: str,
    path_in_repo: str,
    private: bool = True,
    length: str = "short",
) -> dict[str, Any]:
    return {
        "name": name,
        "action": "hf_push",
        "artifact_type": artifact_type,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_path": local_path,
        "path_in_repo": path_in_repo,
        "private": private,
        "resources": {"mode": "cpu", "length": length},
    }
