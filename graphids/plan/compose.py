"""Apex composer — primitives → RowSpec → fit/test/extract row dicts.

Layering (also see ``graphids/plan/__init__.py``)::

    plans/<name>.py::build()              ← you write this
        └── primitives  (spec, GAT, …)    ← leaves          (primitives.py)
              └── compose() / fusion()    ← apex builder    (this file)
                    └── RowSpec.fit/test  ← row emitter     (this file)
                          └── schema.Plan ← typed contract  (schema.py)
                                └── JSON → graphids exec

This module owns:
- ``compose(...)`` / ``fusion(...)``: stitch bare ``{class_path, init_args}``
  blocks + universal trainer/callback overlays into a frozen ``RowSpec``.
- ``RowSpec``: composer's typed return value. Carries ``rendered`` (the
  validated ``RenderedConfig``) plus out-of-band identity bits (``meta``,
  ``resources``, ``upstreams``). ``.fit(name)`` / ``.test(name)`` emit
  ``TrainRow``-shaped dicts.
- ``extract(...)``: one-shot row builder for fusion-feature extraction.
  Doesn't go through ``RowSpec`` (no ``RenderedConfig`` to render).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from graphids.paths import best_ckpt
from graphids.paths import run_dir as _run_dir
from graphids.plan.primitives import fusion_dm
from graphids.plan.schema import ClassPath, RenderedConfig, TrainerCfg

# ============================================================== RowSpec
# Composer output. Not a row yet — call ``.fit()`` / ``.test()``.


@dataclass(frozen=True)
class RowSpec:
    """Composer output. ``rendered`` is a frozen :class:`RenderedConfig` —
    typo'd field access raises ``AttributeError``; constructing one with
    an unknown key raises ``pydantic.ValidationError`` (``extra="forbid"``).
    """

    rendered: RenderedConfig
    meta: dict[str, Any]
    resources: dict[str, Any]
    upstreams: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        required = {"group", "variant", "dataset", "seed", "model_type", "scale"}
        missing = required - set(self.meta)
        if missing:
            raise ValueError(f"RowSpec.meta missing keys: {sorted(missing)}")
        mode = self.resources.get("mode")
        if mode not in {"gpu", "cpu"}:
            raise ValueError(f"RowSpec.resources.mode must be 'gpu'|'cpu', got {mode!r}")

    def fit(self, name: str, *, length: str = "long") -> dict[str, Any]:
        return _emit("fit", name, self, length)

    def test(self, name: str, *, length: str = "long") -> dict[str, Any]:
        return _emit("test", name + "-test", self, length)


def _accelerator_for(mode: str) -> str:
    return "cpu" if mode == "cpu" else "auto"


def _emit(action: str, name: str, spec: RowSpec, length: str) -> dict[str, Any]:
    m = spec.meta
    rendered = spec.rendered.model_dump()
    rendered["trainer"]["accelerator"] = _accelerator_for(spec.resources["mode"])
    return {
        "name": name,
        "action": action,
        "identity": {
            "run_name": f"{m['group']}_{m['variant']}_{m['dataset']}_seed{m['seed']}",
            "run_dir": _run_dir(m["dataset"], m["group"], m["variant"], m["seed"]),
            "jobname": f"{m['model_type']}-{m['scale']}-{m['variant']}",
        },
        "meta": dict(m),
        "rendered_config": rendered,
        "upstreams": list(spec.upstreams),
        "resources": {"mode": spec.resources["mode"], "length": length},
    }


# ============================================================== compose / fusion


def trainer_base() -> dict[str, Any]:
    """Universal trainer defaults. ``callbacks`` filled by the composer."""
    return {
        "accelerator": "auto",
        "devices": "auto",
        "precision": "16-mixed",
        "max_epochs": 300,
        "gradient_clip_val": 1.0,
        "callbacks": [],
    }


def callbacks_base(
    *,
    monitor: str = "val_auroc",
    mode: str = "max",
    patience: int = 100,
    run_dir: str = "",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Universal callbacks trio + ``extras`` merge knob.

    Keys are emitted alphabetically; the composer iterates this dict to
    fill ``trainer.callbacks`` so byte-identical re-renders survive.
    """
    out: dict[str, Any] = {
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
        out.update(extras)
    return out


def compose(
    *,
    model: dict[str, Any],
    data: dict[str, Any],
    meta: dict[str, Any],
    loss: dict[str, Any] | None = None,
    monitor: str = "val_auroc",
    mode: str = "max",
    run_mode: str = "gpu",
    trainer_overrides: dict[str, Any] | None = None,
    upstreams: list[dict[str, Any]] | None = None,
    patience: int = 100,
    callback_extras: dict[str, Any] | None = None,
) -> RowSpec:
    """Compose a training spec from bare ``{class_path, init_args}`` blocks.

    ``model`` / ``data`` are bare blocks (not ``{model: ...}`` wrappers).
    ``loss`` is also a bare block when given; the composer merges it as
    ``model.init_args.loss_fn``. Pass ``loss=None`` for archetypes that
    bake the loss into the model class (DGI, fusion methods).
    """
    rd = _run_dir(meta["dataset"], meta["group"], meta["variant"], meta["seed"])

    model_block = deepcopy(model)
    if loss is not None:
        model_block.setdefault("init_args", {})["loss_fn"] = deepcopy(loss)

    cbs = callbacks_base(
        monitor=monitor, mode=mode, patience=patience, run_dir=rd, extras=callback_extras
    )

    trainer = trainer_base()
    trainer["callbacks"] = [cbs[k] for k in sorted(cbs)]
    trainer["default_root_dir"] = rd
    if trainer_overrides:
        trainer.update(trainer_overrides)

    rendered = RenderedConfig(
        model=ClassPath(**model_block),
        data=ClassPath(**deepcopy(data)),
        callbacks={k: ClassPath(**v) for k, v in cbs.items()},
        trainer=TrainerCfg(**trainer),
        seed_everything=meta["seed"],
    )

    return RowSpec(
        rendered=rendered,
        meta=dict(meta),
        resources={"mode": run_mode},
        upstreams=list(upstreams or []),
    )


_FUSION_TRAINER_OVERLAY: dict[str, Any] = {
    "precision": "32-true",
    "gradient_clip_val": None,
    "max_epochs": 1500,
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 5,
    "reload_dataloaders_every_n_epochs": 1,
}


def fusion(
    *,
    model: dict[str, Any],
    method: str,
    meta: dict[str, Any],
    monitor: str = "val_acc",
    mode: str = "max",
    trainer_overrides: dict[str, Any] | None = None,
    patience: int = 40,
    batch_size: int = 16384,
    episode_sample_size: int = 20_000,
    callback_extras: dict[str, Any] | None = None,
) -> RowSpec:
    """Fusion-archetype convenience wrapper around ``compose``.

    Auto-derives `[vgae, focal]` upstreams from ``meta`` and applies the
    fusion-fixed trainer overlay (callers can still override individual
    keys via ``trainer_overrides``).
    """
    upstreams = [
        {
            "role": "vgae",
            "ckpt_path": best_ckpt(meta["dataset"], "unsupervised", "vgae", meta["seed"]),
            "ckpt_tla": "vgae_ckpt_path",
        },
        {
            "role": "focal",
            "ckpt_path": best_ckpt(meta["dataset"], "gat_loss", "focal", meta["seed"]),
            "ckpt_tla": "gat_ckpt_path",
        },
    ]
    overlay = dict(_FUSION_TRAINER_OVERLAY)
    if trainer_overrides:
        overlay.update(trainer_overrides)
    return compose(
        model=model,
        data=fusion_dm(
            dataset=meta["dataset"],
            seed=meta["seed"],
            method=method,
            batch_size=batch_size,
            episode_sample_size=episode_sample_size,
        ),
        meta=meta,
        monitor=monitor,
        mode=mode,
        run_mode="cpu",
        trainer_overrides=overlay,
        upstreams=upstreams,
        patience=patience,
        callback_extras=callback_extras,
    )


# ============================================================== one-shot row builders


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
    """One-shot per-checkpoint artifact row. plan_id injected by render_plan."""
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
    """One-shot fusion-feature extraction row.

    Doesn't compose a ``RenderedConfig`` — extraction has no Lightning
    trainer, just an upstream-ckpt dict and an output dir.
    """
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
