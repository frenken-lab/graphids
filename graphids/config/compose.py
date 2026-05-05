"""Single composer for all archetypes.

`compose(...)` builds the apex `RowSpec` for any archetype (supervised,
unsupervised, fusion). Archetype variation lives at the call site —
`monitor`, `mode`, `loss`, `run_mode`, `trainer_overrides`, `upstreams`.

`fusion(...)` is a thin convenience wrapper that fills in fusion's
fixed trainer overlay (precision/clip/max_epochs/log_every) and derives
its two upstream ckpts (`vgae`, `focal`) from `meta`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from graphids.config.catalog import best_ckpt
from graphids.config.catalog import run_dir as _run_dir
from graphids.graphids.config.blueprint import ClassPath, RenderedConfig, TrainerCfg
from graphids.graphids.config.lib import fusion_dm
from graphids.graphids.config.row import RowSpec


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
    "log_every_n_steps": 10,
}


def fusion(
    *,
    model: dict[str, Any],
    method: str,
    meta: dict[str, Any],
    monitor: str = "val_acc",
    mode: str = "max",
    trainer_overrides: dict[str, Any] | None = None,
    patience: int = 200,
    batch_size: int = 128,
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
