"""Pydantic schemas for structural validation of rendered jsonnet configs.

Scope: validation of a rendered dict produced by
``graphids.config.jsonnet.render_config``. Nothing else lives here:

- Recipe envelope Pydantic      → ``graphids.orchestrate.planning.recipes``
- Cross-field orchestration rules → ``graphids.orchestrate.resolve.cross_field``
- Filesystem I/O                → ``graphids.core.io``

``PathContext`` is a pure string-composition frozen model used by both
the resolver and callers that need to build run-dir paths without
touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import CKPT_SUBPATH, COMPLETE_MARKER, LAST_CKPT_SUBPATH


class PathContext(BaseModel):
    """Frozen path model — single source for all run-related paths."""

    model_config = ConfigDict(frozen=True)

    lake_root: str
    user: str
    dataset: str
    model_type: str
    scale: str
    stage: str
    identity: str
    kd_tag: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return Path(
            f"{self.lake_root}/dev/{self.user}/{self.dataset}/"
            f"{self.model_type}_{self.scale}_{self.stage}"
            f"{self.identity}{self.kd_tag}/seed_{self.seed}"
        )

    @property
    def ckpt_file(self) -> Path:
        return self.run_dir / CKPT_SUBPATH

    @property
    def complete_marker(self) -> Path:
        return self.run_dir / COMPLETE_MARKER

    @property
    def last_ckpt_file(self) -> Path:
        return self.run_dir / LAST_CKPT_SUBPATH

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / PurePosixPath(CKPT_SUBPATH).parent


class ClassPathBlock(BaseModel):
    """``{class_path, init_args}`` instantiation block used by data and model."""

    model_config = ConfigDict(extra="forbid")

    class_path: str = Field(..., min_length=1)
    init_args: dict[str, Any] = Field(default_factory=dict)


class TrainerSection(BaseModel):
    """Lightning ``Trainer`` kwargs block (allow extras)."""

    model_config = ConfigDict(extra="allow")

    accelerator: str | None = None
    devices: Any | None = None
    precision: str | int | None = None
    max_epochs: int | None = None
    gradient_clip_val: float | None = None
    log_every_n_steps: int | None = None
    default_root_dir: str | None = None
    logger: bool | list | dict | None = None
    callbacks: list[dict] | None = None


_MODEL_LIST_FIELDS: tuple[str, ...] = ("pool_aggrs", "hidden_dims", "auxiliaries")
_ALLOWED_CLASS_PATH_ROOTS: tuple[str, ...] = (
    "graphids.",
    "lightning.pytorch.",
    "pytorch_lightning.",
)


def _find_callback(callbacks: list[dict], name: str) -> dict | None:
    """Find a callback entry by class name suffix in trainer.callbacks list."""
    for cb in callbacks:
        if isinstance(cb, dict) and name in cb.get("class_path", ""):
            return cb
    return None


class ValidatedConfig(BaseModel):
    """Typed representation of a rendered stage config."""

    model_config = ConfigDict(extra="allow")

    seed_everything: int
    trainer: TrainerSection
    data: ClassPathBlock
    model: ClassPathBlock
    callbacks: dict[str, Any] = Field(default_factory=dict)
    ckpt_path: str | None = None

    @property
    def checkpoint_monitor(self) -> str | None:
        cb = _find_callback(self.trainer.callbacks or [], "ModelCheckpoint")
        return cb["init_args"]["monitor"] if cb else None

    @property
    def checkpoint_mode(self) -> str | None:
        cb = _find_callback(self.trainer.callbacks or [], "ModelCheckpoint")
        return cb["init_args"]["mode"] if cb else None

    @model_validator(mode="after")
    def _no_null_list_fields(self) -> ValidatedConfig:
        null_fields = [
            f
            for f in _MODEL_LIST_FIELDS
            if f in self.model.init_args and self.model.init_args[f] is None
        ]
        if null_fields:
            raise ValueError(
                "model.init_args list fields serialized as null: "
                + ", ".join(null_fields)
                + " — stage jsonnet must emit `[]` or omit the key"
            )
        return self

    @model_validator(mode="after")
    def _monitor_pair_consistent(self) -> ValidatedConfig:
        cbs = self.trainer.callbacks or []
        ckpt = _find_callback(cbs, "ModelCheckpoint")
        es = _find_callback(cbs, "EarlyStopping")
        if not ckpt or not es:
            return self
        ckpt_m = ckpt["init_args"].get("monitor")
        ckpt_mode = ckpt["init_args"].get("mode")
        es_m = es["init_args"].get("monitor")
        es_mode = es["init_args"].get("mode")
        if ckpt_m != es_m or ckpt_mode != es_mode:
            raise ValueError(
                f"ModelCheckpoint ({ckpt_m}/{ckpt_mode}) and "
                f"EarlyStopping ({es_m}/{es_mode}) must track the same metric+mode"
            )
        return self

    @model_validator(mode="after")
    def _lr_monitor_requires_logger(self) -> ValidatedConfig:
        if self.trainer.logger is not False:
            return self
        if _find_callback(self.trainer.callbacks or [], "LearningRateMonitor"):
            raise ValueError(
                "LearningRateMonitor callback requires trainer.logger "
                "to be true; got trainer.logger=false"
            )
        return self

    @model_validator(mode="after")
    def _class_paths_namespaced(self) -> ValidatedConfig:
        for label, block in (("data", self.data), ("model", self.model)):
            if not block.class_path.startswith(_ALLOWED_CLASS_PATH_ROOTS):
                raise ValueError(
                    f"{label}.class_path={block.class_path!r} must start with "
                    f"one of {_ALLOWED_CLASS_PATH_ROOTS}"
                )
        return self


class ConfigValidationError(ValueError):
    """Raised when a rendered jsonnet config fails structural/convention checks."""


def validate_config(rendered: dict[str, Any]) -> ValidatedConfig:
    """Validate a rendered jsonnet config and return the typed view."""
    try:
        return ValidatedConfig.model_validate(rendered)
    except Exception as e:  # pragma: no cover - bubbled with context
        raise ConfigValidationError(str(e)) from e
