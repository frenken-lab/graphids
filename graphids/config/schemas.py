"""Pydantic schemas for structural validation of rendered jsonnet configs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MODEL_LIST_FIELDS = ("pool_aggrs", "hidden_dims", "auxiliaries")
_ALLOWED_CLASS_PATH_ROOTS = ("graphids.",)


class ClassPathBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = Field(..., min_length=1)
    init_args: dict[str, Any] = Field(default_factory=dict)


class TrainerSection(BaseModel):
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


def _monitor_pair(block: ClassPathBlock, label: str) -> tuple[str, str]:
    args = block.init_args
    monitor, mode = args.get("monitor"), args.get("mode")
    if not monitor or mode not in ("min", "max"):
        raise ValueError(f"{label} init_args must set monitor and mode ∈ {{min, max}}")
    return monitor, mode


class CallbacksSection(BaseModel):
    """Forced-callback block. Cross-field validator keeps
    ``checkpoint`` and ``early_stopping`` tracking the same
    ``(monitor, mode)`` so a best-epoch ckpt matches the stop trigger.
    """

    model_config = ConfigDict(extra="allow")
    checkpoint: ClassPathBlock
    early_stopping: ClassPathBlock

    @model_validator(mode="after")
    def _monitor_pair_consistent(self) -> CallbacksSection:
        ckpt = _monitor_pair(self.checkpoint, "callbacks.checkpoint")
        es = _monitor_pair(self.early_stopping, "callbacks.early_stopping")
        if ckpt != es:
            raise ValueError(
                f"ModelCheckpoint ({ckpt[0]}/{ckpt[1]}) and "
                f"EarlyStopping ({es[0]}/{es[1]}) must track the same metric+mode"
            )
        return self


class ConfigValidationError(ValueError):
    """Raised by :func:`validate_config` when the rendered dict fails
    any Pydantic check. Wraps Pydantic's ``ValidationError`` so callers
    can catch one exception type regardless of which rule fired.
    """


class ValidatedConfig(BaseModel):
    """The rendered-config contract. ``extra="forbid"`` at the top
    level catches typos in ``configs/ablations/*.jsonnet``; nested
    ``TrainerSection`` stays permissive so Lightning trainer kwargs
    pass through unchecked.
    """

    model_config = ConfigDict(extra="forbid")
    seed_everything: int
    trainer: TrainerSection
    data: ClassPathBlock
    model: ClassPathBlock
    callbacks: CallbacksSection

    @property
    def checkpoint_monitor(self) -> str:
        return self.callbacks.checkpoint.init_args["monitor"]

    @property
    def checkpoint_mode(self) -> str:
        return self.callbacks.checkpoint.init_args["mode"]

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
    def _lr_monitor_requires_logger(self) -> ValidatedConfig:
        if self.trainer.logger is not False:
            return self
        for cb in self.trainer.callbacks or []:
            if isinstance(cb, dict) and "LearningRateMonitor" in cb.get("class_path", ""):
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

    @model_validator(mode="after")
    def _monitor_pair_matches_stage_family(self) -> ValidatedConfig:
        """Gate checkpoint/monitor mode against the stage-family convention.

        Fusion stages track ``val_acc/max``; every other family tracks
        ``val_loss/min``. Mismatches are a silent-bug magnet (training
        completes but ModelCheckpoint picks the wrong epoch) so we fail
        loudly at validate-time instead of warning at resolve-time.
        Family is derived from ``model.class_path``: anything under
        ``graphids.core.models.fusion`` is the fusion stage.
        """
        is_fusion = ".models.fusion" in self.model.class_path
        exp_monitor, exp_mode = ("val_acc", "max") if is_fusion else ("val_loss", "min")
        got_monitor = self.checkpoint_monitor
        got_mode = self.checkpoint_mode
        if got_monitor != exp_monitor or got_mode != exp_mode:
            family = "fusion" if is_fusion else "supervised/unsupervised"
            raise ValueError(
                f"{family} stages must track {exp_monitor}/{exp_mode}; "
                f"got {got_monitor}/{got_mode} "
                f"(model.class_path={self.model.class_path!r})"
            )
        return self


def validate_config(rendered: dict[str, Any]) -> ValidatedConfig:
    """Validate a rendered jsonnet dict before any torch import.

    Called by ``ResolvedConfig.from_rendered`` immediately after
    :func:`graphids.config.jsonnet.render`. Fails fast on null list
    fields, monitor/mode mismatches, un-namespaced ``class_path`` strings,
    and ``LearningRateMonitor`` without a logger.

    Raises:
        ConfigValidationError: any validation failure, with the Pydantic
            error text preserved as the message.
    """
    try:
        return ValidatedConfig.model_validate(rendered)
    except Exception as e:
        raise ConfigValidationError(str(e)) from e
