"""Class-based operations for the training contract."""

from __future__ import annotations

from typing import Any

from graphids.config import CONFIG_DIR, STAGE_MODEL_MAP

from .models import ContractEnvelope, TrainingSpec


class TrainingContract:
    """Single class owning TrainingSpec contract operations."""

    CONTRACT_NAME = "graphids.training_spec"
    CONTRACT_VERSION = 1

    _STAGES_DIR = CONFIG_DIR / "stages"
    _MODELS_DIR = CONFIG_DIR / "models"
    _FUSION_DIR = CONFIG_DIR / "fusion"

    _CKPT_FLAG_BY_MODEL: dict[str, str] = {
        "vgae": "--data.init_args.vgae_ckpt_path",
        "dgi": "--data.init_args.vgae_ckpt_path",
        "gat": "--data.init_args.gat_ckpt_path",
    }

    @classmethod
    def to_dict(cls, spec: TrainingSpec) -> dict[str, Any]:
        return spec.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingSpec:
        normalized = dict(payload)
        if "config_files" in normalized and isinstance(normalized["config_files"], list):
            normalized["config_files"] = tuple(str(p) for p in normalized["config_files"])
        return TrainingSpec(**normalized)

    @classmethod
    def to_envelope(
        cls,
        spec: TrainingSpec,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContractEnvelope:
        return ContractEnvelope(
            contract=cls.CONTRACT_NAME,
            version=cls.CONTRACT_VERSION,
            payload=cls.to_dict(spec),
            metadata=metadata or {},
        )

    @classmethod
    def _validate_envelope(cls, envelope: ContractEnvelope) -> None:
        if envelope.contract != cls.CONTRACT_NAME:
            raise ValueError(
                f"Unexpected contract {envelope.contract!r}; expected {cls.CONTRACT_NAME!r}"
            )
        if envelope.version != cls.CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract version {envelope.version}; expected {cls.CONTRACT_VERSION}"
            )

    @classmethod
    def from_envelope(cls, envelope_dict: dict[str, Any]) -> TrainingSpec:
        envelope = ContractEnvelope(**envelope_dict)
        cls._validate_envelope(envelope)
        return cls.from_dict(envelope.payload)

    @classmethod
    def normalize_scale(cls, scale: str) -> str:
        if scale not in {"small", "large"}:
            raise ValueError(f"Unsupported scale '{scale}'. Expected: small or large.")
        return scale

    @classmethod
    def resolve_config_files(
        cls,
        stage: str,
        scale: str,
        *,
        model_family: str | None = None,
        fusion_method: str | None = None,
        include_kd_overlay: bool = False,
    ) -> tuple[str, ...]:
        files = [str(cls._STAGES_DIR / f"{stage}.yaml")]

        if stage == "fusion":
            if not fusion_method:
                raise ValueError("fusion_method is required when stage='fusion'")
            files.extend(
                [
                    str(cls._FUSION_DIR / "base.yaml"),
                    # fusion/scales/*.yaml is orchestrator metadata (hidden_dim, batch_size),
                    # not jsonargparse config — excluded from the CLI config chain.
                    str(cls._FUSION_DIR / "methods" / f"{fusion_method}.yaml"),
                ]
            )
            return tuple(files)

        family = model_family or STAGE_MODEL_MAP.get(stage)
        if not family:
            raise ValueError(f"Cannot infer model family for stage '{stage}'.")

        files.extend(
            [
                str(cls._MODELS_DIR / family / "base.yaml"),
                str(cls._MODELS_DIR / family / "scales" / f"{scale}.yaml"),
            ]
        )

        if include_kd_overlay:
            kd_file = cls._MODELS_DIR / family / "kd.yaml"
            if kd_file.exists():
                files.append(str(kd_file))

        return tuple(files)

    @classmethod
    def _cli_scalar(cls, value: Any) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    @classmethod
    def to_override_dict(cls, spec: TrainingSpec) -> dict[str, str]:
        """Convert TrainingSpec to dotted-key override dict for merge_yaml_chain."""
        overrides: dict[str, str] = {
            "data.init_args.dataset": spec.dataset,
            "seed_everything": str(spec.seed),
            "trainer.default_root_dir": spec.run_dir,
        }

        for key, value in spec.model_init_overrides.items():
            overrides[f"model.init_args.{key}"] = cls._cli_scalar(value)

        for upstream_asset, ckpt_path in spec.upstream_ckpt_paths.items():
            model_family = spec.upstream_model_families.get(upstream_asset)
            if not model_family:
                from graphids.log import get_logger

                get_logger(__name__).warning(
                    "unmapped_upstream_asset",
                    asset=upstream_asset,
                    known=list(spec.upstream_model_families),
                )
                continue
            flag = cls._CKPT_FLAG_BY_MODEL.get(model_family)
            if not flag:
                raise KeyError(
                    f"No checkpoint flag for model family {model_family!r}. "
                    f"Add it to TrainingContract._CKPT_FLAG_BY_MODEL."
                )
            overrides[flag.lstrip("-")] = ckpt_path

        runtime = {k: cls._cli_scalar(v) for k, v in spec.runtime_overrides.items()}
        conflicts = set(overrides) & set(runtime)
        if conflicts:
            from graphids.log import get_logger

            get_logger(__name__).warning(
                "runtime_overrides_clobber",
                keys=sorted(conflicts),
                msg="runtime_overrides will overwrite earlier values",
            )
        overrides.update(runtime)
        return overrides
