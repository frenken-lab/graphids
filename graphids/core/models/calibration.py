"""Versioned, partition-isolated calibration; learned parameter shapes are unchanged."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

MODES = ("zero-shot", "conditioned")


def weights_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationBundle:
    checkpoint_hash: str
    source_manifest: str
    reference_policy: str
    mode: str
    track: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    available: tuple[bool, ...]
    thresholds: tuple[float | None, ...]
    normal_bins: int
    forecast_bins: int
    quantile: float = 0.999
    version: int = 2

    def __post_init__(self):
        for field in ("center", "scale"):
            object.__setattr__(
                self, field, tuple(float(x) for x in getattr(self, field))
            )
        object.__setattr__(self, "available", tuple(bool(x) for x in self.available))
        object.__setattr__(
            self,
            "thresholds",
            tuple(None if x is None else float(x) for x in self.thresholds),
        )
        if (
            self.version != 2
            or self.mode not in MODES
            or self.track not in ("source-normal", "target-normal-calibrated")
        ):
            raise ValueError("unsupported calibration identity")
        if self.track == "target-normal-calibrated" and self.mode != "conditioned":
            raise ValueError(
                "target-normal calibration is conditioned-only; zero-shot is source-only"
            )
        if (
            len(self.center) != 3
            or len(self.scale) != 3
            or len(self.available) != 3
            or len(self.thresholds) != 2
        ):
            raise ValueError("invalid calibration dimensions")
        if not np.isfinite((*self.center, *self.scale)).all() or min(self.scale) < 1e-4:
            raise ValueError("invalid calibration normalization")
        if not 0 < self.quantile < 1 or any(
            v is not None and not np.isfinite(v) for v in self.thresholds
        ):
            raise ValueError("invalid calibration thresholds")

    def validate(
        self,
        checkpoint_hash,
        source_manifest,
        reference_policy,
        mode,
        track="source-normal",
    ):
        expected = (checkpoint_hash, source_manifest, reference_policy, mode, track)
        if (
            self.checkpoint_hash,
            self.source_manifest,
            self.reference_policy,
            self.mode,
            self.track,
        ) != expected:
            raise ValueError("calibration bundle identity mismatch")

    def to_dict(self):
        return asdict(self)

    def scores(self, components, masks):
        active = masks.astype(bool) & np.asarray(self.available)
        if not np.isfinite(components[active]).all():
            raise ValueError("nonfinite available forecast component")
        return np.where(active, (components - self.center) / self.scale, 0).sum(-1)

    @classmethod
    def fit(
        cls,
        table,
        *,
        checkpoint_hash,
        source_manifest,
        reference_policy,
        mode,
        quantile=0.999,
        source=None,
    ):
        selected = table.mode_mask(mode)
        roles = table.identities["role"]
        selected &= roles == ("val_query" if source is None else "target_calibration")
        # Manifests use val_query; any other source role must be explicitly rejected.
        normal = (
            selected
            & table.column("normal").astype(bool)
            & (table.column("label") == 0)
        )
        forecast = normal & table.column("anomaly_eligible").astype(bool)
        if not normal.any():
            raise ValueError(f"no verified-normal calibration bins for {mode}")
        components, masks = table.values[:, 5:8], table.values[:, 8:11].astype(bool)
        if source is None:
            center, scale, available = np.zeros(3), np.ones(3), np.zeros(3, dtype=bool)
            for i in range(3):
                values = components[forecast & masks[:, i], i]
                if len(values):
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite calibration component")
                    center[i], scale[i], available[i] = (
                        values.mean(),
                        max(values.std(), 1e-4),
                        True,
                    )
        else:
            source.validate(checkpoint_hash, source_manifest, reference_policy, mode)
            center, scale, available = source.center, source.scale, source.available
        classification = table.column("classification")[normal]
        if not np.isfinite(classification).all():
            raise ValueError("nonfinite classification calibration scores")
        active = masks & np.asarray(available)
        scores = np.where(active, (components - center) / scale, 0).sum(-1)
        valid_forecast = forecast & active.any(-1)
        return cls(
            checkpoint_hash,
            source_manifest,
            reference_policy,
            mode,
            "source-normal" if source is None else "target-normal-calibrated",
            tuple(center),
            tuple(scale),
            tuple(bool(x) for x in available),
            (
                float(np.quantile(classification, quantile)),
                float(np.quantile(scores[valid_forecast], quantile))
                if valid_forecast.any()
                else None,
            ),
            int(normal.sum()),
            int(valid_forecast.sum()),
            quantile,
        )
