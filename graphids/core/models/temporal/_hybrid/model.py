"""Temporal hybrid classifier/anomaly detector."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.core.losses import CrossEntropyLoss
from graphids.core.models.base import classification_test_metrics

from ..base import TemporalModuleBase
from .backbone import TemporalStreamBackbone
from .contexts import TemporalIdMemory, TemporalMotifContext, TemporalRhythmContext
from .encoders import TemporalInputEncoder
from .runtime import HybridRuntimeMixin
from .scan import HybridScanMixin
from .state import HybridStateMixin
from .targets import (
    event_iat,
    event_targets,
    lane_targets,
    payload_delta_dim,
    payload_delta_slice,
    prepare_scan_batch,
)

Objective = Literal["supervised", "anomaly", "joint"]
_ANOMALY_HEADS = ("next_id", "iat", "payload_delta")
_ANOMALY_MODES = {"regression", "nll"}


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


class TemporalHybridModel(HybridRuntimeMixin, HybridScanMixin, HybridStateMixin, TemporalModuleBase):
    """Hybrid classifier/anomaly detector with ID memory and stream backbones."""

    _SCALES: dict[str, dict[str, int]] = {
        "small": {"hidden": 64, "layers": 1, "embedding_dim": 16},
        "large": {"hidden": 128, "layers": 2, "embedding_dim": 32},
    }

    def __init__(
        self,
        *,
        loss_fn: nn.Module | None = None,
        scale: str = "small",
        objective: Objective = "supervised",
        input: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        backbone: dict[str, Any] | None = None,
        heads: dict[str, bool] | None = None,
        anomaly: dict[str, Any] | None = None,
        rhythm: dict[str, Any] | None = None,
        motif: dict[str, Any] | None = None,
        profiling: dict[str, Any] | None = None,
        scan: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        anomaly_score_weights: dict[str, float] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        model_type: str = "temporal_hybrid",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        if objective not in {"supervised", "anomaly", "joint"}:
            raise ValueError("objective must be one of: supervised, anomaly, joint")
        preset = self._SCALES.get(scale, self._SCALES["small"])
        input = _drop_none(dict(input or {}))
        backbone = _drop_none(dict(backbone or {}))
        memory = _drop_none(dict(memory or {}))
        heads = _drop_none(dict(heads or {}))
        anomaly = _drop_none(dict(anomaly or {}))
        rhythm = _drop_none(dict(rhythm or {}))
        motif = _drop_none(dict(motif or {}))
        profiling = _drop_none(dict(profiling or {}))
        scan = _drop_none(dict(scan or {}))
        loss_weights = dict(loss_weights or {})
        anomaly_score_weights = dict(anomaly_score_weights or {})

        input.setdefault("hidden", preset["hidden"])
        input.setdefault("embedding_dim", preset["embedding_dim"])
        input.setdefault("dropout", 0.1)
        backbone.setdefault("type", "gru")
        backbone.setdefault("layers", preset["layers"])
        backbone.setdefault("dropout", input["dropout"])
        memory.setdefault("type", "tgn")
        memory.setdefault("enabled", True)
        memory.setdefault("reset_on_stream_end", True)
        memory.setdefault("use_source", True)
        memory.setdefault("use_destination", True)
        memory.setdefault("time_encoding_dim", 0)
        anomaly.setdefault("mode", "regression")
        anomaly.setdefault("min_log_scale", -7.0)
        anomaly.setdefault("max_log_scale", 5.0)
        rhythm.setdefault("enabled", False)
        motif.setdefault("enabled", False)
        motif.setdefault("length", 3)
        motif.setdefault("embedding_dim", max(2, int(input["embedding_dim"]) // 2))
        motif.setdefault("time_dim", max(2, int(input["embedding_dim"]) // 2))
        profiling.setdefault("phase_timing", False)
        profiling.setdefault("prefix", "profile")
        profiling.setdefault("sync_cuda", True)
        profiling.setdefault("warmup_steps", 5)
        profiling.setdefault("log_every_n_steps", 10)
        scan.setdefault("profile_subphases", False)
        scan.setdefault("rhythm_kernel", "torch")
        scan.setdefault("time_kernel", "torch")
        scan.setdefault("motif_writer", "torch")

        default_ssl = objective in {"anomaly", "joint"}
        heads.setdefault("classification", objective in {"supervised", "joint"})
        heads.setdefault("next_id", default_ssl)
        heads.setdefault("iat", default_ssl)
        heads.setdefault("payload_delta", default_ssl)

        loss_weights.setdefault("classification", 1.0)
        loss_weights.setdefault("next_id", 0.2)
        loss_weights.setdefault("iat", 0.1)
        loss_weights.setdefault("payload_delta", 0.1)
        anomaly_score_weights.setdefault("next_id", 1.0)
        anomaly_score_weights.setdefault("iat", 1.0)
        anomaly_score_weights.setdefault("payload_delta", 1.0)

        self._validate_modular_config(
            objective=objective,
            memory=memory,
            backbone=backbone,
            heads=heads,
            anomaly=anomaly,
            rhythm=rhythm,
            motif=motif,
            scan=scan,
            loss_fn=loss_fn,
        )
        if heads.get("classification", False) and loss_fn is None:
            loss_fn = CrossEntropyLoss()

        super().__init__()
        self.log_binary_score_metrics = objective in {"anomaly", "joint"} or any(
            heads.get(k, False) for k in ("next_id", "iat", "payload_delta")
        )
        self.test_metrics = classification_test_metrics(num_classes)
        self._train_state: dict[str, torch.Tensor | None] | None = None
        self._val_state: dict[str, torch.Tensor | None] | None = None
        self._test_states: dict[int, dict[str, torch.Tensor | None] | None] = {}
        self._val_cls_probs: list[torch.Tensor] = []
        self._val_cls_labels: list[torch.Tensor] = []
        self._val_anom_scores: list[torch.Tensor] = []
        self._val_anom_labels: list[torch.Tensor] = []
        self._init_post(locals())

    @property
    def classification_enabled(self) -> bool:
        return bool(self.hparams.heads.get("classification", False))

    @property
    def anomaly_enabled(self) -> bool:
        heads = self.hparams.heads
        return any(bool(heads.get(k, False)) for k in _ANOMALY_HEADS)

    @staticmethod
    def _enabled_anomaly_heads(heads: dict[str, Any]) -> list[str]:
        return [name for name in _ANOMALY_HEADS if bool(heads.get(name, False))]

    @classmethod
    def _validate_modular_config(
        cls,
        *,
        objective: Objective,
        memory: dict[str, Any],
        backbone: dict[str, Any],
        heads: dict[str, Any],
        anomaly: dict[str, Any],
        rhythm: dict[str, Any],
        motif: dict[str, Any],
        scan: dict[str, Any],
        loss_fn: nn.Module | None,
    ) -> None:
        if str(memory.get("type", "tgn")) != "tgn":
            raise ValueError("memory.type must be 'tgn'")
        if int(memory.get("time_encoding_dim", 0)) < 0:
            raise ValueError("memory.time_encoding_dim must be non-negative")
        if str(backbone.get("type", "gru")) not in {"none", "gru", "ssm_lite", "mamba"}:
            raise ValueError("backbone.type must be one of: none, gru, ssm_lite, mamba")
        if str(anomaly.get("mode", "regression")) not in _ANOMALY_MODES:
            raise ValueError("anomaly.mode must be one of: regression, nll")
        if float(anomaly.get("min_log_scale", -7.0)) > float(anomaly.get("max_log_scale", 5.0)):
            raise ValueError("anomaly.min_log_scale must be <= anomaly.max_log_scale")
        if int(motif.get("length", 1)) < 1:
            raise ValueError("motif.length must be positive")
        if int(motif.get("embedding_dim", 1)) < 1:
            raise ValueError("motif.embedding_dim must be positive")
        if int(motif.get("time_dim", 1)) < 1:
            raise ValueError("motif.time_dim must be positive")
        if scan.get("rhythm_kernel", "torch") not in ("torch", "triton"):
            raise ValueError("scan.rhythm_kernel must be torch or triton")
        for key in ("time_kernel", "motif_writer"):
            if scan.get(key, "torch") not in ("torch", "triton"):
                raise ValueError(f"scan.{key} must be torch or triton")
        if scan.get("motif_writer") == "triton" and scan.get("rhythm_kernel") != "triton":
            raise ValueError("scan.motif_writer=triton requires rhythm_kernel=triton")
        unknown_scan_keys = set(scan) - {"profile_subphases", "rhythm_kernel", "time_kernel", "motif_writer"}
        if unknown_scan_keys:
            keys = ", ".join(sorted(unknown_scan_keys))
            raise ValueError(f"unknown scan config key(s): {keys}")
        classification = bool(heads.get("classification", False))
        anomaly_heads = cls._enabled_anomaly_heads(heads)
        if objective == "supervised":
            if not classification:
                raise ValueError("objective='supervised' requires heads.classification=true")
            return
        if objective == "anomaly":
            if loss_fn is not None:
                raise ValueError("objective='anomaly' trains self-supervised heads and does not accept loss_fn")
            if classification:
                raise ValueError("objective='anomaly' requires heads.classification=false")
            if not anomaly_heads:
                raise ValueError("objective='anomaly' requires at least one anomaly head")
            return
        if objective == "joint":
            if not classification:
                raise ValueError("objective='joint' requires heads.classification=true")
            if not anomaly_heads:
                raise ValueError("objective='joint' requires at least one anomaly head")

    def _build(self) -> None:
        hp = self.hparams
        hidden = int(hp.input["hidden"])
        self.encoder = TemporalInputEncoder(
            num_ids=int(hp.num_ids),
            in_channels=int(hp.in_channels),
            embedding_dim=int(hp.input["embedding_dim"]),
            hidden=hidden,
            dropout=float(hp.input["dropout"]),
        )
        self.memory = (
            TemporalIdMemory(
                num_ids=int(hp.num_ids),
                hidden=hidden,
                use_source=bool(hp.memory.get("use_source", True)),
                use_destination=bool(hp.memory.get("use_destination", True)),
                time_encoding_dim=int(hp.memory.get("time_encoding_dim", 0)),
            )
            if bool(hp.memory.get("enabled", True))
            else None
        )
        self.rhythm = (
            TemporalRhythmContext(num_ids=int(hp.num_ids), hidden=hidden)
            if bool(hp.rhythm.get("enabled", False))
            else None
        )
        self.motif = (
            TemporalMotifContext(
                num_ids=int(hp.num_ids),
                hidden=hidden,
                length=int(hp.motif.get("length", 3)),
                embedding_dim=int(hp.motif.get("embedding_dim", max(2, hidden // 8))),
                time_dim=int(hp.motif.get("time_dim", max(2, hidden // 8))),
            )
            if bool(hp.motif.get("enabled", False))
            else None
        )
        self.backbone = TemporalStreamBackbone(
            backbone_type=str(hp.backbone.get("type", "gru")),
            hidden=hidden,
            layers=int(hp.backbone.get("layers", 1)),
            dropout=float(hp.backbone.get("dropout", hp.input["dropout"])),
        )
        self.classifier = nn.Linear(hidden, int(hp.num_classes)) if self.classification_enabled else None
        self.next_id_head = nn.Linear(hidden, int(hp.num_ids)) if hp.heads.get("next_id", False) else None
        anomaly_mode = str(hp.anomaly.get("mode", "regression"))
        if hp.heads.get("iat", False):
            self.iat_head = (
                nn.Linear(hidden, 2)
                if anomaly_mode == "nll"
                else nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
            )
        else:
            self.iat_head = None
        payload_dim = self._payload_delta_dim(int(hp.in_channels))
        if hp.heads.get("payload_delta", False):
            out_dim = payload_dim * 2 if anomaly_mode == "nll" else payload_dim
            self.payload_delta_head = nn.Linear(hidden, out_dim)
        else:
            self.payload_delta_head = None
        self.test_metrics = classification_test_metrics(int(hp.num_classes))

    @staticmethod
    def _rebuild_excluded_kwargs(hp: dict) -> dict:
        if hp.get("objective") == "anomaly":
            return {}
        if dict(hp.get("heads") or {}).get("classification", False):
            from graphids.core.losses.build import build_loss

            return {"loss_fn": build_loss("temporal_hybrid", hp.get("loss_config"))}
        return {}

    @staticmethod
    def _payload_delta_slice(in_channels: int) -> slice:
        return payload_delta_slice(in_channels)

    @classmethod
    def _payload_delta_dim(cls, in_channels: int) -> int:
        return payload_delta_dim(in_channels)

    def _lane_targets(self, batch) -> dict[str, torch.Tensor]:
        return lane_targets(batch, in_channels=int(self.hparams.in_channels))

    def _targets(self, batch) -> dict[str, torch.Tensor]:
        if getattr(batch, "valid_mask", None) is not None:
            return self._lane_targets(batch)
        return event_targets(batch, in_channels=int(self.hparams.in_channels))

    def _event_iat(self, batch, idx: int, ref: torch.Tensor) -> torch.Tensor:
        return event_iat(batch, idx, ref, in_channels=int(self.hparams.in_channels))

    def _prepare_scan_batch(self, batch, ref: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return prepare_scan_batch(
            batch,
            ref,
            num_ids=int(self.hparams.num_ids),
            in_channels=int(self.hparams.in_channels),
        )

    def _heads_from_features(self, features: torch.Tensor, batch) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"features": features}
        if self.classifier is not None:
            out["logits"] = self.classifier(features)
        if self.next_id_head is not None:
            out["next_id_logits"] = self.next_id_head(features)
        if self.iat_head is not None:
            iat_raw = self.iat_head(features)
            if str(self.hparams.anomaly.get("mode", "regression")) == "nll":
                out["iat_loc"] = iat_raw[:, 0]
                out["iat_log_scale"] = self._clamped_log_scale(iat_raw[:, 1])
            else:
                out["iat_pred"] = iat_raw.squeeze(-1)
        if self.payload_delta_head is not None:
            payload_raw = self.payload_delta_head(features)
            if str(self.hparams.anomaly.get("mode", "regression")) == "nll":
                loc, log_scale = payload_raw.chunk(2, dim=-1)
                out["payload_delta_loc"] = loc
                out["payload_delta_log_scale"] = self._clamped_log_scale(log_scale)
            else:
                out["payload_delta_pred"] = payload_raw
        if self.anomaly_enabled:
            components = self._anomaly_components(out, batch)
            out.update(components)
            out["anomaly_score"] = self._weighted_anomaly_score(components)
        return out

    def forward_temporal(self, batch, state=None) -> dict[str, torch.Tensor]:
        out, _state = self._forward_with_state(batch, self._detach_state(state))
        return out

    def forward(self, batch) -> dict[str, torch.Tensor]:
        return self.forward_temporal(batch)

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch) -> torch.Tensor:
        return self.loss_fn(logits, labels, graph=batch)

    def _clamped_log_scale(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.clamp(
            min=float(self.hparams.anomaly.get("min_log_scale", -7.0)),
            max=float(self.hparams.anomaly.get("max_log_scale", 5.0)),
        )

    @staticmethod
    def _gaussian_nll(target: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        inv_scale = torch.exp(-log_scale)
        return 0.5 * ((target - loc) * inv_scale).pow(2) + log_scale + 0.5 * torch.log(
            target.new_tensor(2.0 * torch.pi)
        )

    def _anomaly_components(self, out: dict[str, torch.Tensor], batch) -> dict[str, torch.Tensor]:
        targets = self._targets(batch)
        valid_next = targets["valid_next"]
        components: dict[str, torch.Tensor] = {"valid_next": valid_next}
        if "next_id_logits" in out:
            nll = F.cross_entropy(out["next_id_logits"], targets["next_id"], reduction="none")
            components["next_id_nll"] = torch.where(valid_next, nll, torch.zeros_like(nll))
        if "iat_loc" in out and "iat_log_scale" in out:
            target = torch.log1p(targets["iat"].float())
            nll = self._gaussian_nll(target, out["iat_loc"], out["iat_log_scale"])
            components["iat_nll"] = torch.where(valid_next, nll, torch.zeros_like(nll))
        elif "iat_pred" in out:
            err = F.smooth_l1_loss(out["iat_pred"], targets["iat"], reduction="none")
            components["iat_error"] = torch.where(valid_next, err, torch.zeros_like(err))
        if "payload_delta_loc" in out and "payload_delta_log_scale" in out:
            nll = self._gaussian_nll(
                targets["payload_delta"].float(),
                out["payload_delta_loc"],
                out["payload_delta_log_scale"],
            ).mean(dim=-1)
            components["payload_delta_nll"] = torch.where(valid_next, nll, torch.zeros_like(nll))
        elif "payload_delta_pred" in out:
            err = F.smooth_l1_loss(
                out["payload_delta_pred"],
                targets["payload_delta"],
                reduction="none",
            ).mean(dim=-1)
            components["payload_delta_error"] = torch.where(valid_next, err, torch.zeros_like(err))
        return components

    def _weighted_anomaly_score(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        score: torch.Tensor | None = None
        weights = self.hparams.anomaly_score_weights
        mapping = {
            "next_id": "next_id_nll",
            "iat": "iat_nll" if "iat_nll" in components else "iat_error",
            "payload_delta": (
                "payload_delta_nll" if "payload_delta_nll" in components else "payload_delta_error"
            ),
        }
        for name, key in mapping.items():
            if key not in components:
                continue
            weighted = components[key] * float(weights.get(name, 1.0))
            score = weighted if score is None else score + weighted
        if score is None:
            raise RuntimeError("anomaly score requested but no anomaly heads are enabled")
        return score

    def _loss_terms(self, out: dict[str, torch.Tensor], batch) -> dict[str, torch.Tensor]:
        mask = self.scored_mask(batch)
        labels_all = batch.y.reshape(-1)
        terms: dict[str, torch.Tensor] = {}
        weights = self.hparams.loss_weights
        if self.classification_enabled and "logits" in out and mask.any():
            labels = labels_all[mask].long()
            terms["classification"] = self._classification_loss(out["logits"][mask], labels, batch)
        if self.anomaly_enabled:
            valid = out["valid_next"].bool() & mask
            if valid.any():
                if "next_id_nll" in out:
                    terms["next_id"] = out["next_id_nll"][valid].mean()
                if "iat_nll" in out:
                    terms["iat"] = out["iat_nll"][valid].mean()
                elif "iat_error" in out:
                    terms["iat"] = out["iat_error"][valid].mean()
                if "payload_delta_nll" in out:
                    terms["payload_delta"] = out["payload_delta_nll"][valid].mean()
                elif "payload_delta_error" in out:
                    terms["payload_delta"] = out["payload_delta_error"][valid].mean()
        if not terms:
            terms["zero"] = out["features"].sum() * 0.0
        terms["total"] = sum(
            value * float(weights.get(name, 1.0))
            for name, value in terms.items()
            if name != "total"
        )
        return terms
