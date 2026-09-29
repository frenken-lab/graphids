"""Composable temporal hybrid model for CAN event streams."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.core.losses import CrossEntropyLoss
from graphids.core.models.base import classification_test_metrics

from .base import TemporalModuleBase

Objective = Literal["supervised", "anomaly", "joint"]
_ANOMALY_HEADS = ("next_id", "iat", "payload_delta")
_ANOMALY_MODES = {"regression", "nll"}


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


class TemporalInputEncoder(nn.Module):
    """Encode event bytes/features plus source and destination ID embeddings."""

    def __init__(
        self,
        *,
        num_ids: int,
        in_channels: int,
        embedding_dim: int,
        hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(max(1, int(num_ids)), int(embedding_dim))
        self.dst_embedding = nn.Embedding(max(1, int(num_ids)), int(embedding_dim))
        input_dim = int(in_channels) + (2 * int(embedding_dim))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, batch) -> torch.Tensor:
        src = batch.src.clamp_min(0).clamp_max(self.src_embedding.num_embeddings - 1).long()
        dst = batch.dst.clamp_min(0).clamp_max(self.dst_embedding.num_embeddings - 1).long()
        x = torch.cat([batch.msg.float(), self.src_embedding(src), self.dst_embedding(dst)], dim=-1)
        return self.proj(x)


class TemporalTimeEncoder(nn.Module):
    """Sinusoidal encoding for positive elapsed times."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = max(1, int(dim))
        half = max(1, (self.dim + 1) // 2)
        self.register_buffer("freqs", torch.logspace(0, 3, steps=half), persistent=False)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(delta.float().clamp_min(0)).unsqueeze(-1) / self.freqs
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return encoded[..., : self.dim]


class TemporalIdMemory(nn.Module):
    """TGN-style per-ID memory updated causally after each event."""

    def __init__(
        self,
        *,
        num_ids: int,
        hidden: int,
        use_source: bool = True,
        use_destination: bool = True,
        time_encoding_dim: int = 0,
    ):
        super().__init__()
        self.num_ids = max(1, int(num_ids))
        self.hidden = int(hidden)
        self.use_source = bool(use_source)
        self.use_destination = bool(use_destination)
        self.time_encoding_dim = max(0, int(time_encoding_dim))
        self.update_cell = nn.GRUCell(self.hidden, self.hidden)
        self.time_encoder = (
            TemporalTimeEncoder(self.time_encoding_dim) if self.time_encoding_dim > 0 else None
        )
        context_dim = (self.hidden if self.use_source else 0) + (self.hidden if self.use_destination else 0)
        context_dim += 2 * self.time_encoding_dim
        self._has_context = context_dim > 0
        self.mix = nn.Sequential(
            nn.Linear(max(1, context_dim), self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
        )

    def initial_state(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.num_ids, self.hidden, device=device, dtype=dtype)

    def ensure_state(self, state: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if state is None or state.device != ref.device or state.dtype != ref.dtype:
            return self.initial_state(device=ref.device, dtype=ref.dtype)
        if state.shape != (self.num_ids, self.hidden):
            return self.initial_state(device=ref.device, dtype=ref.dtype)
        return state

    def initial_last_seen(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((self.num_ids,), -1.0, device=device, dtype=dtype)

    def ensure_last_seen(self, last_seen: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if last_seen is None or last_seen.device != ref.device or last_seen.dtype != ref.dtype:
            return self.initial_last_seen(device=ref.device, dtype=ref.dtype)
        if last_seen.shape != (self.num_ids,):
            return self.initial_last_seen(device=ref.device, dtype=ref.dtype)
        return last_seen

    def read_context(
        self,
        state: torch.Tensor,
        last_seen: torch.Tensor | None,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        src_idx = src.clamp_min(0).clamp_max(self.num_ids - 1).long()
        dst_idx = dst.clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._read_context_indexed(state, last_seen, src_idx, dst_idx, t)

    def _read_context_indexed(
        self,
        state: torch.Tensor,
        last_seen: torch.Tensor | None,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._has_context:
            return state.new_zeros(self.hidden)
        parts: list[torch.Tensor] = []
        if self.use_source:
            parts.append(state[src_idx])
        if self.use_destination:
            parts.append(state[dst_idx])
        if self.time_encoder is not None:
            assert last_seen is not None
            now = state.new_tensor(0.0) if t is None else t.to(device=state.device, dtype=state.dtype)
            src_last = last_seen[src_idx]
            dst_last = last_seen[dst_idx]
            src_elapsed = torch.where(src_last >= 0, now - src_last, torch.zeros_like(now))
            dst_elapsed = torch.where(dst_last >= 0, now - dst_last, torch.zeros_like(now))
            parts.append(self.time_encoder(src_elapsed).reshape(-1))
            parts.append(self.time_encoder(dst_elapsed).reshape(-1))
        return self.mix(torch.cat(parts, dim=-1))

    def _read_context_rows(
        self,
        src_row: torch.Tensor,
        dst_row: torch.Tensor,
        src_last_seen: torch.Tensor | None,
        dst_last_seen: torch.Tensor | None,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._has_context:
            return src_row.new_zeros(self.hidden)
        parts: list[torch.Tensor] = []
        if self.use_source:
            parts.append(src_row)
        if self.use_destination:
            parts.append(dst_row)
        if self.time_encoder is not None:
            assert src_last_seen is not None
            assert dst_last_seen is not None
            now = src_row.new_tensor(0.0) if t is None else t.to(device=src_row.device, dtype=src_row.dtype)
            src_elapsed = torch.where(src_last_seen >= 0, now - src_last_seen, torch.zeros_like(now))
            dst_elapsed = torch.where(dst_last_seen >= 0, now - dst_last_seen, torch.zeros_like(now))
            parts.append(self.time_encoder(src_elapsed).reshape(-1))
            parts.append(self.time_encoder(dst_elapsed).reshape(-1))
        return self.mix(torch.cat(parts, dim=-1))

    def _updated_rows(
        self,
        src_row: torch.Tensor,
        dst_row: torch.Tensor,
        message: torch.Tensor,
        *,
        same_id: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_new = self.update_cell(message.unsqueeze(0), src_row.unsqueeze(0)).squeeze(0)
        dst_input = src_new if same_id else dst_row
        dst_new = self.update_cell(message.unsqueeze(0), dst_input.unsqueeze(0)).squeeze(0)
        return src_new, dst_new

    def update_one(
        self,
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        src_idx = src.to(device=state.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        dst_idx = dst.to(device=state.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._update_one_indexed(state, src_idx, dst_idx, message)

    def _update_one_indexed(
        self,
        state: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        src_new = self.update_cell(message.unsqueeze(0), state[src_idx].unsqueeze(0)).squeeze(0)
        after_src = state.clone()
        after_src[src_idx] = src_new

        dst_new = self.update_cell(message.unsqueeze(0), after_src[dst_idx].unsqueeze(0)).squeeze(0)
        updated = after_src.clone()
        updated[dst_idx] = dst_new
        return updated

    def update_last_seen(
        self,
        last_seen: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if t is None:
            return last_seen
        src_idx = src.to(device=last_seen.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        dst_idx = dst.to(device=last_seen.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._update_last_seen_indexed(last_seen, src_idx, dst_idx, t)

    def _update_last_seen_indexed(
        self,
        last_seen: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if t is None:
            return last_seen
        now = t.to(device=last_seen.device, dtype=last_seen.dtype)
        updated = last_seen.clone()
        updated[src_idx] = now
        updated[dst_idx] = now
        return updated


class TemporalRhythmContext(nn.Module):
    """Causal rolling IAT summaries for source and destination IDs."""

    def __init__(self, *, num_ids: int, hidden: int):
        super().__init__()
        self.num_ids = max(1, int(num_ids))
        self.hidden = int(hidden)
        self.proj = nn.Sequential(
            nn.Linear(6, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
        )

    def initial_state(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.num_ids, 3, device=device, dtype=dtype)

    def ensure_state(self, state: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if state is None or state.device != ref.device or state.dtype != ref.dtype:
            return self.initial_state(device=ref.device, dtype=ref.dtype)
        if state.shape != (self.num_ids, 3):
            return self.initial_state(device=ref.device, dtype=ref.dtype)
        return state

    def read_context(self, state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        src_idx = src.clamp_min(0).clamp_max(self.num_ids - 1).long()
        dst_idx = dst.clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._read_context_indexed(state, src_idx, dst_idx)

    def _read_context_indexed(
        self,
        state: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self._read_context_rows(state[src_idx], state[dst_idx])

    def _read_context_rows(self, src_row: torch.Tensor, dst_row: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self._features(src_row), self._features(dst_row)], dim=-1))

    @staticmethod
    def _features(stats: torch.Tensor) -> torch.Tensor:
        count = stats[0].clamp_min(0)
        mean = stats[1]
        variance = torch.where(count > 1, stats[2] / (count - 1).clamp_min(1), torch.zeros_like(count))
        return torch.stack([torch.log1p(count), mean, torch.sqrt(variance.clamp_min(0))])

    def update_one(self, state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, iat: torch.Tensor) -> torch.Tensor:
        src_idx = src.to(device=state.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        dst_idx = dst.to(device=state.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._update_one_indexed(state, src_idx, dst_idx, iat)

    def _update_one_indexed(
        self,
        state: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        iat: torch.Tensor,
    ) -> torch.Tensor:
        updated = self._update_id_indexed(state, src_idx, iat)
        return self._update_id_indexed(updated, dst_idx, iat)

    def _update_id(self, state: torch.Tensor, idx: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        pos = idx.to(device=state.device).clamp_min(0).clamp_max(self.num_ids - 1).long()
        return self._update_id_indexed(state, pos, value)

    def _update_id_indexed(self, state: torch.Tensor, pos: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        replacement = self._updated_row(state[pos], value)
        updated = state.clone()
        updated[pos] = replacement
        return updated

    def _updated_row(self, old: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        val = value.to(device=old.device, dtype=old.dtype).clamp_min(0)
        count = old[0] + 1.0
        delta = val - old[1]
        mean = old[1] + (delta / count)
        m2 = old[2] + delta * (val - mean)
        return torch.stack([count, mean, m2])

    def _updated_rows(
        self,
        src_row: torch.Tensor,
        dst_row: torch.Tensor,
        value: torch.Tensor,
        *,
        same_id: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_new = self._updated_row(src_row, value)
        dst_new = self._updated_row(src_new if same_id else dst_row, value)
        return src_new, dst_new


class TemporalMotifContext(nn.Module):
    """Encode recent destination-ID and IAT motifs from the current stream."""

    def __init__(self, *, num_ids: int, hidden: int, length: int, embedding_dim: int, time_dim: int):
        super().__init__()
        self.num_ids = max(1, int(num_ids))
        self.hidden = int(hidden)
        self.length = max(1, int(length))
        self.embedding = nn.Embedding(self.num_ids, int(embedding_dim))
        self.time_encoder = TemporalTimeEncoder(max(1, int(time_dim)))
        input_dim = self.length * (int(embedding_dim) + max(1, int(time_dim)))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
        )

    def initial_ids(self, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.length, device=device, dtype=torch.long)

    def initial_iats(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.length, device=device, dtype=dtype)

    def ensure_ids(self, ids: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if ids is None or ids.device != ref.device or ids.shape != (self.length,):
            return self.initial_ids(device=ref.device)
        return ids.long()

    def ensure_iats(self, iats: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if iats is None or iats.device != ref.device or iats.dtype != ref.dtype or iats.shape != (self.length,):
            return self.initial_iats(device=ref.device, dtype=ref.dtype)
        return iats

    def read_context(self, ids: torch.Tensor, iats: torch.Tensor) -> torch.Tensor:
        encoded_ids = self.embedding(ids.clamp_min(0).clamp_max(self.num_ids - 1).long()).flatten()
        encoded_iats = self.time_encoder(iats).flatten()
        return self.proj(torch.cat([encoded_ids, encoded_iats], dim=-1))

    def update_one(
        self,
        ids: torch.Tensor,
        iats: torch.Tensor,
        dst: torch.Tensor,
        iat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_id = dst.clamp_min(0).clamp_max(self.num_ids - 1).long().reshape(1)
        return self._update_one_indexed(ids, iats, next_id, iat)

    def _update_one_indexed(
        self,
        ids: torch.Tensor,
        iats: torch.Tensor,
        dst_idx: torch.Tensor,
        iat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_id = dst_idx.to(device=ids.device, dtype=torch.long).reshape(1)
        next_iat = iat.to(device=iats.device, dtype=iats.dtype).clamp_min(0).reshape(1)
        return torch.cat([ids[1:], next_id]), torch.cat([iats[1:], next_iat])


class _SsmLiteCell(nn.Module):
    """Small gated recurrent state-space-like block with causal linear-time scans."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden)
        self.decay_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros_like(x)
        decay = torch.sigmoid(self.decay_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))
        candidate = torch.tanh(self.in_proj(x))
        next_state = decay * state + gate * candidate
        out = self.out_proj(next_state)
        return self.norm(x + self.dropout(out)), next_state


class TemporalStreamBackbone(nn.Module):
    """Configurable causal stream backbone."""

    def __init__(self, *, backbone_type: str, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.backbone_type = str(backbone_type)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.dropout = nn.Dropout(float(dropout))

        if self.backbone_type == "none":
            self.cells = nn.ModuleList()
        elif self.backbone_type == "gru":
            self.cells = nn.ModuleList([nn.GRUCell(self.hidden, self.hidden) for _ in range(self.layers)])
        elif self.backbone_type == "ssm_lite":
            self.cells = nn.ModuleList([_SsmLiteCell(self.hidden, float(dropout)) for _ in range(self.layers)])
        elif self.backbone_type == "mamba":
            try:
                import mamba_ssm  # noqa: F401
            except ImportError as exc:  # pragma: no cover - depends on optional external package.
                raise ImportError(
                    "backbone.type='mamba' requires the optional 'mamba_ssm' package. "
                    "Install it explicitly or use backbone.type='ssm_lite'."
                ) from exc
            raise NotImplementedError("backbone.type='mamba' is reserved for the optional backend.")
        else:
            raise ValueError("backbone.type must be one of: none, gru, ssm_lite, mamba")

    def reset_state(self) -> torch.Tensor | None:
        return None

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backbone_type == "none":
            return x, None
        if state is None:
            state = x.new_zeros((self.layers, self.hidden))
        h = x
        next_layers: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            if self.backbone_type == "gru":
                h = cell(h.unsqueeze(0), state[layer_idx].unsqueeze(0)).squeeze(0)
                h = self.dropout(h)
                next_layers.append(h)
            else:
                h, next_state = cell(h, state[layer_idx])
                next_layers.append(next_state)
        return h, torch.stack(next_layers, dim=0)


class TemporalHybridModel(TemporalModuleBase):
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
        if in_channels >= 16:
            return slice(8, 16)
        return slice(0, in_channels)

    @classmethod
    def _payload_delta_dim(cls, in_channels: int) -> int:
        s = cls._payload_delta_slice(in_channels)
        return int(s.stop - s.start)

    @staticmethod
    def _detach_state(
        state: dict[str, torch.Tensor | None] | None,
    ) -> dict[str, torch.Tensor | None] | None:
        if state is None:
            return None
        return {k: (None if v is None else v.detach()) for k, v in state.items()}

    def _initial_state_like(self, encoded: torch.Tensor) -> dict[str, torch.Tensor | None]:
        memory = self.memory.ensure_state(None, encoded) if self.memory is not None else None
        last_seen = self.memory.ensure_last_seen(None, encoded) if self.memory is not None else None
        rhythm = self.rhythm.ensure_state(None, encoded) if getattr(self, "rhythm", None) is not None else None
        motif_ids = self.motif.ensure_ids(None, encoded) if getattr(self, "motif", None) is not None else None
        motif_iats = self.motif.ensure_iats(None, encoded) if getattr(self, "motif", None) is not None else None
        return {
            "memory": memory,
            "last_seen": last_seen,
            "rhythm": rhythm,
            "motif_ids": motif_ids,
            "motif_iats": motif_iats,
            "backbone": None,
        }

    def _initial_lane_state_like(self, encoded: torch.Tensor) -> dict[str, torch.Tensor | None]:
        lanes = int(encoded.size(0))
        flat_ref = encoded.reshape(-1, encoded.size(-1))
        state = self._initial_state_like(flat_ref)
        if state["memory"] is not None:
            state["memory"] = state["memory"].unsqueeze(0).expand(lanes, -1, -1).clone()
        if state["last_seen"] is not None:
            state["last_seen"] = state["last_seen"].unsqueeze(0).expand(lanes, -1).clone()
        if state["rhythm"] is not None:
            state["rhythm"] = state["rhythm"].unsqueeze(0).expand(lanes, -1, -1).clone()
        if state["motif_ids"] is not None:
            state["motif_ids"] = state["motif_ids"].unsqueeze(0).expand(lanes, -1).clone()
        if state["motif_iats"] is not None:
            state["motif_iats"] = state["motif_iats"].unsqueeze(0).expand(lanes, -1).clone()
        state["prev_t"] = torch.full((lanes,), -1.0, device=encoded.device, dtype=encoded.dtype)
        return state

    def _ensure_lane_state(
        self,
        state: dict[str, torch.Tensor | None] | None,
        encoded: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        lanes = int(encoded.size(0))
        current = self._initial_lane_state_like(encoded) if state is None else dict(state)
        initial = self._initial_lane_state_like(encoded)
        for key, value in initial.items():
            current_value = current.get(key)
            if current_value is None or value is None:
                current[key] = value if current_value is None else current_value
                continue
            if current_value.device != value.device or current_value.dtype != value.dtype or current_value.shape != value.shape:
                current[key] = value
        backbone = current.get("backbone")
        if backbone is not None and (
            backbone.device != encoded.device
            or backbone.dtype != encoded.dtype
            or backbone.shape[1:] != (lanes, encoded.size(-1))
        ):
            current["backbone"] = None
        return current

    @staticmethod
    def _reset_lane_rows(value: torch.Tensor | None, reset_mask: torch.Tensor, replacement: torch.Tensor | None) -> torch.Tensor | None:
        if value is None or replacement is None or not reset_mask.any():
            return value
        updated = value.clone()
        updated[reset_mask] = replacement[reset_mask]
        return updated

    @staticmethod
    def _reset_backbone_lanes(
        value: torch.Tensor | None,
        reset_mask: torch.Tensor,
        *,
        lanes: int,
        hidden: int,
    ) -> torch.Tensor | None:
        if value is None or not reset_mask.any():
            return value
        updated = value.clone()
        updated[:, reset_mask, :] = torch.zeros(
            (value.size(0), int(reset_mask.sum().item()), hidden),
            device=value.device,
            dtype=value.dtype,
        )
        return updated

    def _lane_memory_context(
        self,
        memory_state: torch.Tensor,
        last_seen: torch.Tensor | None,
        lane_idx: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.memory is not None
        if not self.memory._has_context:
            return memory_state.new_zeros((lane_idx.numel(), self.memory.hidden))
        parts: list[torch.Tensor] = []
        src_rows = memory_state[lane_idx, src_idx]
        dst_rows = memory_state[lane_idx, dst_idx]
        if self.memory.use_source:
            parts.append(src_rows)
        if self.memory.use_destination:
            parts.append(dst_rows)
        if self.memory.time_encoder is not None:
            assert last_seen is not None
            now = memory_state.new_zeros(lane_idx.numel()) if t is None else t.to(device=memory_state.device, dtype=memory_state.dtype)
            src_last = last_seen[lane_idx, src_idx]
            dst_last = last_seen[lane_idx, dst_idx]
            src_elapsed = torch.where(src_last >= 0, now - src_last, torch.zeros_like(now))
            dst_elapsed = torch.where(dst_last >= 0, now - dst_last, torch.zeros_like(now))
            parts.append(self.memory.time_encoder(src_elapsed))
            parts.append(self.memory.time_encoder(dst_elapsed))
        return self.memory.mix(torch.cat(parts, dim=-1))

    def _lane_rhythm_context(
        self,
        rhythm_state: torch.Tensor,
        lane_idx: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        assert self.rhythm is not None

        def features(rows: torch.Tensor) -> torch.Tensor:
            count = rows[:, 0].clamp_min(0)
            mean = rows[:, 1]
            variance = torch.where(count > 1, rows[:, 2] / (count - 1).clamp_min(1), torch.zeros_like(count))
            return torch.stack([torch.log1p(count), mean, torch.sqrt(variance.clamp_min(0))], dim=-1)

        src_rows = rhythm_state[lane_idx, src_idx]
        dst_rows = rhythm_state[lane_idx, dst_idx]
        return self.rhythm.proj(torch.cat([features(src_rows), features(dst_rows)], dim=-1))

    def _lane_motif_context(
        self,
        motif_ids: torch.Tensor,
        motif_iats: torch.Tensor,
        lane_idx: torch.Tensor,
    ) -> torch.Tensor:
        assert self.motif is not None
        ids = motif_ids[lane_idx].clamp_min(0).clamp_max(self.motif.num_ids - 1).long()
        iats = motif_iats[lane_idx]
        encoded_ids = self.motif.embedding(ids).flatten(start_dim=1)
        encoded_iats = self.motif.time_encoder(iats).flatten(start_dim=1)
        return self.motif.proj(torch.cat([encoded_ids, encoded_iats], dim=-1))

    def _lane_targets(self, batch) -> dict[str, torch.Tensor]:
        valid = batch.valid_mask.bool()
        lanes, chunk = valid.shape
        device = batch.dst.device
        valid_next = torch.zeros((lanes, chunk), dtype=torch.bool, device=device)
        if chunk > 1:
            valid_next[:, :-1] = valid[:, :-1] & valid[:, 1:] & ~batch.reset_after[:, :-1].bool()
        next_id = batch.dst.clone().long()
        if chunk > 1:
            next_id[:, :-1] = batch.dst[:, 1:].long()

        iat = batch.msg.new_zeros((lanes, chunk))
        if int(self.hparams.in_channels) > 16 and batch.msg.shape[-1] > 16:
            iat = batch.msg[..., 16].to(device=device, dtype=batch.msg.dtype).clamp_min(0)
        elif chunk > 1:
            iat[:, 1:] = (batch.t[:, 1:].to(dtype=batch.msg.dtype) - batch.t[:, :-1].to(dtype=batch.msg.dtype)).clamp_min(0)

        payload_slice = self._payload_delta_slice(int(self.hparams.in_channels))
        payload_delta = batch.msg[..., payload_slice].float().clone()
        if chunk > 1:
            payload_delta[:, :-1] = batch.msg[:, 1:, payload_slice].float()
        return {
            "valid_next": valid_next.reshape(-1),
            "next_id": next_id.reshape(-1),
            "iat": iat.reshape(-1),
            "payload_delta": payload_delta.reshape(lanes * chunk, -1),
        }

    def _targets(self, batch) -> dict[str, torch.Tensor]:
        if getattr(batch, "valid_mask", None) is not None:
            return self._lane_targets(batch)
        n = int(batch.dst.numel())
        device = batch.dst.device
        valid_next = torch.zeros(n, dtype=torch.bool, device=device)
        if n > 1:
            reset_after = getattr(batch, "reset_after", None)
            if reset_after is None:
                reset_after = torch.zeros(n, dtype=torch.bool, device=device)
            else:
                reset_after = reset_after.to(device=device, dtype=torch.bool)
            valid_next[:-1] = ~reset_after[:-1]

        next_id = batch.dst.clone().long()
        if n > 1:
            next_id[:-1] = batch.dst[1:].long()
        t = getattr(batch, "t", None)
        iat = batch.msg.new_zeros(n)
        if t is not None and n > 1:
            iat[:-1] = (
                t[1:].to(device=device, dtype=batch.msg.dtype)
                - t[:-1].to(device=device, dtype=batch.msg.dtype)
            ).clamp_min(0)
        payload_slice = self._payload_delta_slice(int(self.hparams.in_channels))
        payload_delta = batch.msg[:, payload_slice].float().clone()
        if n > 1:
            payload_delta[:-1] = batch.msg[1:, payload_slice].float()
        return {"valid_next": valid_next, "next_id": next_id, "iat": iat, "payload_delta": payload_delta}

    def _event_iat(self, batch, idx: int, ref: torch.Tensor) -> torch.Tensor:
        if int(self.hparams.in_channels) > 16 and batch.msg.shape[1] > 16:
            return batch.msg[idx, 16].to(device=ref.device, dtype=ref.dtype).clamp_min(0)
        t = getattr(batch, "t", None)
        if t is not None and idx > 0:
            return (
                t[idx].to(device=ref.device, dtype=ref.dtype)
                - t[idx - 1].to(device=ref.device, dtype=ref.dtype)
            ).clamp_min(0)
        return ref.new_tensor(0.0)

    def _prepare_scan_batch(self, batch, ref: torch.Tensor) -> dict[str, torch.Tensor | None]:
        src_idx = batch.src.to(device=ref.device).clamp_min(0).clamp_max(int(self.hparams.num_ids) - 1).long()
        dst_idx = batch.dst.to(device=ref.device).clamp_min(0).clamp_max(int(self.hparams.num_ids) - 1).long()
        n = int(src_idx.numel())

        t = getattr(batch, "t", None)
        t_cast = None if t is None else t.to(device=ref.device, dtype=ref.dtype)
        iat = ref.new_zeros(n)
        if int(self.hparams.in_channels) > 16 and batch.msg.shape[1] > 16:
            iat = batch.msg[:, 16].to(device=ref.device, dtype=ref.dtype).clamp_min(0)
        elif t_cast is not None and n > 1:
            iat[1:] = (t_cast[1:] - t_cast[:-1]).clamp_min(0)

        reset_after = getattr(batch, "reset_after", None)
        if reset_after is None:
            reset_after = torch.zeros(n, dtype=torch.bool, device=ref.device)
        else:
            reset_after = reset_after.to(device=ref.device, dtype=torch.bool)
        return {"src_idx": src_idx, "dst_idx": dst_idx, "iat": iat, "t": t_cast, "reset_after": reset_after}

    def _scan_segment(
        self,
        encoded: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        iat: torch.Tensor,
        t: torch.Tensor | None,
        memory_state: torch.Tensor | None,
        last_seen: torch.Tensor | None,
        rhythm_state: torch.Tensor | None,
        motif_ids: torch.Tensor | None,
        motif_iats: torch.Tensor | None,
        backbone_state: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        outputs: list[torch.Tensor] = []
        src_ids = [int(value) for value in src_idx.detach().cpu().tolist()]
        dst_ids = [int(value) for value in dst_idx.detach().cpu().tolist()]
        memory_rows: dict[int, torch.Tensor] = {}
        last_seen_values: dict[int, torch.Tensor] = {}
        rhythm_rows: dict[int, torch.Tensor] = {}

        def memory_row(row_id: int) -> torch.Tensor:
            assert memory_state is not None
            row = memory_rows.get(row_id)
            return memory_state[row_id] if row is None else row

        def last_seen_value(row_id: int) -> torch.Tensor:
            assert last_seen is not None
            value = last_seen_values.get(row_id)
            return last_seen[row_id] if value is None else value

        def rhythm_row(row_id: int) -> torch.Tensor:
            assert rhythm_state is not None
            row = rhythm_rows.get(row_id)
            return rhythm_state[row_id] if row is None else row

        for idx in range(encoded.size(0)):
            x_t = encoded[idx]
            iat_t = iat[idx]
            t_t = None if t is None else t[idx]
            src_id = src_ids[idx]
            dst_id = dst_ids[idx]
            same_id = src_id == dst_id
            if self.memory is not None and memory_state is not None:
                mem_ctx = self.memory._read_context_rows(
                    memory_row(src_id),
                    memory_row(dst_id),
                    None if last_seen is None else last_seen_value(src_id),
                    None if last_seen is None else last_seen_value(dst_id),
                    t_t,
                )
                x_t = x_t + mem_ctx
            if self.rhythm is not None and rhythm_state is not None:
                x_t = x_t + self.rhythm._read_context_rows(rhythm_row(src_id), rhythm_row(dst_id))
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                x_t = x_t + self.motif.read_context(motif_ids, motif_iats)
            z_t, backbone_state = self.backbone.step(x_t, backbone_state)
            outputs.append(z_t)
            if self.memory is not None and memory_state is not None:
                src_new, dst_new = self.memory._updated_rows(
                    memory_row(src_id),
                    memory_row(dst_id),
                    z_t,
                    same_id=same_id,
                )
                memory_rows[src_id] = src_new
                memory_rows[dst_id] = dst_new
                if last_seen is not None and t_t is not None:
                    now = t_t.to(device=last_seen.device, dtype=last_seen.dtype)
                    last_seen_values[src_id] = now
                    last_seen_values[dst_id] = now
            if self.rhythm is not None and rhythm_state is not None:
                src_new, dst_new = self.rhythm._updated_rows(
                    rhythm_row(src_id),
                    rhythm_row(dst_id),
                    iat_t,
                    same_id=same_id,
                )
                rhythm_rows[src_id] = src_new
                rhythm_rows[dst_id] = dst_new
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                motif_ids, motif_iats = self.motif._update_one_indexed(motif_ids, motif_iats, dst_idx[idx], iat_t)

        if memory_rows and memory_state is not None:
            row_ids = torch.tensor(list(memory_rows), device=memory_state.device, dtype=torch.long)
            rows = torch.stack([memory_rows[row_id] for row_id in memory_rows], dim=0)
            materialized_memory = memory_state.clone()
            materialized_memory[row_ids] = rows
            memory_state = materialized_memory
        if last_seen_values and last_seen is not None:
            row_ids = torch.tensor(list(last_seen_values), device=last_seen.device, dtype=torch.long)
            values = torch.stack([last_seen_values[row_id] for row_id in last_seen_values], dim=0)
            materialized_last_seen = last_seen.clone()
            materialized_last_seen[row_ids] = values
            last_seen = materialized_last_seen
        if rhythm_rows and rhythm_state is not None:
            row_ids = torch.tensor(list(rhythm_rows), device=rhythm_state.device, dtype=torch.long)
            rows = torch.stack([rhythm_rows[row_id] for row_id in rhythm_rows], dim=0)
            materialized_rhythm = rhythm_state.clone()
            materialized_rhythm[row_ids] = rows
            rhythm_state = materialized_rhythm

        features = torch.stack(outputs, dim=0) if outputs else encoded.new_empty((0, encoded.size(-1)))
        return features, memory_state, last_seen, rhythm_state, motif_ids, motif_iats, backbone_state

    def _forward_lane_with_state(
        self,
        batch,
        state: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        encoded = self.encoder(batch)
        lanes, chunk, hidden = encoded.shape
        current_state = self._ensure_lane_state(state, encoded)
        initial_state = self._initial_lane_state_like(encoded)

        lane_reset = batch.lane_reset.to(device=encoded.device, dtype=torch.bool)
        lane_end = batch.lane_stream_end.to(device=encoded.device, dtype=torch.bool)
        memory_state = self._reset_lane_rows(current_state.get("memory"), lane_reset, initial_state.get("memory"))
        last_seen = self._reset_lane_rows(current_state.get("last_seen"), lane_reset, initial_state.get("last_seen"))
        rhythm_state = self._reset_lane_rows(current_state.get("rhythm"), lane_reset, initial_state.get("rhythm"))
        motif_ids = self._reset_lane_rows(current_state.get("motif_ids"), lane_reset, initial_state.get("motif_ids"))
        motif_iats = self._reset_lane_rows(current_state.get("motif_iats"), lane_reset, initial_state.get("motif_iats"))
        prev_t = self._reset_lane_rows(current_state.get("prev_t"), lane_reset, initial_state.get("prev_t"))
        backbone_state = self._reset_backbone_lanes(
            current_state.get("backbone"),
            lane_reset,
            lanes=lanes,
            hidden=hidden,
        )

        valid = batch.valid_mask.to(device=encoded.device, dtype=torch.bool)
        src_idx_all = batch.src.to(device=encoded.device).clamp_min(0).clamp_max(int(self.hparams.num_ids) - 1).long()
        dst_idx_all = batch.dst.to(device=encoded.device).clamp_min(0).clamp_max(int(self.hparams.num_ids) - 1).long()
        t_all = batch.t.to(device=encoded.device, dtype=encoded.dtype)
        if int(self.hparams.in_channels) > 16 and batch.msg.shape[-1] > 16:
            iat_all = batch.msg[..., 16].to(device=encoded.device, dtype=encoded.dtype).clamp_min(0)
        else:
            assert prev_t is not None
            previous_t = torch.cat([prev_t.view(lanes, 1), t_all[:, :-1]], dim=1)
            seen_previous = torch.cat(
                [prev_t.ge(0).view(lanes, 1), valid[:, :-1] if chunk > 1 else valid[:, :0]],
                dim=1,
            )
            iat_all = torch.where(seen_previous, (t_all - previous_t).clamp_min(0), encoded.new_zeros((lanes, chunk)))

        features_2d = encoded.new_zeros((lanes, chunk, hidden))
        lane_ids_all = torch.arange(lanes, device=encoded.device)

        for step_idx in range(chunk):
            active = valid[:, step_idx]
            if not active.any():
                continue
            lane_idx = lane_ids_all[active]
            src_idx = src_idx_all[active, step_idx]
            dst_idx = dst_idx_all[active, step_idx]
            iat_t = iat_all[active, step_idx]
            t_t = t_all[active, step_idx]
            x_t = encoded[active, step_idx]

            if self.memory is not None and memory_state is not None:
                x_t = x_t + self._lane_memory_context(memory_state, last_seen, lane_idx, src_idx, dst_idx, t_t)
            if self.rhythm is not None and rhythm_state is not None:
                x_t = x_t + self._lane_rhythm_context(rhythm_state, lane_idx, src_idx, dst_idx)
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                x_t = x_t + self._lane_motif_context(motif_ids, motif_iats, lane_idx)

            active_backbone = None if backbone_state is None else backbone_state[:, active, :]
            z_t, next_backbone = self.backbone.step(x_t, active_backbone)
            features_2d[active, step_idx] = z_t
            if next_backbone is not None:
                if backbone_state is None:
                    backbone_state = encoded.new_zeros((next_backbone.size(0), lanes, hidden))
                else:
                    backbone_state = backbone_state.clone()
                backbone_state[:, active, :] = next_backbone

            same_id = src_idx == dst_idx
            if self.memory is not None and memory_state is not None:
                src_rows = memory_state[lane_idx, src_idx]
                dst_rows = memory_state[lane_idx, dst_idx]
                src_new = self.memory.update_cell(z_t, src_rows)
                dst_input = torch.where(same_id.unsqueeze(-1), src_new, dst_rows)
                dst_new = self.memory.update_cell(z_t, dst_input)
                memory_state = memory_state.clone()
                memory_state[lane_idx, src_idx] = src_new
                memory_state[lane_idx, dst_idx] = dst_new
                if last_seen is not None:
                    now = t_t.to(device=last_seen.device, dtype=last_seen.dtype)
                    last_seen = last_seen.clone()
                    last_seen[lane_idx, src_idx] = now
                    last_seen[lane_idx, dst_idx] = now
            if self.rhythm is not None and rhythm_state is not None:
                src_rows = rhythm_state[lane_idx, src_idx]
                dst_rows = rhythm_state[lane_idx, dst_idx]
                src_new = torch.stack(
                    [self.rhythm._updated_row(src_rows[row], iat_t[row]) for row in range(src_rows.size(0))],
                    dim=0,
                )
                dst_base = torch.where(same_id.view(-1, 1), src_new, dst_rows)
                dst_new = torch.stack(
                    [self.rhythm._updated_row(dst_base[row], iat_t[row]) for row in range(dst_base.size(0))],
                    dim=0,
                )
                rhythm_state = rhythm_state.clone()
                rhythm_state[lane_idx, src_idx] = src_new
                rhythm_state[lane_idx, dst_idx] = dst_new
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                motif_ids = motif_ids.clone()
                motif_iats = motif_iats.clone()
                motif_ids[lane_idx] = torch.cat(
                    [motif_ids[lane_idx, 1:], dst_idx.to(device=motif_ids.device).view(-1, 1)],
                    dim=1,
                )
                motif_iats[lane_idx] = torch.cat(
                    [motif_iats[lane_idx, 1:], iat_t.to(device=motif_iats.device, dtype=motif_iats.dtype).view(-1, 1)],
                    dim=1,
                )
            if prev_t is not None:
                prev_t = prev_t.clone()
                prev_t[lane_idx] = t_t.to(device=prev_t.device, dtype=prev_t.dtype)

        reset_memory = bool(self.hparams.memory.get("reset_on_stream_end", True))
        if lane_end.any():
            backbone_state = self._reset_backbone_lanes(backbone_state, lane_end, lanes=lanes, hidden=hidden)
            if self.memory is not None and reset_memory:
                memory_state = self._reset_lane_rows(memory_state, lane_end, initial_state.get("memory"))
                last_seen = self._reset_lane_rows(last_seen, lane_end, initial_state.get("last_seen"))
            rhythm_state = self._reset_lane_rows(rhythm_state, lane_end, initial_state.get("rhythm"))
            motif_ids = self._reset_lane_rows(motif_ids, lane_end, initial_state.get("motif_ids"))
            motif_iats = self._reset_lane_rows(motif_iats, lane_end, initial_state.get("motif_iats"))
            prev_t = self._reset_lane_rows(prev_t, lane_end, initial_state.get("prev_t"))

        features = features_2d.reshape(lanes * chunk, hidden)
        next_state = {
            "memory": memory_state,
            "last_seen": last_seen,
            "rhythm": rhythm_state,
            "motif_ids": motif_ids,
            "motif_iats": motif_iats,
            "prev_t": prev_t,
            "backbone": backbone_state,
        }
        return self._heads_from_features(features, batch), next_state

    def _forward_with_state(
        self,
        batch,
        state: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        if getattr(batch, "valid_mask", None) is not None:
            return self._forward_lane_with_state(batch, state)
        encoded = self.encoder(batch)
        if encoded.numel() == 0:
            empty = encoded.new_empty((0, int(self.hparams.input["hidden"])))
            out = self._heads_from_features(empty, batch)
            next_state = self._initial_state_like(encoded)
            return out, next_state

        current_state = self._initial_state_like(encoded) if state is None else dict(state)
        if self.memory is not None:
            current_state["memory"] = self.memory.ensure_state(current_state.get("memory"), encoded)
            current_state["last_seen"] = self.memory.ensure_last_seen(current_state.get("last_seen"), encoded)
        if self.rhythm is not None:
            current_state["rhythm"] = self.rhythm.ensure_state(current_state.get("rhythm"), encoded)
        if self.motif is not None:
            current_state["motif_ids"] = self.motif.ensure_ids(current_state.get("motif_ids"), encoded)
            current_state["motif_iats"] = self.motif.ensure_iats(current_state.get("motif_iats"), encoded)

        scan_batch = self._prepare_scan_batch(batch, encoded)
        reset_after = scan_batch["reset_after"]
        assert reset_after is not None
        reset_points = reset_after.nonzero(as_tuple=False).flatten().tolist()

        outputs: list[torch.Tensor] = []
        memory_state = current_state.get("memory")
        last_seen = current_state.get("last_seen")
        rhythm_state = current_state.get("rhythm")
        motif_ids = current_state.get("motif_ids")
        motif_iats = current_state.get("motif_iats")
        backbone_state = current_state.get("backbone")
        reset_memory = bool(self.hparams.memory.get("reset_on_stream_end", True))

        segment_start = 0
        segment_ends = [point + 1 for point in reset_points]
        if not segment_ends or segment_ends[-1] < encoded.size(0):
            segment_ends.append(encoded.size(0))

        for segment_end in segment_ends:
            if segment_end > segment_start:
                segment_t = None
                if scan_batch["t"] is not None:
                    segment_t = scan_batch["t"][segment_start:segment_end]
                (
                    segment_features,
                    memory_state,
                    last_seen,
                    rhythm_state,
                    motif_ids,
                    motif_iats,
                    backbone_state,
                ) = self._scan_segment(
                    encoded[segment_start:segment_end],
                    scan_batch["src_idx"][segment_start:segment_end],
                    scan_batch["dst_idx"][segment_start:segment_end],
                    scan_batch["iat"][segment_start:segment_end],
                    segment_t,
                    memory_state,
                    last_seen,
                    rhythm_state,
                    motif_ids,
                    motif_iats,
                    backbone_state,
                )
                outputs.append(segment_features)
            reset_idx = segment_end - 1
            if reset_idx in reset_points:
                backbone_state = self.backbone.reset_state()
                if self.memory is not None and reset_memory:
                    memory_state = self.memory.initial_state(device=encoded.device, dtype=encoded.dtype)
                    last_seen = self.memory.initial_last_seen(device=encoded.device, dtype=encoded.dtype)
                if self.rhythm is not None:
                    rhythm_state = self.rhythm.initial_state(device=encoded.device, dtype=encoded.dtype)
                if self.motif is not None:
                    motif_ids = self.motif.initial_ids(device=encoded.device)
                    motif_iats = self.motif.initial_iats(device=encoded.device, dtype=encoded.dtype)
            segment_start = segment_end

        features = torch.cat(outputs, dim=0)
        next_state = {
            "memory": memory_state,
            "last_seen": last_seen,
            "rhythm": rhythm_state,
            "motif_ids": motif_ids,
            "motif_iats": motif_iats,
            "backbone": backbone_state,
        }
        return self._heads_from_features(features, batch), next_state

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

    def on_train_epoch_start(self) -> None:
        self._train_state = None

    def on_validation_epoch_start(self) -> None:
        self._val_state = None
        self._val_cls_probs.clear()
        self._val_cls_labels.clear()
        self._val_anom_scores.clear()
        self._val_anom_labels.clear()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._test_states = {}

    def training_step(self, batch, _idx):
        out, state = self._forward_with_state(batch, self._detach_state(self._train_state))
        self._train_state = self._detach_state(state)
        terms = self._loss_terms(out, batch)
        mask = self.scored_mask(batch)
        bs = max(1, int(mask.sum().item()))
        self.log("train_loss", terms["total"], batch_size=bs)
        for name, value in terms.items():
            if name in {"total", "zero"}:
                continue
            self.log(f"train_{name}_loss", value, batch_size=bs)
        if self.classification_enabled and "logits" in out and mask.any():
            labels = batch.y.reshape(-1)[mask].long()
            acc = (out["logits"][mask].argmax(1) == labels).float().mean()
            self.log("train_acc", acc, batch_size=int(labels.numel()))
        return terms["total"]

    def validation_step(self, batch, _idx):
        out, state = self._forward_with_state(batch, self._detach_state(self._val_state))
        self._val_state = self._detach_state(state)
        terms = self._loss_terms(out, batch)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        labels = batch.y.reshape(-1)[mask].long()
        bs = int(labels.numel())
        self.log("val_loss", terms["total"], batch_size=bs)
        if self.classification_enabled and "logits" in out:
            probs = F.softmax(out["logits"][mask], dim=1)
            self.log("val_acc", (probs.argmax(1) == labels).float().mean(), batch_size=bs)
            if probs.shape[1] == 2:
                self._val_cls_probs.append(probs[:, 1].detach().cpu())
                self._val_cls_labels.append(labels.detach().cpu())
        if self.anomaly_enabled and "anomaly_score" in out:
            self._val_anom_scores.append(out["anomaly_score"][mask].detach().cpu())
            self._val_anom_labels.append(labels.detach().cpu())
        return None

    def on_validation_epoch_end(self) -> None:
        from torchmetrics.functional.classification import binary_auroc

        if self._val_cls_probs:
            labels = torch.cat(self._val_cls_labels)
            if labels.unique().numel() >= 2:
                self.log("val_cls_auroc", binary_auroc(torch.cat(self._val_cls_probs), labels))
        if self._val_anom_scores:
            labels = torch.cat(self._val_anom_labels)
            if labels.unique().numel() >= 2:
                self.log("val_anomaly_auroc", binary_auroc(torch.cat(self._val_anom_scores), labels))
        self._val_cls_probs.clear()
        self._val_cls_labels.clear()
        self._val_anom_scores.clear()
        self._val_anom_labels.clear()

    def test_step(self, batch, _idx, dataloader_idx=0):
        out, state = self._forward_with_state(
            batch,
            self._detach_state(self._test_states.get(dataloader_idx)),
        )
        self._test_states[dataloader_idx] = self._detach_state(state)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        labels = batch.y.reshape(-1)[mask].long()
        attack_type = getattr(batch, "attack_type", None)
        attack_type = attack_type.reshape(-1)[mask] if attack_type is not None else None
        if self.classification_enabled and "logits" in out:
            probs = F.softmax(out["logits"][mask], dim=1)
            self._record_test_batch(
                dataloader_idx,
                preds=probs.argmax(1),
                scores=probs,
                labels=labels,
                attack_type=attack_type,
            )
        if self.anomaly_enabled and "anomaly_score" in out:
            self._record_binary_score_batch(
                dataloader_idx,
                scores=out["anomaly_score"][mask],
                labels=labels,
                attack_type=attack_type,
            )
        return None

    def predict_step(self, batch, _idx):
        out = self(batch)
        result: dict[str, torch.Tensor] = {"labels": batch.y.reshape(-1)}
        if "logits" in out:
            probs = F.softmax(out["logits"], dim=1)
            result["preds"] = probs.argmax(1)
            result["class_scores"] = probs[:, 1] if probs.shape[1] == 2 else probs
        if "anomaly_score" in out:
            result["scores"] = out["anomaly_score"]
        event_id = getattr(batch, "event_id", None)
        if event_id is not None:
            result["event_id"] = event_id.reshape(-1)
        return result
