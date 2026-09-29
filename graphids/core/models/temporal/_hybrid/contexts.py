"""Stateful context modules for temporal hybrid event streams."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import TemporalTimeEncoder


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
