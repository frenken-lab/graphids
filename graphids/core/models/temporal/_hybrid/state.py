"""State helpers for the temporal hybrid model."""

from __future__ import annotations

import torch


class HybridStateMixin:
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
