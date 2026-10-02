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
