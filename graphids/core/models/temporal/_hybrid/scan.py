"""Causal scan implementations for temporal hybrid event batches."""

from __future__ import annotations

import torch


class HybridScanMixin:
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

        if self.memory is None and self.rhythm is None and self.motif is None:
            features_2d, backbone_state = self.backbone.forward_sequence(encoded, backbone_state, valid)
            if prev_t is not None and valid.any():
                widths = valid.long().sum(dim=1)
                has_events = widths > 0
                if has_events.any():
                    prev_t = prev_t.clone()
                    event_lanes = torch.arange(lanes, device=encoded.device)[has_events]
                    last_pos = widths[has_events] - 1
                    prev_t[event_lanes] = t_all[event_lanes, last_pos].to(device=prev_t.device, dtype=prev_t.dtype)
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

        features_2d = encoded.new_zeros((lanes, chunk, hidden))
        lane_ids_all = torch.arange(lanes, device=encoded.device)
        if memory_state is not None:
            memory_state = memory_state.clone()
        if last_seen is not None:
            last_seen = last_seen.clone()
        if rhythm_state is not None:
            rhythm_state = rhythm_state.clone()
        if motif_ids is not None:
            motif_ids = motif_ids.clone()
        if motif_iats is not None:
            motif_iats = motif_iats.clone()
        if backbone_state is not None:
            backbone_state = backbone_state.clone()

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
                assert last_seen is not None
                src_rows = memory_state[lane_idx, src_idx]
                dst_rows = memory_state[lane_idx, dst_idx]
                src_last = last_seen[lane_idx, src_idx]
                dst_last = last_seen[lane_idx, dst_idx]
                if self.memory._has_context:
                    parts: list[torch.Tensor] = []
                    if self.memory.use_source:
                        parts.append(src_rows)
                    if self.memory.use_destination:
                        parts.append(dst_rows)
                    if self.memory.time_encoder is not None:
                        now = t_t.to(device=memory_state.device, dtype=memory_state.dtype)
                        src_elapsed = torch.where(src_last >= 0, now - src_last, torch.zeros_like(now))
                        dst_elapsed = torch.where(dst_last >= 0, now - dst_last, torch.zeros_like(now))
                        parts.append(self.memory.time_encoder(src_elapsed))
                        parts.append(self.memory.time_encoder(dst_elapsed))
                    x_t = x_t + self.memory.mix(torch.cat(parts, dim=-1))
            if self.rhythm is not None and rhythm_state is not None:
                src_rows = rhythm_state[lane_idx, src_idx]
                dst_rows = rhythm_state[lane_idx, dst_idx]
                src_count = src_rows[:, 0].clamp_min(0)
                src_variance = torch.where(
                    src_count > 1,
                    src_rows[:, 2] / (src_count - 1).clamp_min(1),
                    torch.zeros_like(src_count),
                )
                dst_count = dst_rows[:, 0].clamp_min(0)
                dst_variance = torch.where(
                    dst_count > 1,
                    dst_rows[:, 2] / (dst_count - 1).clamp_min(1),
                    torch.zeros_like(dst_count),
                )
                src_features = torch.stack(
                    [torch.log1p(src_count), src_rows[:, 1], torch.sqrt(src_variance.clamp_min(0))],
                    dim=-1,
                )
                dst_features = torch.stack(
                    [torch.log1p(dst_count), dst_rows[:, 1], torch.sqrt(dst_variance.clamp_min(0))],
                    dim=-1,
                )
                x_t = x_t + self.rhythm.proj(torch.cat([src_features, dst_features], dim=-1))
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                ids = motif_ids[lane_idx]
                iats = motif_iats[lane_idx]
                encoded_ids = self.motif.embedding(ids.clamp_min(0).clamp_max(self.motif.num_ids - 1).long()).flatten(start_dim=1)
                encoded_iats = self.motif.time_encoder(iats).flatten(start_dim=1)
                x_t = x_t + self.motif.proj(torch.cat([encoded_ids, encoded_iats], dim=-1))

            active_backbone = None if backbone_state is None else backbone_state[:, active, :]
            z_t, next_backbone = self.backbone.step(x_t, active_backbone)
            features_2d[active, step_idx] = z_t
            if next_backbone is not None:
                if backbone_state is None:
                    backbone_state = encoded.new_zeros((next_backbone.size(0), lanes, hidden))
                backbone_state[:, active, :] = next_backbone

            same_id = src_idx == dst_idx
            if self.memory is not None and memory_state is not None:
                src_rows = memory_state[lane_idx, src_idx]
                dst_rows = memory_state[lane_idx, dst_idx]
                src_new = self.memory.update_cell(z_t, src_rows)
                dst_input = torch.where(same_id.unsqueeze(-1), src_new, dst_rows)
                dst_new = self.memory.update_cell(z_t, dst_input)
                memory_state[lane_idx, src_idx] = src_new
                memory_state[lane_idx, dst_idx] = dst_new
                if last_seen is not None:
                    now = t_t.to(device=last_seen.device, dtype=last_seen.dtype)
                    last_seen[lane_idx, src_idx] = now
                    last_seen[lane_idx, dst_idx] = now
            if self.rhythm is not None and rhythm_state is not None:
                src_rows = rhythm_state[lane_idx, src_idx]
                dst_rows = rhythm_state[lane_idx, dst_idx]
                src_new, dst_new = self.rhythm._updated_rows_indexed_batch(src_rows, dst_rows, iat_t, same_id)
                rhythm_state[lane_idx, src_idx] = src_new
                rhythm_state[lane_idx, dst_idx] = dst_new
            if self.motif is not None and motif_ids is not None and motif_iats is not None:
                current_ids = motif_ids[lane_idx]
                current_iats = motif_iats[lane_idx]
                next_ids = torch.cat(
                    [current_ids[:, 1:], dst_idx.to(device=motif_ids.device).view(-1, 1)],
                    dim=1,
                )
                next_iats = torch.cat(
                    [current_iats[:, 1:], iat_t.to(device=motif_iats.device, dtype=motif_iats.dtype).view(-1, 1)],
                    dim=1,
                )
                motif_ids[lane_idx] = next_ids
                motif_iats[lane_idx] = next_iats
        if prev_t is not None and valid.any():
            widths = valid.long().sum(dim=1)
            has_events = widths > 0
            if has_events.any():
                prev_t = prev_t.clone()
                event_lanes = lane_ids_all[has_events]
                last_pos = widths[has_events] - 1
                prev_t[event_lanes] = t_all[event_lanes, last_pos].to(device=prev_t.device, dtype=prev_t.dtype)

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
