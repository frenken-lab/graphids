"""Causal scan implementations for temporal hybrid event batches."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class _LaneStepGather:
    active: torch.Tensor
    lane_idx: torch.Tensor
    src_idx: torch.Tensor
    dst_idx: torch.Tensor
    iat_t: torch.Tensor
    t_t: torch.Tensor
    flat_src: torch.Tensor
    flat_dst: torch.Tensor
    same_id: torch.Tensor
    all_lanes_active: bool
    memory_src_rows: torch.Tensor | None = None
    memory_dst_rows: torch.Tensor | None = None
    rhythm_src_rows: torch.Tensor | None = None
    rhythm_dst_rows: torch.Tensor | None = None
    rhythm_updated: bool = False
    motif_updated: bool = False


class HybridScanMixin:
    def _try_time_kernel(self, kind: str, values: torch.Tensor, *args):
        """Use fixed time features on supported inputs; disable failed launches once.

        Gradient-bearing timestamps/IATs keep the PyTorch encoder. Memory rows
        and learned projections are never passed into this non-autograd kernel.
        """
        failure_attr = f"_{kind}_time_kernel_failed"
        if (self.hparams.scan.get("time_kernel", "torch") != "triton"
                or getattr(self, failure_attr, False) or not values.is_cuda
                or values.dtype != torch.float32 or values.requires_grad):
            return None
        try:
            from .kernels.time_features import (
                memory_time_features_triton,
                time_features_triton,
            )

            kernel = memory_time_features_triton if kind == "memory" else time_features_triton
            return kernel(values, *args)
        except Exception as exc:
            setattr(self, failure_attr, True)
            warnings.warn(f"Triton {kind} time kernel unavailable; using torch: {exc}", RuntimeWarning, stacklevel=2)
            return None

    @staticmethod
    def _add_lane_context(base: torch.Tensor, current: torch.Tensor | None, context: torch.Tensor) -> torch.Tensor:
        if current is None:
            current = base.clone()
        current.add_(context.to(device=current.device, dtype=current.dtype))
        return current

    @staticmethod
    def _flat_index_select(rows: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return rows.index_select(0, index)

    @staticmethod
    def _flat_index_copy(rows: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> None:
        rows.index_copy_(0, index, values.to(device=rows.device, dtype=rows.dtype))

    def _lane_step_context(
        self,
        *,
        encoded: torch.Tensor,
        valid: torch.Tensor,
        src_idx_all: torch.Tensor,
        dst_idx_all: torch.Tensor,
        iat_all: torch.Tensor,
        t_all: torch.Tensor,
        lane_ids_all: torch.Tensor,
        step_idx: int,
        num_ids: int,
        full_chunk_fastpath: bool,
        profile_scan_subphases: bool,
        memory_state: torch.Tensor | None,
        last_seen: torch.Tensor | None,
        rhythm_state: torch.Tensor | None,
        motif_ids: torch.Tensor | None,
        motif_iats: torch.Tensor | None,
        memory_flat: torch.Tensor | None,
        last_seen_flat: torch.Tensor | None,
        rhythm_flat: torch.Tensor | None,
        flat_src_all: torch.Tensor | None,
        flat_dst_all: torch.Tensor | None,
    ) -> tuple[torch.Tensor, _LaneStepGather | None]:
        active = valid[:, step_idx]
        if full_chunk_fastpath:
            all_lanes_active = True
        elif not active.any():
            return encoded.new_empty((0, encoded.size(-1))), None
        else:
            all_lanes_active = bool(active.all())

        if all_lanes_active:
            lane_idx = lane_ids_all
            src_idx = src_idx_all[:, step_idx]
            dst_idx = dst_idx_all[:, step_idx]
            iat_t = iat_all[:, step_idx]
            t_t = t_all[:, step_idx]
            x_t = encoded[:, step_idx]
        else:
            lane_idx = lane_ids_all[active]
            src_idx = src_idx_all[active, step_idx]
            dst_idx = dst_idx_all[active, step_idx]
            iat_t = iat_all[active, step_idx]
            t_t = t_all[active, step_idx]
            x_t = encoded[active, step_idx]

        if flat_src_all is not None and flat_dst_all is not None:
            if all_lanes_active:
                flat_src = flat_src_all[:, step_idx]
                flat_dst = flat_dst_all[:, step_idx]
            else:
                flat_src = flat_src_all[active, step_idx]
                flat_dst = flat_dst_all[active, step_idx]
        else:
            flat_src = lane_idx * num_ids + src_idx
            flat_dst = lane_idx * num_ids + dst_idx
        gathered = _LaneStepGather(
            active=active,
            lane_idx=lane_idx,
            src_idx=src_idx,
            dst_idx=dst_idx,
            iat_t=iat_t,
            t_t=t_t,
            flat_src=flat_src,
            flat_dst=flat_dst,
            same_id=src_idx == dst_idx,
            all_lanes_active=all_lanes_active,
        )

        x_with_context: torch.Tensor | None = None

        if self.memory is not None and memory_state is not None:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            assert last_seen is not None
            if memory_flat is not None and last_seen_flat is not None:
                src_rows = self._flat_index_select(memory_flat, flat_src)
                dst_rows = self._flat_index_select(memory_flat, flat_dst)
                encoded_time = None
                if self.memory.use_source and self.memory.use_destination and self.memory.time_encoder is not None:
                    encoded_time = self._try_time_kernel(
                        "memory", last_seen_flat, flat_src, flat_dst, t_t,
                        self.memory.time_encoder.freqs, self.memory.time_encoder.dim,
                    )
                # The fused time path gathers timestamps itself, avoiding two
                # additional index_select launches. Memory gathers retain autograd.
                src_last = None if encoded_time is not None else self._flat_index_select(last_seen_flat, flat_src)
                dst_last = None if encoded_time is not None else self._flat_index_select(last_seen_flat, flat_dst)
            else:
                src_rows = memory_state[lane_idx, src_idx]
                dst_rows = memory_state[lane_idx, dst_idx]
                encoded_time = None
                src_last = last_seen[lane_idx, src_idx]
                dst_last = last_seen[lane_idx, dst_idx]
            gathered.memory_src_rows = src_rows
            gathered.memory_dst_rows = dst_rows
            if self.memory._has_context:
                now = t_t.to(device=memory_state.device, dtype=memory_state.dtype)
                memory_context = self.memory.read_context_rows_batch(
                    src_rows, dst_rows, src_last, dst_last, now, encoded_time=encoded_time,
                )
                x_with_context = self._add_lane_context(x_t, x_with_context, memory_context)
            if profile_scan_subphases:
                self._scan_subphase_end("memory_context", subphase_start)

        # Advanced indexing owns these snapshots. The optional rhythm writer
        # can now mutate motif history without changing current context or the
        # ID indices saved by embedding backward.
        motif_ids_snapshot = None if motif_ids is None else motif_ids[lane_idx]
        motif_iats_snapshot = None if motif_iats is None else motif_iats[lane_idx]

        if self.rhythm is not None and rhythm_state is not None:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            features = None
            if (
                self.hparams.scan.get("rhythm_kernel", "torch") == "triton"
                and not getattr(self, "_rhythm_kernel_failed", False)
                and rhythm_flat is not None
                and rhythm_flat.is_cuda
                and rhythm_flat.dtype == torch.float32
            ):
                try:
                    from .kernels.rhythm import rhythm_features_update_triton

                    fuse_motif = (
                        self.hparams.scan.get("motif_writer", "torch") == "triton"
                        and self.motif is not None and motif_ids is not None and motif_iats is not None
                        and motif_iats.dtype == torch.float32 and not motif_iats.requires_grad
                    )
                    features = rhythm_features_update_triton(
                        rhythm_flat, flat_src, flat_dst, iat_t, gathered.same_id,
                        **({"motif_ids": motif_ids, "motif_iats": motif_iats,
                            "lane_idx": lane_idx, "dst_idx": dst_idx} if fuse_motif else {}),
                    )
                except Exception as exc:
                    # Import/compilation/launch failures before execution use torch.
                    self._rhythm_kernel_failed = True
                    warnings.warn(f"Triton rhythm kernel unavailable; using torch: {exc}", RuntimeWarning, stacklevel=2)
                else:
                    gathered.rhythm_updated = True
                    gathered.motif_updated = fuse_motif
            if features is None:
                if rhythm_flat is not None:
                    src_rows = self._flat_index_select(rhythm_flat, flat_src)
                    dst_rows = self._flat_index_select(rhythm_flat, flat_dst)
                else:
                    src_rows = rhythm_state[lane_idx, src_idx]
                    dst_rows = rhythm_state[lane_idx, dst_idx]
                gathered.rhythm_src_rows = src_rows
                gathered.rhythm_dst_rows = dst_rows
                rhythm_context = self.rhythm.read_context_rows_batch(src_rows, dst_rows)
            else:
                rhythm_context = self.rhythm.proj(features)
            x_with_context = self._add_lane_context(x_t, x_with_context, rhythm_context)
            if profile_scan_subphases:
                self._scan_subphase_end("rhythm_context", subphase_start)

        if self.motif is not None and motif_ids is not None and motif_iats is not None:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            assert motif_ids_snapshot is not None and motif_iats_snapshot is not None
            encoded_time = self._try_time_kernel(
                "motif", motif_iats_snapshot, self.motif.time_encoder.freqs, self.motif.time_encoder.dim,
            )
            motif_context = self.motif.read_context_batch(
                motif_ids_snapshot, motif_iats_snapshot, encoded_time=encoded_time,
            )
            x_with_context = self._add_lane_context(x_t, x_with_context, motif_context)
            if profile_scan_subphases:
                self._scan_subphase_end("motif_context", subphase_start)

        return x_t if x_with_context is None else x_with_context, gathered

    def _write_lane_step_state(
        self,
        *,
        gathered: _LaneStepGather,
        features_2d: torch.Tensor,
        step_idx: int,
        z_t: torch.Tensor,
        next_backbone: torch.Tensor | None,
        backbone_state: torch.Tensor | None,
        memory_state: torch.Tensor | None,
        last_seen: torch.Tensor | None,
        rhythm_state: torch.Tensor | None,
        motif_ids: torch.Tensor | None,
        motif_iats: torch.Tensor | None,
        memory_flat: torch.Tensor | None,
        last_seen_flat: torch.Tensor | None,
        rhythm_flat: torch.Tensor | None,
        profile_scan_subphases: bool,
    ) -> torch.Tensor | None:
        if gathered.all_lanes_active:
            features_2d[:, step_idx] = z_t
        else:
            features_2d[gathered.active, step_idx] = z_t

        if next_backbone is not None:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            if backbone_state is None:
                backbone_state = features_2d.new_zeros(
                    (next_backbone.size(0), features_2d.size(0), features_2d.size(-1))
                )
            if gathered.all_lanes_active:
                backbone_state = next_backbone
            else:
                backbone_state[:, gathered.active, :] = next_backbone
            if profile_scan_subphases:
                self._scan_subphase_end("backbone_write", subphase_start)

        if self.memory is not None and memory_state is not None:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            assert gathered.memory_src_rows is not None
            assert gathered.memory_dst_rows is not None
            src_new = self.memory.update_cell(z_t, gathered.memory_src_rows)
            dst_input = torch.where(gathered.same_id.unsqueeze(-1), src_new, gathered.memory_dst_rows)
            dst_new = self.memory.update_cell(z_t, dst_input)

            if memory_flat is not None:
                self._flat_index_copy(memory_flat, gathered.flat_src, src_new)
                self._flat_index_copy(memory_flat, gathered.flat_dst, dst_new)
            else:
                memory_state[gathered.lane_idx, gathered.src_idx] = src_new
                memory_state[gathered.lane_idx, gathered.dst_idx] = dst_new
            if last_seen is not None:
                now = gathered.t_t.to(device=last_seen.device, dtype=last_seen.dtype)
                if last_seen_flat is not None:
                    self._flat_index_copy(last_seen_flat, gathered.flat_src, now)
                    self._flat_index_copy(last_seen_flat, gathered.flat_dst, now)
                else:
                    last_seen[gathered.lane_idx, gathered.src_idx] = now
                    last_seen[gathered.lane_idx, gathered.dst_idx] = now
            if profile_scan_subphases:
                self._scan_subphase_end("memory_update", subphase_start)

        if self.rhythm is not None and rhythm_state is not None and not gathered.rhythm_updated:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            assert gathered.rhythm_src_rows is not None
            assert gathered.rhythm_dst_rows is not None
            src_new, dst_new = self.rhythm._updated_rows_indexed_batch(
                gathered.rhythm_src_rows,
                gathered.rhythm_dst_rows,
                gathered.iat_t,
                gathered.same_id,
            )
            if rhythm_flat is not None:
                self._flat_index_copy(rhythm_flat, gathered.flat_src, src_new)
                self._flat_index_copy(rhythm_flat, gathered.flat_dst, dst_new)
            else:
                rhythm_state[gathered.lane_idx, gathered.src_idx] = src_new
                rhythm_state[gathered.lane_idx, gathered.dst_idx] = dst_new
            if profile_scan_subphases:
                self._scan_subphase_end("rhythm_update", subphase_start)

        if self.motif is not None and motif_ids is not None and motif_iats is not None and not gathered.motif_updated:
            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            if motif_ids.size(1) > 1:
                motif_ids[gathered.lane_idx, :-1] = motif_ids[gathered.lane_idx, 1:]
                motif_iats[gathered.lane_idx, :-1] = motif_iats[gathered.lane_idx, 1:]
            motif_ids[gathered.lane_idx, -1] = gathered.dst_idx.to(device=motif_ids.device)
            motif_iats[gathered.lane_idx, -1] = gathered.iat_t.to(device=motif_iats.device, dtype=motif_iats.dtype)
            if profile_scan_subphases:
                self._scan_subphase_end("motif_update", subphase_start)

        return backbone_state

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
        *,
        regular_scored: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        phase_start = self._phase_start()
        encoded = self.encoder(batch)
        self._phase_end("encode", phase_start)
        phase_start = self._phase_start()
        lanes, chunk, hidden = encoded.shape
        current_state = self._ensure_lane_state(state, encoded)
        initial_state = self._initial_lane_state_like(encoded)

        lane_reset = batch.lane_reset.to(device=encoded.device, dtype=torch.bool)
        section_end = getattr(batch, "lane_section_end", None)
        lane_end = (batch.lane_stream_end if section_end is None else section_end).to(
            device=encoded.device, dtype=torch.bool
        )
        if regular_scored:
            assert state is not None
            memory_state = current_state.get("memory")
            last_seen = current_state.get("last_seen")
            rhythm_state = current_state.get("rhythm")
            motif_ids = current_state.get("motif_ids")
            motif_iats = current_state.get("motif_iats")
            prev_t = current_state.get("prev_t")
            backbone_state = current_state.get("backbone")
        else:
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
            if regular_scored and prev_t is not None:
                prev_t = t_all[:, -1].clone()
            elif prev_t is not None and valid.any():
                widths = valid.long().sum(dim=1)
                has_events = widths > 0
                if has_events.any():
                    prev_t = prev_t.clone()
                    event_lanes = torch.arange(lanes, device=encoded.device)[has_events]
                    last_pos = widths[has_events] - 1
                    prev_t[event_lanes] = t_all[event_lanes, last_pos].to(device=prev_t.device, dtype=prev_t.dtype)
            reset_memory = bool(self.hparams.memory.get("reset_on_stream_end", True))
            if not regular_scored and lane_end.any():
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
            self._phase_end("scan", phase_start)
            heads_start = self._phase_start()
            out = self._heads_from_features(features, batch)
            self._phase_end("heads", heads_start)
            return out, next_state

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

        profile_scan_subphases = self._scan_subphase_timing_enabled()
        num_ids = int(self.hparams.num_ids)
        memory_flat = memory_state.reshape(lanes * num_ids, hidden) if memory_state is not None else None
        last_seen_flat = last_seen.reshape(lanes * num_ids) if last_seen is not None else None
        rhythm_flat = rhythm_state.reshape(lanes * num_ids, 3) if rhythm_state is not None else None
        full_chunk_fastpath = regular_scored or bool(valid.all())
        lane_offsets = lane_ids_all.view(lanes, 1) * num_ids
        flat_src_all = lane_offsets + src_idx_all
        flat_dst_all = lane_offsets + dst_idx_all

        for step_idx in range(chunk):
            x_t, gathered = self._lane_step_context(
                encoded=encoded,
                valid=valid,
                src_idx_all=src_idx_all,
                dst_idx_all=dst_idx_all,
                iat_all=iat_all,
                t_all=t_all,
                lane_ids_all=lane_ids_all,
                step_idx=step_idx,
                num_ids=num_ids,
                full_chunk_fastpath=full_chunk_fastpath,
                profile_scan_subphases=profile_scan_subphases,
                memory_state=memory_state,
                last_seen=last_seen,
                rhythm_state=rhythm_state,
                motif_ids=motif_ids,
                motif_iats=motif_iats,
                memory_flat=memory_flat,
                last_seen_flat=last_seen_flat,
                rhythm_flat=rhythm_flat,
                flat_src_all=flat_src_all,
                flat_dst_all=flat_dst_all,
            )
            if gathered is None:
                continue

            subphase_start = self._scan_subphase_start() if profile_scan_subphases else None
            if backbone_state is None:
                active_backbone = None
            elif gathered.all_lanes_active:
                active_backbone = backbone_state
            else:
                active_backbone = backbone_state[:, gathered.active, :]
            z_t, next_backbone = self.backbone.step(x_t, active_backbone)
            if profile_scan_subphases:
                self._scan_subphase_end("backbone", subphase_start)

            backbone_state = self._write_lane_step_state(
                gathered=gathered,
                features_2d=features_2d,
                step_idx=step_idx,
                z_t=z_t,
                next_backbone=next_backbone,
                backbone_state=backbone_state,
                memory_state=memory_state,
                last_seen=last_seen,
                rhythm_state=rhythm_state,
                motif_ids=motif_ids,
                motif_iats=motif_iats,
                memory_flat=memory_flat,
                last_seen_flat=last_seen_flat,
                rhythm_flat=rhythm_flat,
                profile_scan_subphases=profile_scan_subphases,
            )
        if regular_scored and prev_t is not None:
            prev_t = t_all[:, -1].clone()
        elif prev_t is not None and valid.any():
            widths = valid.long().sum(dim=1)
            has_events = widths > 0
            if has_events.any():
                prev_t = prev_t.clone()
                event_lanes = lane_ids_all[has_events]
                last_pos = widths[has_events] - 1
                prev_t[event_lanes] = t_all[event_lanes, last_pos].to(device=prev_t.device, dtype=prev_t.dtype)

        reset_memory = bool(self.hparams.memory.get("reset_on_stream_end", True))
        if not regular_scored and lane_end.any():
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
        self._phase_end("scan", phase_start)
        heads_start = self._phase_start()
        out = self._heads_from_features(features, batch)
        self._phase_end("heads", heads_start)
        return out, next_state

    def _forward_with_state(
        self,
        batch,
        state: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        if getattr(batch, "valid_mask", None) is not None:
            return self._forward_lane_with_state(batch, state)
        phase_start = self._phase_start()
        encoded = self.encoder(batch)
        self._phase_end("encode", phase_start)
        phase_start = self._phase_start()
        if encoded.numel() == 0:
            empty = encoded.new_empty((0, int(self.hparams.input["hidden"])))
            next_state = self._initial_state_like(encoded)
            self._phase_end("scan", phase_start)
            heads_start = self._phase_start()
            out = self._heads_from_features(empty, batch)
            self._phase_end("heads", heads_start)
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
        self._phase_end("scan", phase_start)
        heads_start = self._phase_start()
        out = self._heads_from_features(features, batch)
        self._phase_end("heads", heads_start)
        return out, next_state
