"""Target construction helpers for temporal hybrid anomaly heads."""

from __future__ import annotations

import torch


def payload_delta_slice(in_channels: int) -> slice:
    if in_channels >= 16:
        return slice(8, 16)
    return slice(0, in_channels)


def payload_delta_dim(in_channels: int) -> int:
    payload_slice = payload_delta_slice(in_channels)
    return int(payload_slice.stop - payload_slice.start)


def lane_targets(batch, *, in_channels: int) -> dict[str, torch.Tensor]:
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
    if int(in_channels) > 16 and batch.msg.shape[-1] > 16:
        iat = batch.msg[..., 16].to(device=device, dtype=batch.msg.dtype).clamp_min(0)
    elif chunk > 1:
        iat[:, 1:] = (
            batch.t[:, 1:].to(dtype=batch.msg.dtype) - batch.t[:, :-1].to(dtype=batch.msg.dtype)
        ).clamp_min(0)

    payload_slice = payload_delta_slice(int(in_channels))
    payload_delta = batch.msg[..., payload_slice].float().clone()
    if chunk > 1:
        payload_delta[:, :-1] = batch.msg[:, 1:, payload_slice].float()
    return {
        "valid_next": valid_next.reshape(-1),
        "next_id": next_id.reshape(-1),
        "iat": iat.reshape(-1),
        "payload_delta": payload_delta.reshape(lanes * chunk, -1),
    }


def event_targets(batch, *, in_channels: int) -> dict[str, torch.Tensor]:
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
    payload_slice = payload_delta_slice(int(in_channels))
    payload_delta = batch.msg[:, payload_slice].float().clone()
    if n > 1:
        payload_delta[:-1] = batch.msg[1:, payload_slice].float()
    return {"valid_next": valid_next, "next_id": next_id, "iat": iat, "payload_delta": payload_delta}


def event_iat(batch, idx: int, ref: torch.Tensor, *, in_channels: int) -> torch.Tensor:
    if int(in_channels) > 16 and batch.msg.shape[1] > 16:
        return batch.msg[idx, 16].to(device=ref.device, dtype=ref.dtype).clamp_min(0)
    t = getattr(batch, "t", None)
    if t is not None and idx > 0:
        return (
            t[idx].to(device=ref.device, dtype=ref.dtype)
            - t[idx - 1].to(device=ref.device, dtype=ref.dtype)
        ).clamp_min(0)
    return ref.new_tensor(0.0)


def prepare_scan_batch(batch, ref: torch.Tensor, *, num_ids: int, in_channels: int) -> dict[str, torch.Tensor | None]:
    src_idx = batch.src.to(device=ref.device).clamp_min(0).clamp_max(int(num_ids) - 1).long()
    dst_idx = batch.dst.to(device=ref.device).clamp_min(0).clamp_max(int(num_ids) - 1).long()
    n = int(src_idx.numel())

    t = getattr(batch, "t", None)
    t_cast = None if t is None else t.to(device=ref.device, dtype=ref.dtype)
    iat = ref.new_zeros(n)
    if int(in_channels) > 16 and batch.msg.shape[1] > 16:
        iat = batch.msg[:, 16].to(device=ref.device, dtype=ref.dtype).clamp_min(0)
    elif t_cast is not None and n > 1:
        iat[1:] = (t_cast[1:] - t_cast[:-1]).clamp_min(0)

    reset_after = getattr(batch, "reset_after", None)
    if reset_after is None:
        reset_after = torch.zeros(n, dtype=torch.bool, device=ref.device)
    else:
        reset_after = reset_after.to(device=ref.device, dtype=torch.bool)
    return {"src_idx": src_idx, "dst_idx": dst_idx, "iat": iat, "t": t_cast, "reset_after": reset_after}
