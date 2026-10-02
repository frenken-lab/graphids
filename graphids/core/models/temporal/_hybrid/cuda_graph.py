"""Partial-network CUDA capture; optimizer and recording ownership stay in Lightning."""

from __future__ import annotations

import time
import traceback
import warnings
from collections import Counter
from types import SimpleNamespace

import torch
from torch import nn

_BATCH_FIELDS = (
    "src",
    "dst",
    "t",
    "msg",
    "y",
    "valid_mask",
    "is_scored",
    "lane_reset",
    "lane_stream_end",
    "lane_section_end",
    "reset_after",
    "successor_index",
    "successor_dst",
    "successor_msg",
)
_MIXED_SCORED_EVENTS = 8160


def eligibility_reason(batch) -> str | None:
    """Inspect CPU metadata before transfer; unknown or already moved batches fall back."""
    valid = getattr(batch, "valid_mask", None)
    if valid is None or valid.ndim != 2:
        return "non_lane_batch"
    metadata = (
        "valid_mask",
        "is_scored",
        "lane_reset",
        "lane_stream_end",
        "reset_after",
        "successor_index",
        "successor_dst",
        "successor_msg",
    )
    for name in metadata:
        value = getattr(batch, name, None)
        if value is None:
            return "missing_successor_or_metadata"
        if value.device.type != "cpu":
            return "metadata_not_cpu"
    if not valid.numel() or not bool(valid.all()):
        return "partial_lanes"
    if not bool(batch.is_scored.all()):
        return "warmup_or_mixed"
    for name in (
        "lane_reset",
        "lane_stream_end",
        "lane_section_end",
        "reset_after",
        "warmup_mask",
    ):
        value = getattr(batch, name, None)
        if value is not None and (value.device.type != "cpu" or bool(value.any())):
            return name
    if not bool((batch.successor_index >= 0).all()):
        return "missing_successor"
    return None


def mixed_scored_index(batch) -> torch.Tensor | None:
    """CPU indices for one fixed-shape mixed case with no state boundaries."""
    valid, scored = getattr(batch, "valid_mask", None), getattr(batch, "is_scored", None)
    if valid is None or scored is None or valid.device.type != "cpu" or scored.device.type != "cpu":
        return None
    if tuple(valid.shape) != (256, 32) or not bool(valid.all()):
        return None
    index = scored.reshape(-1).nonzero().reshape(-1)
    if index.numel() != _MIXED_SCORED_EVENTS:
        return None
    for name in ("lane_reset", "lane_stream_end", "lane_section_end", "reset_after"):
        value = getattr(batch, name, None)
        if value is not None and (value.device.type != "cpu" or bool(value.any())):
            return None
    successor = getattr(batch, "successor_index", None)
    if successor is None or successor.device.type != "cpu" or not bool((successor.reshape(-1)[index] >= 0).all()):
        return None
    return index


class _ScoredAdapter(nn.Module):
    def __init__(self, model, batch, state, *, mixed=False):
        super().__init__()
        self.mixed = mixed
        self.model = (
            model  # Shared parameters must be explicit inputs to graphed autograd.
        )
        self.batch_keys = tuple(
            key for key in _BATCH_FIELDS if getattr(batch, key, None) is not None
        )
        self.state_keys = tuple(
            key for key, value in state.items() if value is not None
        )
        self.empty_state = {key: None for key, value in state.items() if value is None}
        self.output_keys = ()
        self.term_keys = ()

    def inputs(self, batch, state, scored_index=None):
        values = tuple(getattr(batch, key) for key in self.batch_keys) + tuple(
            state[key] for key in self.state_keys
        )
        return values + ((scored_index,) if self.mixed else ())

    def forward(self, *inputs):
        batch = SimpleNamespace(
            **dict(zip(self.batch_keys, inputs[: len(self.batch_keys)], strict=True))
        )
        state = dict(self.empty_state)
        state_end = len(self.batch_keys) + len(self.state_keys)
        state.update(zip(self.state_keys, inputs[len(self.batch_keys) : state_end], strict=True))
        out, next_state = self.model._forward_lane_with_state(
            batch, state, regular_scored=True
        )
        terms = self.model._loss_terms(
            out, batch, regular_scored=not self.mixed,
            scored_index=inputs[-1] if self.mixed else None,
        )
        self.output_keys = tuple(out)
        self.term_keys = tuple(terms)
        return (
            tuple(value.detach() for value in out.values())
            + tuple(
                value if key == "total" else value.detach()
                for key, value in terms.items()
            )
            + tuple(next_state[key].detach() for key in self.state_keys)
        )

    def unpack(self, values):
        n, m = len(self.output_keys), len(self.term_keys)
        # Graph output storage is reused. Own every retained tensor, including losses.
        out = {
            key: value.detach().clone()
            for key, value in zip(self.output_keys, values[:n], strict=True)
        }
        terms = {
            key: value.clone() if key == "total" else value.detach().clone()
            for key, value in zip(self.term_keys, values[n : n + m], strict=True)
        }
        state = dict(self.empty_state)
        state.update(
            (key, value.detach().clone())
            for key, value in zip(self.state_keys, values[n + m :], strict=True)
        )
        return out, state, terms


def _signature(inputs):
    return tuple((tuple(v.shape), v.dtype, v.device, v.requires_grad) for v in inputs)


class ScoredCudaGraph:
    """One fixed geometry per model; capture failures disable capture before live replay."""

    def __init__(self):
        self.adapter = None
        self.mixed_adapter = None
        self.grad_hooks = []
        self.signature = None
        self.mixed_signature = None
        self.disabled = False
        self.mixed_disabled = False
        self.setup_error = None
        self.mixed_setup_error = None
        self.capture_seconds = 0.0
        self.replays = 0
        self.mixed_replays = 0
        self.scored_replay_events = 0
        self.scored_events = 0
        self.fallbacks = Counter()

    def run(self, model, batch, state, reason):
        self.scored_events += int(getattr(model, "_cpu_scored_events", 0))
        mixed = (reason == "warmup_or_mixed"
                 and getattr(model, "_cpu_scored_index", None) is not None)
        if mixed:
            reason = None
        if reason is None:
            if not batch.msg.is_cuda:
                reason = "cpu"
            elif batch.msg.dtype != torch.float32 or torch.is_autocast_enabled("cuda"):
                reason = "requires_fp32"
            elif model._phase_timing_enabled():
                reason = "phase_profiling"
            elif state is None or (
                model.backbone.backbone_type != "none" and state.get("backbone") is None
            ):
                reason = "uninitialized_state"
            elif (mixed and self.mixed_disabled) or (not mixed and self.disabled):
                reason = "mixed_capture_disabled" if mixed else "capture_disabled"
        if reason is not None:
            self.fallbacks[reason] += 1
            return None
        adapter = self.mixed_adapter if mixed else self.adapter
        candidate = adapter or _ScoredAdapter(model, batch, state, mixed=mixed)
        scored_index = (model._cpu_scored_index.to(device=batch.msg.device, non_blocking=True)
                        if mixed else None)
        inputs = candidate.inputs(batch, state, scored_index)
        signature = _signature(inputs)
        saved_signature = self.mixed_signature if mixed else self.signature
        if adapter is not None and signature != saved_signature:
            key = "mixed_geometry_changed" if mixed else "geometry_changed"
            self.fallbacks[key] += 1
            return None
        if adapter is None:
            started = time.perf_counter()
            buffers = [(buf, buf.clone()) for buf in model.buffers()]
            grads = [(p, p.grad) for p in model.parameters()]
            try:
                # Warmup sees owned inputs. make_graphed_callables warms up on a side
                # stream, compiling Triton before capture, without optimizer steps.
                samples = tuple(value.detach().clone() for value in inputs)
                with torch.random.fork_rng(devices=[batch.msg.device.index]):
                    # Keep warmup allocations and backward seed gradients off the
                    # legacy stream, including native dropout backward dependencies.
                    stream = torch.cuda.Stream(device=batch.msg.device)
                    stream.wait_stream(torch.cuda.current_stream(batch.msg.device))
                    with torch.cuda.stream(stream):
                        torch.cuda.make_graphed_callables(
                            candidate, samples, allow_unused_input=True
                        )
                    torch.cuda.current_stream(batch.msg.device).wait_stream(stream)
                # make_graphed_callables returns reusable gradient buffers.
                # Autograd can adopt them as Parameter.grad on the first backward;
                # own those gradients before another replay or accumulation.
                if not self.grad_hooks:
                    self.grad_hooks = [
                        parameter.register_hook(lambda grad: grad.clone())
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                if mixed:
                    self.mixed_adapter = candidate
                    self.mixed_signature = signature
                else:
                    self.adapter = candidate
                    self.signature = signature
            except Exception as exc:
                if mixed:
                    self.mixed_disabled = True
                    self.mixed_setup_error = traceback.format_exc()
                else:
                    self.disabled = True
                    self.setup_error = traceback.format_exc()
                key = "mixed_capture_setup_failed" if mixed else "capture_setup_failed"
                self.fallbacks[key] += 1
                warnings.warn(
                    f"CUDA graph setup failed; using eager batches: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            finally:
                with torch.no_grad():
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                    for parameter, grad in grads:
                        parameter.grad = grad
                self.capture_seconds += time.perf_counter() - started
        # Never catch replay errors: a recording batch may already have executed.
        adapter = self.mixed_adapter if mixed else self.adapter
        result = adapter.unpack(adapter(*inputs))
        self.replays += 1
        if mixed:
            self.mixed_replays += 1
            self.scored_replay_events += _MIXED_SCORED_EVENTS
        else:
            self.scored_replay_events += batch.y.numel()
        return result

    def close(self):
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks.clear()
        self.adapter = None
        self.mixed_adapter = None

    def report(self):
        return {
            "capture_seconds": self.capture_seconds,
            "replays": self.replays,
            "mixed_replays": self.mixed_replays,
            "scored_replay_events": self.scored_replay_events,
            "scored_events": self.scored_events,
            "scored_replay_fraction": self.scored_replay_events
            / max(1, self.scored_events),
            "fallback_batches": dict(self.fallbacks),
            "disabled": self.disabled,
            "mixed_disabled": self.mixed_disabled,
            "mixed_setup_error": self.mixed_setup_error,
            "setup_error": self.setup_error,
        }
