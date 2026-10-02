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


class _ScoredAdapter(nn.Module):
    def __init__(self, model, batch, state):
        super().__init__()
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

    def inputs(self, batch, state):
        return tuple(getattr(batch, key) for key in self.batch_keys) + tuple(
            state[key] for key in self.state_keys
        )

    def forward(self, *inputs):
        batch = SimpleNamespace(
            **dict(zip(self.batch_keys, inputs[: len(self.batch_keys)], strict=True))
        )
        state = dict(self.empty_state)
        state.update(zip(self.state_keys, inputs[len(self.batch_keys) :], strict=True))
        out, next_state = self.model._forward_lane_with_state(
            batch, state, regular_scored=True
        )
        terms = self.model._loss_terms(out, batch, regular_scored=True)
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
        self.grad_hooks = []
        self.signature = None
        self.disabled = False
        self.setup_error = None
        self.capture_seconds = 0.0
        self.replays = 0
        self.scored_replay_events = 0
        self.scored_events = 0
        self.fallbacks = Counter()

    def run(self, model, batch, state, reason):
        self.scored_events += int(getattr(model, "_cpu_scored_events", 0))
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
            elif self.disabled:
                reason = "capture_disabled"
        if reason is not None:
            self.fallbacks[reason] += 1
            return None
        candidate = self.adapter or _ScoredAdapter(model, batch, state)
        inputs = candidate.inputs(batch, state)
        signature = _signature(inputs)
        if self.adapter is not None and signature != self.signature:
            self.fallbacks["geometry_changed"] += 1
            return None
        if self.adapter is None:
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
                self.grad_hooks = [
                    parameter.register_hook(lambda grad: grad.clone())
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.adapter = candidate
                self.signature = signature
            except Exception as exc:
                self.disabled = True
                self.setup_error = traceback.format_exc()
                self.fallbacks["capture_setup_failed"] += 1
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
        result = self.adapter.unpack(self.adapter(*inputs))
        self.replays += 1
        self.scored_replay_events += batch.y.numel()
        return result

    def close(self):
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks.clear()
        self.adapter = None

    def report(self):
        return {
            "capture_seconds": self.capture_seconds,
            "replays": self.replays,
            "scored_replay_events": self.scored_replay_events,
            "scored_events": self.scored_events,
            "scored_replay_fraction": self.scored_replay_events
            / max(1, self.scored_events),
            "fallback_batches": dict(self.fallbacks),
            "disabled": self.disabled,
            "setup_error": self.setup_error,
        }
