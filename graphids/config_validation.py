"""Shared constructor/config validation; no training framework imports."""

from __future__ import annotations

from typing import Any


def validate_lane_config(options: Any, *, val_split: str | None = None) -> None:
    if options.batch_mode not in {"events", "stream_lanes"}:
        raise ValueError("batch_mode must be one of: events, stream_lanes")
    if options.coverage_policy not in {"full", "diagnostic"}:
        raise ValueError("coverage_policy must be full or diagnostic")
    if options.stream_execution not in {"continuous", "segmented"}:
        raise ValueError("stream_execution must be continuous or segmented")
    if options.eval_batch_mode not in {"events", "stream_lanes"}:
        raise ValueError("eval_batch_mode must be events or stream_lanes")
    if options.batch_mode == "stream_lanes":
        if options.stream_lanes is None or options.stream_lanes < 2:
            raise ValueError("stream_lanes must be >= 2 for batch_mode='stream_lanes'")
        if options.chunk_size is None or options.chunk_size < 1:
            raise ValueError("chunk_size must be positive for batch_mode='stream_lanes'")
        if options.coverage_policy == "full" and options.drop_partial_lane_batches:
            raise ValueError("full coverage requires drop_partial_lane_batches: false")
    if options.stream_execution == "segmented" and options.batch_mode != "stream_lanes":
        raise ValueError("segmented execution requires stream_lanes")
    if options.eval_batch_mode == "stream_lanes" and (options.stream_lanes is None or options.chunk_size is None):
        raise ValueError("lane evaluation requires stream_lanes and chunk_size")
    if val_split == "recordings" and options.val_prefix_events is not None:
        raise ValueError("recording validation requires complete streams; remove val_prefix_events")


def validate_scan_config(scan: dict[str, Any]) -> None:
    if scan.get("rhythm_kernel", "torch") not in ("torch", "triton"):
        raise ValueError("scan.rhythm_kernel must be torch or triton")
    for key in ("time_kernel", "motif_writer", "ssm_kernel"):
        if scan.get(key, "torch") not in ("torch", "triton"):
            raise ValueError(f"scan.{key} must be torch or triton")
    if scan.get("motif_writer") == "triton" and scan.get("rhythm_kernel") != "triton":
        raise ValueError("scan.motif_writer=triton requires rhythm_kernel=triton")
    unknown_scan_keys = set(scan) - {"profile_subphases", "rhythm_kernel", "time_kernel", "motif_writer", "ssm_kernel", "cuda_graph"}
    if unknown_scan_keys:
        keys = ", ".join(sorted(unknown_scan_keys))
        raise ValueError(f"unknown scan config key(s): {keys}")


def validate_hybrid_config(*, objective, memory, backbone, heads, anomaly, motif, scan, loss_fn=None, rhythm=None):
    default_ssl = objective in {"anomaly", "joint"}
    heads = {"classification": objective in {"supervised", "joint"},
             **dict.fromkeys(("next_id", "iat", "payload_delta"), default_ssl),
             **{key: value for key, value in heads.items() if value is not None}}
    if str(memory.get("type", "tgn")) != "tgn":
        raise ValueError("memory.type must be 'tgn'")
    if int(memory.get("time_encoding_dim", 0)) < 0:
        raise ValueError("memory.time_encoding_dim must be non-negative")
    if str(backbone.get("type", "gru")) not in {"none", "gru", "ssm_lite", "mamba"}:
        raise ValueError("backbone.type must be one of: none, gru, ssm_lite, mamba")
    if str(anomaly.get("mode", "regression")) not in {"regression", "nll"}:
        raise ValueError("anomaly.mode must be one of: regression, nll")
    if float(anomaly.get("min_log_scale", -7.0)) > float(anomaly.get("max_log_scale", 5.0)):
        raise ValueError("anomaly.min_log_scale must be <= anomaly.max_log_scale")
    if int(motif.get("length", 1)) < 1:
        raise ValueError("motif.length must be positive")
    if motif.get("embedding_dim") is not None and int(motif["embedding_dim"]) < 1:
        raise ValueError("motif.embedding_dim must be positive")
    if motif.get("time_dim") is not None and int(motif["time_dim"]) < 1:
        raise ValueError("motif.time_dim must be positive")
    validate_scan_config(scan)
    if scan.get("ssm_kernel") == "triton" and backbone.get("type") != "ssm_lite":
        raise ValueError("scan.ssm_kernel=triton requires backbone.type=ssm_lite")
    classification = bool(heads.get("classification", False))
    anomaly_heads = [name for name in ("next_id", "iat", "payload_delta") if heads.get(name, False)]
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
