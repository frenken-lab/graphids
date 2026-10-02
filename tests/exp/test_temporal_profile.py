import json

import pytest

from graphids.exp.temporal_profile import summarize_trace, union_us


def test_busy_time_counts_concurrent_kernels_once(tmp_path):
    events = [
        {"ph": "X", "name": "temporal/train_batch", "cat": "user_annotation", "ts": 0, "dur": 100},
        {"ph": "X", "name": "temporal/train_batch", "cat": "gpu_user_annotation", "ts": 0, "dur": 100},
        {"ph": "X", "name": "first", "cat": "kernel", "ts": 10, "dur": 30},
        {"ph": "X", "name": "second", "cat": "kernel", "ts": 20, "dur": 40},
        {"ph": "X", "name": "copy", "cat": "gpu_memcpy", "ts": 80, "dur": 40},
        {"ph": "X", "name": "outside", "cat": "kernel", "ts": 200, "dur": 30},
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    report = summarize_trace(trace)
    assert report["device_busy_fraction"] == pytest.approx(0.7)
    assert report["captured_batches"] == 1
    assert len(report["top_device_and_runtime"]) == 3
    assert union_us([]) == 0
