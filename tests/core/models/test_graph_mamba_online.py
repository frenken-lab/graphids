import torch

from graphids.core.data.snapshots.batch import recording_batch
from graphids.core.data.snapshots.manifest import read_rows
from graphids.core.data.snapshots.online import CausalCANBinner
from graphids.core.data.snapshots.preprocessing import (
    CANSnapshotRepresentationCfg,
    build_snapshots,
)
from tests.core.data.test_snapshots import write_record
from tests.core.models.test_graph_mamba import model


def test_online_raw_binning_matches_offline_model_with_expiry(tmp_path):
    rec = write_record(
        tmp_path,
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "B", "00", 0),
            (0.05, "A", "01", 0),
            (0.06, "B", "01", 0),
        ],
    )
    offline = build_snapshots(
        rec, CANSnapshotRepresentationCfg(), retention_seconds=0.015
    )
    batch = recording_batch(offline, None)
    net = model().eval()
    online = CausalCANBinner(rec, retention_seconds=0.015)
    bins = []
    for row in read_rows(rec):
        bins.extend(
            online.push(row["time"], row["id"], row["data"], label=row["label"])
        )
    bins.extend(online.finish())
    assert len(bins) == len(offline.closes)
    state = None
    predictions = []
    with torch.no_grad():
        expected, _ = net.stream(batch)
        for item in bins:
            output, state = net.stream(item, state)
            predictions.append(output["classification_logits"])
    torch.testing.assert_close(
        expected["classification_logits"],
        torch.cat(predictions, 1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert len(online.state.streams) <= online.transient_capacity
    assert sum(int(item.event_count.sum()) for item in bins) == len(read_rows(rec))
