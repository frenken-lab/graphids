import csv

from graphids.core.data.recordings import Recording, file_hash


def write_record(
    tmp_path,
    *,
    key="capture",
    vehicle="v",
    role="train_query",
    normal=False,
    rows=None,
    session=None,
):
    path = tmp_path / f"{key}.csv"
    rows = rows or [
        (0.0, "A", "00", 0),
        (0.01, "A", "01", 0),
        (0.03, "B", "02", 1),
        (0.04, "A", "03", 0),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
        writer.writerows(rows)
    return Recording(
        key,
        "set_01",
        vehicle,
        session or key,
        str(path),
        file_hash(path),
        role,
        verified_normal=normal,
        payload_length_verified=True,
        labels_verified=True,
    )
