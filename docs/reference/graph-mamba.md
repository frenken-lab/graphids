# Retired snapshot graph–Mamba

The snapshot implementation, configurations, launch scripts and dedicated tests
were retired on 2026-10-16. Restore the exact working tree from
`archives/snapshot-2026-10-16.tar.gz` in a separate checkout; see
[archive instructions](snapshot-archive.md). The companion manifest records
file hashes, original Git state, and extraction smoke verification.

Historical evidence in `docs/empirical-notes/` and external datasets, manifests,
clock repairs, checkpoints, calibration and results remain preserved. Use the
archived environment and scripts for their reproduction.

Active work uses [message-preserving events](graph-mamba-events.md) on one GPU.
Snapshot checkpoints and calibration cannot be loaded into the event model.
