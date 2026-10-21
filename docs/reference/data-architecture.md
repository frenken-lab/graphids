# Event data architecture

Immutable CAN recordings and reviewed manifests are the source of truth.
`graphids.core.data.recordings` validates explicit source roles, hashes,
clock units, payload provenance and supervision. Reviewed clock repair mappings
remain external artifacts. Ingestion preserves physical order and rejects
decreasing clocks; it never silently sorts or repairs a source.

`can_event` is the only active representation. `can_event_dm` writes lossless
`event-v1-<recording-digest>` caches with unchanged content identities: FP64
physical times, all 29 ID bits, uint8 payload bytes, lengths and unknown labels.
Only selected source queries are opened. Reserved target recordings and reference
roles never become training queries.

`EventStreamWindows` schedules chronological chunks from independent recordings.
It groups equal lengths, carries separate row positions, processes short tails,
and replaces finished streams without padding, replay or dropped scored rows.
`EventTrainingWindow` declares complete classification and normal-payload target
totals for one optimizer update. The model detaches history between chunks.

`graphids.core.evaluation.EventScores` writes one bounded frame-score shard per
chronological chunk and verifies exact recording coverage before publishing the
index. Epoch AP consumes scalar columns via a disk mapping; node histories and
score blocks are not retained by the model. This artifact component has no torch
or Lightning dependency.

See [model and training contract](graph-mamba-events.md) and the
[retired snapshot archive](graph-mamba.md). The experiment CLI, Ray, Lightning
and Slurm infrastructure retain their existing roles.
