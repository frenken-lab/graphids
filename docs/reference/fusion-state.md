# Fusion State

> Status: **historical**

The old fusion cache belonged to the windowed GAT/VGAE training stack. That
stack was removed when `TemporalData` became the primary training surface.

The current live model families consume temporal event batches directly:

- `temporal_event_classifier`
- `temporal_rnn_classifier`
- `temporal_gat`
- `temporal_vgae`

Future fusion work should start from temporal event predictions or temporal
state features, then aggregate to segment/source-level decisions where needed.
Do not build new code against the old fixed-width graph-level fusion state.
