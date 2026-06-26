# CAN bus segmenter draft

The current pipeline hard-codes one shape:

- raw rows -> one sliding window -> one graph

That is fine for the baseline snapshot view, but it is too rigid for the
views we want next.

Proposed segment primitives:

- `window`
  - the current behavior
  - one fixed-time window becomes one graph sample

- `sequence`
  - several consecutive windows become one training example
  - this is the best next step when a single snapshot is not predictive enough

- `multi_scale`
  - multiple window sizes over the same raw rows
  - useful when anomalies happen at different temporal scales

- `entity`
  - center the segment on one arbitration ID or entity family
  - useful for attribution and local behavior analysis

What should stay separate:

- raw CAN normalization and payload parsing
- vocabulary scan and dataset metadata
- graph feature construction
- tensor packing

The sample-shape decision has since moved to `representation_cfg` and
`build_graph_tables`. Snapshot and snapshot-sequence are first-class; the
other segment families in this draft remain future work.
