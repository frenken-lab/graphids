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

The main thing to decompose next is the sample-shape decision.
Today that decision lives inside `GraphPipeline`; that is the bundling
that blocks snapshot-sequence and multi-scale views from being first-class.
