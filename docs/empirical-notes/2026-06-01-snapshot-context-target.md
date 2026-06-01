# 2026-06-01 — Snapshot context-to-target representation

`snapshot_sequence` should not mean "one label for any attack anywhere in the
sequence." That muddies detection timing and makes validation analysis harder.

Use sequence context with a local target:

```text
context: [w(t-2), w(t-1), w(t)]
target:  w(t)
label:   y(t)
```

Each sample still contains `sequence_length` snapshot graphs, but supervised
training makes one prediction for the target/current snapshot window. Earlier
sequence steps are context only.

For `sequence_length=3`, samples look like:

```text
[w0, w1, w2] -> y2
[w1, w2, w3] -> y3
[w2, w3, w4] -> y4
```

Split safety rule:

- train and validation samples must not share underlying snapshot windows
- validation should be chosen from blocked window ranges with enough positive
  target labels to make AUROC meaningful

The old chunk-level `max(y over context)` semantics should not be used for GAT
model selection.
