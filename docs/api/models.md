# Core: Models

The live model surface consumes PyG `TemporalData` event batches.

Current temporal model families:

- `TemporalEventClassifier`: stateless MLP baseline over event messages plus ID
  embeddings.
- `TemporalRNNClassifier`: supervised GRU classifier that carries hidden state
  across adjacent `TemporalDataLoader` batches and resets at `reset_after`
  boundaries.
- `TemporalGAT`: causal event-attention classifier.
- `TemporalVGAE`: recurrent variational event autoencoder for event-level
  surprise scoring.
- `TemporalHybridModel`: modular temporal hybrid for supervised, anomaly, or
  joint learning. It composes an event encoder, optional TGN-style arbitration
  ID memory, a stream backbone (`none`, `gru`, `ssm_lite`, or optional
  `mamba`), and independently enabled classifier / self-supervised anomaly
  heads. Its memory can include per-ID elapsed-time encodings, and optional
  rhythm and motif contexts add causal CAN schedule features before the stream
  backbone.
- `id_encoding/`: categorical-ID encoders with reserved `UNK` at index `0`.

`TemporalHybridModel` is configured through the `temporal_hybrid` primitive.
Use `objective: supervised` for classifier-only training, `objective: anomaly`
for self-supervised attack-free anomaly scoring, and `objective: joint` when a
classifier and anomaly heads should train together. Runtime dimensions
(`num_ids`, `in_channels`, and `num_classes`) are still injected from the
`TemporalDataModule` after `setup()`, so YAML configs do not hard-code dataset
vocabulary sizes.

Anomaly heads default to the original regression/error scoring mode. Set
`anomaly.mode: nll` to score IAT and payload-delta heads with Gaussian negative
log-likelihood while keeping next-ID categorical NLL.

## `graphids.core.models`

::: graphids.core.models
    options:
      show_submodules: true
