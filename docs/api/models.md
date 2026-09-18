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
- `id_encoding/`: categorical-ID encoders with reserved `UNK` at index `0`.

## `graphids.core.models`

::: graphids.core.models
    options:
      show_submodules: true
