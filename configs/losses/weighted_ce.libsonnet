// Class-weighted cross-entropy — class_path block. `weights` is required
// (per-dataset class distribution).

function(weights)
  { loss_fn: {
      class_path: 'graphids.core.losses.WeightedCrossEntropyLoss',
      init_args: { weights: weights },
  } }
