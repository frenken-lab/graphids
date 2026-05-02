// Focal loss — class_path block. orchestrate._instantiate recursively
// builds it before constructing the model.

function(gamma=2.0)
  { loss_fn: {
      class_path: 'graphids.core.losses.FocalLoss',
      init_args: { gamma: gamma },
  } }
