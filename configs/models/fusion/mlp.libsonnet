// MLP fusion baseline (supervised logistic combination).

function(lr=0.001, hidden_dims=[64, 32])
  {
    model: {
      class_path: 'graphids.core.models.fusion.mlp.MLPFusionModule',
      init_args: { lr: lr, hidden_dims: hidden_dims },
    },
  }
