// MLP fusion baseline (supervised logistic combination).
// Ported from graphids/config/fusion/methods/mlp.yaml.

{
  model: {
    class_path: 'graphids.core.models.fusion.mlp.MLPFusionModule',
    init_args: {
      lr: 0.001,
      hidden_dims: [64, 32],
    },
  },
}
