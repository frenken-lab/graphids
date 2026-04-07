// Weighted-average fusion baseline.
// Ported from graphids/config/fusion/methods/weighted_avg.yaml.

{
  model: {
    class_path: 'graphids.core.models.fusion.fusion_baselines.WeightedAvgModule',
    init_args: {
      lr: 0.01,
      decision_threshold: 0.5,
    },
  },
}
