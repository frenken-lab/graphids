// Weighted-average fusion baseline.

function(lr=0.01, decision_threshold=0.5)
  {
    model: {
      class_path: 'graphids.core.models.fusion.weighted_avg.WeightedAvgModule',
      init_args: { lr: lr, decision_threshold: decision_threshold },
    },
  }
