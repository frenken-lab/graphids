// Supervised model family — GAT classification.
//
// Pre-merged configs indexed by scale: sup[model_type][scale].
// num_workers is left to the autosizer (budget.py:autosize_workers,
// `ceil((t_io + t_collation) / t_gpu)` capped at SLURM_CPUS_PER_TASK - 2).
// Hardcoded overrides go stale when the data path changes — see the prior
// `num_workers: 4` removed once prebatching landed.
//
// KD is handled at the loss level (core/losses/distillation.py), not here.

local _gat_base = {
  model: {
    class_path: 'graphids.core.models.supervised.gat_module.GATModule',
    init_args: {
      conv_type: 'gatv2',
      edge_dim: 11,
      loss_fn: 'ce',
      focal_gamma: 2.0,
      loss_weight: 10.0,
      pool_aggrs: ['mean'],
      compile_model: false,
    },
  },
};

{
  gat: {
    small: _gat_base + {
      model+: { init_args+: {
        scale: 'small',
        hidden: 24,
        layers: 2,
        heads: 4,
        embedding_dim: 8,
        dropout: 0.1,
        proj_dim: 32,
        fc_layers: 2,
      } },
    },
    large: _gat_base + {
      model+: { init_args+: {
        scale: 'large',
        hidden: 64,
        layers: 3,
        heads: 4,
        embedding_dim: 8,
        dropout: 0.11,
        proj_dim: 48,
        fc_layers: 4,
      } },
    },
  },
}
