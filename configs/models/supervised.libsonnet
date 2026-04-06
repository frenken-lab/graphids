// Supervised model family — GAT classification.
//
// Indexed by the supervised stage via model_type TLA (currently only 'gat').
// GAT is compute-bound (cg_ratio ≈ 0.21 at W=6); base sets num_workers=4
// to override the auto-sized default.
//
// KD is handled at the loss level (core/losses/distillation.py), not here.

{
  gat: {
    base: {
      model: {
        class_path: 'graphids.core.models.supervised.gat.GATModule',
        init_args: {
          conv_type: 'gatv2',
          edge_dim: 11,
          loss_fn: 'ce',
          focal_gamma: 2.0,
          loss_weight: 10.0,
          pool_aggrs: ['mean'],
          compile_model: true,
        },
      },
      data: {
        init_args: {
          num_workers: 4,
        },
      },
    },

    scales: {
      small: {
        model+: {
          init_args+: {
            scale: 'small',
            hidden: 24,
            layers: 2,
            heads: 4,
            embedding_dim: 8,
            dropout: 0.1,
            proj_dim: 32,
            fc_layers: 2,
          },
        },
      },
      large: {
        model+: {
          init_args+: {
            scale: 'large',
            hidden: 64,
            layers: 3,
            heads: 4,
            embedding_dim: 8,
            dropout: 0.11,
            proj_dim: 48,
            fc_layers: 4,
          },
        },
      },
    },
  },
}
