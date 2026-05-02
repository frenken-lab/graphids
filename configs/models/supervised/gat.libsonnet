// GAT model primitive — class_path + scale-tunable init_args.
// `loss` is merged in by the supervised archetype composer (each preset
// passes its loss block from `_lib/loss/<name>.libsonnet`).

local _scales = {
  small: { hidden: 24, layers: 2, heads: 4 },
  large: { hidden: 64, layers: 3, heads: 8 },
};

function(scale='small', conv_type='gatv2', dropout=0.3, lr=0.001)
  {
    model: {
      class_path: 'graphids.core.models.supervised.gat_module.GATModule',
      init_args: {
        conv_type: conv_type,
        edge_dim: 11,
        pool_aggrs: ['mean'],
        compile_model: false,
        scale: scale,
        dropout: dropout,
        lr: lr,
      } + _scales[scale],
    },
  }
