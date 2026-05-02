// DGI model primitive — contrastive-pretraining unsupervised architecture.
// OCGIN-style centroid scoring at test time (see DGIModule docstring).

local _scales = {
  small: { latent_dim: 48, embedding_dim: 32, heads: 4 },
  large: { latent_dim: 96, embedding_dim: 64, heads: 8 },
};

function(scale='small', conv_type='gatv2', dropout=0.15, lr=0.001)
  {
    model: {
      class_path: 'graphids.core.models.autoencoder.dgi_module.DGIModule',
      init_args: {
        conv_type: conv_type,
        edge_dim: 11,
        dropout: dropout,
        lr: lr,
        scale: scale,
      } + _scales[scale],
    },
  }
