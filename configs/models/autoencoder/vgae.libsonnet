// VGAE model primitive — class_path + scale-tunable init_args.
// `kl_weight` etc. belong to the VGAE *loss* and are deferred to the
// VGAE-loss class_path lift (separate refactor); the loss falls back to
// `VGAETaskLoss` defaults for now.

local _scales = {
  small: { latent_dim: 64,  hidden_dims: [64]  },
  large: { latent_dim: 128, hidden_dims: [128] },
};

function(scale='small', conv_type='gatv2', mask_rate=0.15, lr=0.002)
  {
    model: {
      class_path: 'graphids.core.models.autoencoder.vgae.VGAE',
      init_args: {
        conv_type: conv_type,
        edge_dim: 11,
        mask_rate: mask_rate,
        lr: lr,
      } + _scales[scale],
    },
  }
