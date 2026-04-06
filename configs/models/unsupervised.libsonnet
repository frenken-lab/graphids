// Unsupervised model family — VGAE + DGI architectures.
//
// Each sub-key exposes {base, scales} for composition by stage jsonnets.
// The autoencoder stage indexes via model_type TLA: unsup[model_type].base.
//
// KD is handled at the loss level (core/losses/distillation.py), not here.

{
  vgae: {
    base: {
      model: {
        class_path: 'graphids.core.models.autoencoder.vgae.VGAEModule',
        init_args: {
          conv_type: 'gatv2',
          edge_dim: 11,
          variational: true,
          mask_ratio: 0.3,
          k_neg: 32,
          canid_weight: 0.1,
          nbr_weight: 0.05,
          kl_weight: 0.01,
          lr: 0.002,
          compile_model: true,
          gradient_checkpointing: true,
        },
      },
    },

    scales: {
      small: {
        model+: {
          init_args+: {
            scale: 'small',
            hidden_dims: [80, 40, 16],
            latent_dim: 16,
            heads: 1,
            embedding_dim: 4,
            dropout: 0.1,
            proj_dim: 32,
          },
        },
      },
      large: {
        model+: {
          init_args+: {
            scale: 'large',
            hidden_dims: [480, 240, 64],
            latent_dim: 64,
            heads: 4,
            embedding_dim: 32,
            dropout: 0.15,
            proj_dim: 48,
          },
        },
      },
    },
  },

  dgi: {
    base: {
      model: {
        class_path: 'graphids.core.models.autoencoder.dgi.DGIModule',
        init_args: {
          conv_type: 'gatv2',
          edge_dim: 11,
          compile_model: false,
          gradient_checkpointing: true,
        },
      },
    },

    scales: {
      small: {
        model+: {
          init_args+: {
            scale: 'small',
            hidden_dims: [80, 40, 16],
            latent_dim: 16,
            heads: 1,
            embedding_dim: 4,
            dropout: 0.1,
            proj_dim: 32,
          },
        },
      },
      large: {
        model+: {
          init_args+: {
            scale: 'large',
            hidden_dims: [480, 240, 64],
            latent_dim: 64,
            heads: 4,
            embedding_dim: 32,
            dropout: 0.15,
            proj_dim: 48,
          },
        },
      },
    },
  },
}
