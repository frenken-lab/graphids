// Unsupervised model family — VGAE + DGI architectures.
//
// Each sub-key exposes pre-merged configs indexed by scale:
//   unsup[model_type][scale]  →  complete model block ready for `+` merge.
//
// KD is handled at the loss level (core/losses/distillation.py), not here.

local _vgae_base = {
  model: {
    class_path: 'graphids.core.models.autoencoder.vgae_module.VGAEModule',
    init_args: {
      conv_type: 'gatv2',
      edge_dim: 11,
      // mask-recon training: 15% random node masking per batch.
      mask_rate: 0.15,
      // routed to VGAETaskLoss via inject_loss_fn (loss-builder owns the kw).
      kl_weight: 0.01,
      lr: 0.002,
      compile_model: false,
      gradient_checkpointing: true,
    },
  },
};

local _dgi_base = {
  model: {
    class_path: 'graphids.core.models.autoencoder.dgi_module.DGIModule',
    init_args: {
      conv_type: 'gatv2',
      edge_dim: 11,
      compile_model: false,
      gradient_checkpointing: true,
    },
  },
};

{
  vgae: {
    small: _vgae_base + {
      model+: { init_args+: {
        scale: 'small',
        hidden_dims: [80, 40, 16],
        latent_dim: 16,
        heads: 1,
        embedding_dim: 4,
        dropout: 0.1,
        proj_dim: 32,
      } },
    },
    large: _vgae_base + {
      model+: { init_args+: {
        scale: 'large',
        hidden_dims: [480, 240, 64],
        latent_dim: 64,
        heads: 4,
        embedding_dim: 32,
        // 0.1 (was 0.15) — under 15% input masking + dropout, compound noise
        // stalled training; encoder dropout drops to match small variant.
        dropout: 0.1,
        proj_dim: 48,
      } },
    },
  },

  dgi: {
    small: _dgi_base + {
      model+: { init_args+: {
        scale: 'small',
        hidden_dims: [80, 40, 16],
        latent_dim: 16,
        heads: 1,
        embedding_dim: 4,
        dropout: 0.1,
        proj_dim: 32,
      } },
    },
    large: _dgi_base + {
      model+: { init_args+: {
        scale: 'large',
        hidden_dims: [480, 240, 64],
        latent_dim: 64,
        heads: 4,
        embedding_dim: 32,
        dropout: 0.15,
        proj_dim: 48,
      } },
    },
  },
}
