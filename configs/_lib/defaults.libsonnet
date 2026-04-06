// Shared trainer, checkpoint, and early-stopping defaults.
//
// Baked into every stage via `import '../_lib/defaults.libsonnet'` so there
// is no silent `parser_kwargs.default_config_files` injection.
//
// Ported verbatim from graphids/config/defaults/trainer.yaml
// (deleted during Phase 1).

{
  trainer: {
    trainer: {
      accelerator: 'auto',
      devices: 'auto',
      precision: '16-mixed',
      max_epochs: 300,
      gradient_clip_val: 1.0,
      log_every_n_steps: 50,
    },
  },

  checkpoint: {
    checkpoint: {
      monitor: 'val_loss',
      mode: 'min',
      save_top_k: 1,
      save_last: true,
      filename: 'best_model',
    },
  },

  early_stopping: {
    early_stopping: {
      monitor: 'val_loss',
      mode: 'min',
      patience: 100,
    },
  },
}
