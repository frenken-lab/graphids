// Shared defaults — callbacks, trainer, and monitoring.
//
// `callbacks` is a named object so stages can deep-merge individual
// entries (e.g. fusion overrides checkpoint monitor). The `trainer`
// block derives `trainer.callbacks` from `$.callbacks` via late-binding
// so overrides are picked up automatically.

{
  callbacks: {
    checkpoint: {
      class_path: 'graphids.core.callbacks.ModelCheckpoint',
      init_args: {
        monitor: 'val_loss',
        mode: 'min',
        save_top_k: 1,
        save_last: true,
        filename: 'best_model',
      },
    },
    early_stopping: {
      class_path: 'graphids.core.callbacks.EarlyStopping',
      init_args: {
        monitor: 'val_loss',
        mode: 'min',
        patience: 100,
      },
    },
    curriculum: {
      class_path: 'graphids.core.data.curriculum.CurriculumEpochCallback',
      init_args: {},
    },
    otel: {
      class_path: 'graphids.core.monitoring.OTelTrainingCallback',
      init_args: {},
    },
  },

  trainer: {
    accelerator: 'auto',
    devices: 'auto',
    precision: '16-mixed',
    max_epochs: 300,
    gradient_clip_val: 1.0,
    log_every_n_steps: 50,
    callbacks: [$.callbacks[k] for k in std.objectFields($.callbacks)],
    logger: [
      {
        class_path: 'graphids.core.monitoring.OTelTrainingLogger',
        init_args: {},
      },
    ],
  },
}
