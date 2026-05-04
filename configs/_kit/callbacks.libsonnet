// Callbacks primitive — checkpoint + early_stopping, parameterized by monitor.
// monitor and mode flow into BOTH callbacks (kept in lockstep — checkpoint
// and early_stop must track the same metric, see CallbacksSection schema).

// `extras` is the merge knob for optional callbacks. Universal trio
// (checkpoint, early_stopping, mlflow) is mandatory and not parameterized
// at this level.
//
// `run_dir` is the apex's ``trainer.default_root_dir`` — passed through so
// the checkpoint sidecar lands at ``{run_dir}/checkpoints/`` (matches the
// path layout in data-layout.md). Lightning otherwise writes under
// ``default_root_dir/lightning_logs/version_N/checkpoints``, which the rest
// of graphids (resume, KD teacher loading) doesn't read.
function(monitor='val_auroc', mode='max', patience=100, run_dir='', extras={})
  {
    callbacks: {
      // graphids.core.callbacks.Sha256ModelCheckpoint = Lightning's
      // ModelCheckpoint + a `<ckpt>.sha256` sidecar for atomic_load
      // integrity verification on GPFS.
      checkpoint: {
        class_path: 'graphids.core.callbacks.Sha256ModelCheckpoint',
        init_args: {
          monitor: monitor,
          mode: mode,
          save_top_k: 1,
          save_last: true,
          dirpath: run_dir + '/checkpoints',
          filename: 'best_model',
        },
      },
      early_stopping: {
        class_path: 'lightning.pytorch.callbacks.EarlyStopping',
        init_args: {
          monitor: monitor,
          mode: mode,
          patience: patience,
        },
      },
      // MLflow per-epoch metric forwarding — universal, no parameterization.
      // Catalog ``LoggedModel`` lifecycle + graphids-specific run state.
      // Per-epoch metric forwarding is owned by Lightning's MLFlowLogger
      // (wired in ``orchestrate._make_trainer``). The callback pulls run_id
      // + client from ``trainer.logger`` in ``on_train_start``, so
      // ``init_args`` is genuinely empty.
      mlflow: {
        class_path: 'graphids._mlflow.MLflowTrainingCallback',
        init_args: {},
      },
    } + extras,
  }
