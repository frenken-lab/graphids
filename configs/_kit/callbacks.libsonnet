// Callbacks primitive — checkpoint + early_stopping, parameterized by monitor.
// monitor and mode flow into BOTH callbacks (kept in lockstep — checkpoint
// and early_stop must track the same metric, see CallbacksSection schema).

// `extras` is the merge knob for optional callbacks (e.g.
// CurriculumEpochCallback). Universal trio (checkpoint, early_stopping,
// mlflow) is mandatory and not parameterized at this level.
function(monitor='val_auroc', mode='max', patience=100, extras={})
  {
    callbacks: {
      checkpoint: {
        class_path: 'graphids.core.callbacks.ModelCheckpoint',
        init_args: {
          monitor: monitor,
          mode: mode,
          save_top_k: 1,
          save_last: true,
          filename: 'best_model',
        },
      },
      early_stopping: {
        class_path: 'graphids.core.callbacks.EarlyStopping',
        init_args: {
          monitor: monitor,
          mode: mode,
          patience: patience,
        },
      },
      // MLflow per-epoch metric forwarding — universal, no parameterization.
      // Owns nothing the trainer can know on its own; without this, MLflow
      // rows have no per-epoch metrics. See data-layout.md.
      //
      // ``init_args`` is empty: the callback's __init__ reads its ``run_id``
      // from ``$GRAPHIDS_MLFLOW_RUN_ID``, set by ``orchestrate.train`` /
      // ``.evaluate`` immediately after ``start_training_run``. Authoritative
      // construction goes through ``class_path`` like every other callback.
      mlflow: {
        class_path: 'graphids._mlflow.MLflowTrainingCallback',
        init_args: {},
      },
    } + extras,
  }
