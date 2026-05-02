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
      // NOTE: ``init_args`` is empty here, but the callback's __init__
      // requires ``run_id``. ``orchestrate._build`` injects it by
      // string-matching this dict-key (``mlflow``) and instantiating
      // ``graphids._mlflow.MLflowTrainingCallback(run_id=run_id)`` directly,
      // bypassing the ``class_path`` below. The ``class_path`` is therefore
      // documentation, not a live constructor target. Renaming the dict-key
      // breaks every fit; renaming the class breaks nothing here. (Tier 2
      // cleanup: drop the special-case in orchestrate so this becomes a
      // regular instantiation.)
      mlflow: {
        class_path: 'graphids._mlflow.MLflowTrainingCallback',
        init_args: {},
      },
    } + extras,
  }
