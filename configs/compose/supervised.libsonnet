// Supervised archetype composer — GAT classification (CE / focal /
// weighted_ce / tau_norm losses, with sampler ablations).
// `loss` is a fragment {loss: {...}} merged into model.init_args at compose.

local trainer   = import '../_kit/trainer.libsonnet';
local callbacks = import '../_kit/callbacks.libsonnet';
local v         = import '../_kit/validate.libsonnet';

function(model, data, loss, meta,
         monitor='val_auroc', mode='max',
         trainer_overrides={},
         upstreams=[],
         patience=50,
         callback_extras={})

  local run_dir = std.native('paths.run_dir')(
    meta.dataset, meta.group, meta.variant, meta.seed
  );

  v.spec(
    trainer
    + model {
      // Merge the loss fragment ({loss_fn: {class_path, init_args}}) into
      // model.init_args. orchestrate._instantiate recursively builds the
      // class_path block so the model receives a real nn.Module.
      model+: { init_args+: loss },
    }
    + data
    + callbacks(monitor=monitor, mode=mode, patience=patience,
                run_dir=run_dir, extras=callback_extras)
    + {
      seed_everything: meta.seed,
      trainer+: {
        default_root_dir: run_dir,
      } + trainer_overrides,
      _meta: meta,
      // Archetype-fixed: supervised GAT needs GPU. Length is row-level.
      _resources: {mode: 'gpu'},
      _upstreams: upstreams,
    }
  )
