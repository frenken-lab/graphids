// Unsupervised archetype composer (VGAE / DGI / future reconstruction stages).
// All unsupervised presets share this composition shape; only model,
// monitor metric, and trainer overrides vary per architecture.

local trainer   = import '../_kit/trainer.libsonnet';
local callbacks = import '../_kit/callbacks.libsonnet';
local v         = import '../_kit/validate.libsonnet';

// `loss` is optional — VGAE supplies it via configs/losses/vgae_task.libsonnet
// (merged into model.init_args, same pattern as supervised composer). DGI
// constructs its contrastive loss internally and passes `loss={}` (no-op merge).
function(model, data, monitor, meta,
         loss={},
         trainer_overrides={},
         upstreams=[],
         patience=100,
         callback_extras={})

  v.spec(
    trainer
    + model {
      model+: { init_args+: loss },
    }
    + data
    + callbacks(monitor=monitor, mode='max', patience=patience, extras=callback_extras)
    + {
      seed_everything: meta.seed,
      trainer+: {
        default_root_dir: std.native('paths.run_dir')(
          meta.dataset, meta.group, meta.variant, meta.seed
        ),
      } + trainer_overrides,
      _meta: meta,
      // Archetype-fixed: unsupervised reconstruction needs GPU.
      // Length is row-level (smoke vs production), set by the plan.
      _resources: {mode: 'gpu'},
      _upstreams: upstreams,
    }
  )
