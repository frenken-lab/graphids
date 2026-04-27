// Fusion stage — dispatch on fusion_method TLA over the 4 method overlays.
//
// Unlike the other stages, fusion overrides trainer defaults (cpu,
// precision 32, 1500 epochs, no gradient clip) via deep-merge from
// fusion.base, applied after defaults so the base block wins.

local defaults = import '../_lib/defaults.libsonnet';
local fusion = import '../models/fusion.libsonnet';
local pd = (import '../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset,
  seed=pd.seed,
  run_dir='',

  fusion_method=pd.fusion_method,
  scale=pd.scale,

  ckpt_path=null,
)

  defaults
  + {
    callbacks+: fusion.base.callbacks,
    trainer+: fusion.base.trainer,
    data: fusion.base.data,
  }
  + fusion.methods[fusion_method]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
    },

    data+: {
      init_args+: {
        // Fusion training reads VGAE+GAT state tensors written once by
        // `extract-fusion-states` per (dataset, seed) — shared across all
        // fusion methods, so the cache lives outside any single run_dir.
        cached_states_dir: std.native('paths.states_dir')(dataset, seed),
      },
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})
