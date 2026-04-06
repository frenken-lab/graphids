// Fusion stage — dispatch on fusion_method TLA over the 4 method overlays.
//
// Unlike the other stages, fusion overrides every trainer default (cpu,
// precision 32, 1500 epochs, no gradient clip) — those live in
// fusion.base, applied after defaults.trainer so the base block wins.

local defaults = import '../_lib/defaults.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';
local fusion = import '../fusion.libsonnet';

function(
  dataset='hcrl_ch',
  seed=42,
  run_dir='',

  fusion_method='bandit',
  scale='small',

  // Upstream teacher checkpoints.
  gat_ckpt_path=null,
  vgae_ckpt_path=null,

  trainer_overrides={},
  stage_overrides={},
  ckpt_path=null,
)

  defaults.trainer
  + defaults.checkpoint
  + defaults.early_stopping

  // Fusion base fully replaces trainer/checkpoint/early_stopping/data.
  + { trainer+: fusion.base.trainer }
  + { checkpoint+: fusion.base.checkpoint }
  + { early_stopping+: fusion.base.early_stopping }
  + { data: fusion.base.data }

  // Method-specific model class + init_args
  + fusion.methods[fusion_method]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
    },

    data+: {
      init_args+: {
        dataset: dataset,
      } + (if gat_ckpt_path != null
           then { gat_ckpt_path: gat_ckpt_path }
           else {}),
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})

  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
