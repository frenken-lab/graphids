// Autoencoder stage — VGAE/DGI unsupervised pretraining.
//
// The `model_type` TLA selects the architecture within the unsupervised
// family ('vgae' or 'dgi'). Every TLA has a sensible default so
// `python -m graphids fit --config configs/stages/autoencoder.jsonnet`
// works with zero TLAs (dev smoke).

local defaults = import '../_lib/defaults.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';
local unsup = import '../models/unsupervised.libsonnet';

function(
  dataset='hcrl_ch',
  seed=42,
  run_dir='',

  scale='small',
  conv_type='gatv2',
  variational=true,
  model_type='vgae',

  // KD — loss-level distillation config (null = no distillation)
  distillation_config=null,

  trainer_overrides={},
  stage_overrides={},
  ckpt_path=null,
)

  defaults.trainer
  + defaults.checkpoint
  + defaults.early_stopping

  + unsup[model_type].base
  + unsup[model_type].scales[scale]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
    },

    data: {
      class_path: 'graphids.core.data.datamodule.CANBusDataModule',
      init_args: {
        window_size: 100,
        stride: 100,
        val_fraction: 0.2,
        batch_size: 8192,
        num_workers: null,  // auto-sized from GPU-first sizing chain
        dynamic_batching: true,
        dataset: dataset,
      },
    },

    model+: {
      init_args+: {
        conv_type: conv_type,
        variational: variational,
      } + (if distillation_config != null
           then { distillation_config: distillation_config }
           else {}),
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})

  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
