// Autoencoder stage — VGAE/DGI unsupervised pretraining.
//
// The `model_type` TLA selects the architecture within the unsupervised
// family ('vgae' or 'dgi'). Every TLA has a sensible default so
// `python -m graphids fit --config configs/stages/autoencoder.jsonnet`
// works with zero TLAs (dev smoke).

local defaults = import '../_lib/defaults.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';
local unsup = import '../models/unsupervised.libsonnet';
local pd = (import '../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset,
  seed=pd.seed,
  run_dir='',

  scale=pd.scale,
  conv_type=pd.conv_type,
  variational=pd.variational,
  model_type='vgae',

  // KD — loss-level distillation config (null = no distillation)
  distillation_config=null,

  trainer_overrides={},
  stage_overrides={},
  ckpt_path=null,
)

  defaults
  + unsup[model_type][scale]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
    },

    data: {
      class_path: 'graphids.core.data.datamodule.GraphDataModule',
      init_args: {
        window_size: 100,
        stride: 100,
        val_fraction: 0.2,
        batch_size: 8192,
        num_workers: null,  // auto-sized from GPU-first sizing chain
        dynamic_batching: true,
        dataset: dataset,
        seed: seed,
        conv_type: conv_type,
        heads: $.model.init_args.heads,  // late-bind from model libsonnet
      },
    },

    model+: {
      init_args+: {
        dataset: dataset,
        seed: seed,
        conv_type: conv_type,
        variational: variational,
      } + (if distillation_config != null
           then { distillation_config: distillation_config }
           else {}),
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})

  + helpers.otel_identity('autoencoder', dataset, scale, seed, model_type)
  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
