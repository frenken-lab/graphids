// Autoencoder stage — VGAE/DGI unsupervised pretraining.
//
// The `model_type` TLA selects the architecture within the unsupervised
// family ('vgae' or 'dgi'). Every TLA has a sensible default so
// `python -m graphids fit --config configs/stages/autoencoder.jsonnet`
// works with zero TLAs (dev smoke).

local defaults = import '../_lib/defaults.libsonnet';
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

  ckpt_path=null,
)

  defaults
  + unsup[model_type][scale]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
      // VGAE training plateaus for ~50% of epochs at the standardization
      // floor before breakthrough — bump max_epochs so EarlyStopping
      // (patience 100) has runway after the breakthrough phase.
      max_epochs: 1200,
      // cache v10 z_benign scaler divides by benign-only stddev (smaller
      // than benign+attack used in v9), so post-standardization attack-row
      // magnitudes can exceed fp16's max (~65504). 32-true is the safe
      // default; override back via --set trainer.precision='16-mixed'
      // once a (dataset, scaler_strategy) pair is empirically validated.
      precision: '32-true',
    },

    data: {
      class_path: 'graphids.core.data.datamodule.GraphDataModule',
      init_args: {
        dataset: {
          class_path: 'graphids.core.data.datasets.can_bus.CANBusSource',
          init_args: {
            name: dataset,
            seed: seed,
            window_size: 100,
            stride: 100,
            val_fraction: 0.2,
          },
        },
        num_workers: null,  // auto-sized from GPU-first sizing chain
        dynamic_batching: true,  // batch_size is unused on this path — sampler uses probe budget
        // Reconstruction stages train on benign traffic only — attack rows
        // would teach the decoder to reproduce anomalies. Supervised stages
        // omit this field (defaults to null = full train set).
        label_filter: 'benign',
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
