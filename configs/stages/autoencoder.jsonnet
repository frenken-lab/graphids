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
  model_type='vgae',

  // KD — loss-level distillation config (null = no distillation)
  distillation_config=null,

  ckpt_path=null,
)

  defaults
  + unsup[model_type][scale]

  + {
    seed_everything: seed,

    callbacks+: {
      // val_loss is benign-only (label_filter='benign' below) so it can't
      // detect when discrimination emerges. val_discrimination_ratio =
      // val_loss_attack / (val_loss_benign + 1e-6) is logged per-epoch
      // from vgae_module.validation_step. We monitor the RATIO not the
      // GAP because the gap (attack - benign) decreases monotonically as
      // both losses converge toward zero — under mode='max', ModelCheckpoint
      // would save epoch 0's untrained weights. The ratio grows as the
      // model learns to separate the classes (smoke 687f3a07: 1.11 → 1.35
      // over 50 ep while gap shrank 0.204 → 0.087). Checkpoint and
      // EarlyStopping must track the same (monitor, mode) pair (enforced
      // by config/schemas.py:CallbacksSection) so the saved best-epoch
      // ckpt is the same epoch the stop trigger fires on.
      checkpoint+:      { init_args+: { monitor: 'val_discrimination_ratio', mode: 'max' } },
      early_stopping+: { init_args+: { monitor: 'val_discrimination_ratio', mode: 'max' } },
    },

    trainer+: {
      default_root_dir: run_dir,
      // 600-epoch ceiling. EarlyStopping monitors val_discrimination_ratio
      // (above), so the run typically stops well before this; the cap is
      // a hard upper bound, not a target. 1200 was the prior ceiling
      // chosen for cosine-LR runway; with constant LR + a real stop
      // signal the doubled budget bought nothing.
      max_epochs: 600,
      // 32-true: 16-mixed was tried (smoke 687f3a07, 50/50 epochs all-finite
      // with the logvar ±10 clamp and moment ±10 clamps in place) but gave
      // no throughput improvement (40.8 s/ep fp16 vs ~41 s/ep fp32). The
      // 8.5× regression vs the historical 4.8 s/ep run is NOT a precision
      // issue (suspect: validation_step running 3 forwards per batch,
      // separately tracked). The budget probe is calibrated for fp32 and
      // the fp16 smoke triggered an OOM absorbed by _oom_safe_step,
      // indicating the budget is over-aggressive under fp16. Stay on
      // 32-true until the throughput regression is solved.
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
      } + (if distillation_config != null
           then { distillation_config: distillation_config }
           else {}),
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})
