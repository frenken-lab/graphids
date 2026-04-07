// Supervised GAT classification stage.
//
// Merges the old normal.jsonnet and curriculum.jsonnet into one stage.
// The `sampler` TLA controls whether curriculum difficulty-ramping is
// active ('curriculum') or standard batching is used ('default').
//
// NOTE: gat.base sets `data.init_args.num_workers: 4`, which overrides
// the stage's default of null (auto-sized). GAT is compute-bound
// (cg_ratio ≈ 0.21), so extra workers idle waiting for the GPU.

local defaults = import '../_lib/defaults.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';
local sup = import '../models/supervised.libsonnet';
local pd = (import '../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset,
  seed=pd.seed,
  run_dir='',

  scale=pd.scale,
  conv_type=pd.conv_type,
  loss_fn=pd.loss_fn,
  model_type='gat',

  // Sampler toggle: 'default' or 'curriculum'
  sampler='default',

  // Curriculum-specific (ignored when sampler='default')
  curriculum_start_ratio=1.0,
  curriculum_end_ratio=10.0,
  canid_weight=0.1,
  curriculum_max_epochs=300,
  num_tiers=10,

  // Upstream checkpoint (VGAE teacher for curriculum scoring)
  vgae_ckpt_path=null,

  // KD — loss-level distillation config (null = no distillation)
  distillation_config=null,

  trainer_overrides={},
  stage_overrides={},
  ckpt_path=null,
)

  defaults
  + sup[model_type][scale]

  + {
    seed_everything: seed,

    trainer+: {
      default_root_dir: run_dir,
    },

    data+: {
      class_path: 'graphids.core.data.datamodule.GraphDataModule',
      init_args+: {
        window_size: 100,
        stride: 100,
        val_fraction: 0.2,
        batch_size: 8192,
        dynamic_batching: true,
        dataset: dataset,
        seed: seed,
        conv_type: conv_type,
        sampler: sampler,
      } + (if sampler == 'curriculum' then {
        curriculum_start_ratio: curriculum_start_ratio,
        curriculum_end_ratio: curriculum_end_ratio,
        canid_weight: canid_weight,
        max_epochs: curriculum_max_epochs,
        num_tiers: num_tiers,
      } else {})
        + (if vgae_ckpt_path != null
           then { vgae_ckpt_path: vgae_ckpt_path }
           else {}),
    },

    model+: {
      init_args+: {
        dataset: dataset,
        seed: seed,
        conv_type: conv_type,
        loss_fn: loss_fn,
      } + (if distillation_config != null
           then { distillation_config: distillation_config }
           else {}),
    },
  } + (if ckpt_path != null then { ckpt_path: ckpt_path } else {})

  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
