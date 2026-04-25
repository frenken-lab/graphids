// Ablation: supervised stage with curriculum sampler + RandomScorer.
// Tests attack-ratio ramp alone — difficulty ordering is random.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  curriculum_start_ratio=1.0, curriculum_end_ratio=10.0,
  curriculum_max_epochs=300, num_tiers=10,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=std.native('paths.run_dir')(dataset, 'gat_sampling', 'curriculum_random', seed),
    conv_type=conv_type, loss_fn=loss_fn,
    sampler='curriculum',
    curriculum_start_ratio=curriculum_start_ratio,
    curriculum_end_ratio=curriculum_end_ratio,
    curriculum_max_epochs=curriculum_max_epochs,
    num_tiers=num_tiers,
    curriculum_scorer={
      class_path: 'graphids.core.data.curriculum.RandomScorer',
      init_args: { seed: seed },
    },
    ckpt_path=ckpt_path,
  ), std.extVar('overrides'))
