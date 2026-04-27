// Ablation: curriculum sampler with attack-ratio ramp 1.0 → 10.0 (i.e.
// start balanced, end imbalanced 10:1 normal:attack). Random within-tier
// ordering — direction is isolated from scorer choice. Pair with
// high_to_low.jsonnet for the direction A/B per
// ~/plans/curriculum-imbalance-direction-ablation.md.
//
// This is the current code default; declared explicitly so the
// curriculum_direction axis has both arms in the matrix.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  curriculum_max_epochs=300, num_tiers=10,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'curriculum_direction', 'low_to_high', seed),
      conv_type=conv_type, loss_fn=loss_fn,
      sampler='curriculum',
      curriculum_start_ratio=1.0,
      curriculum_end_ratio=10.0,
      curriculum_max_epochs=curriculum_max_epochs,
      num_tiers=num_tiers,
      curriculum_scorer={
        class_path: 'graphids.core.data.curriculum.RandomScorer',
        init_args: { seed: seed },
      },
      ckpt_path=ckpt_path,
    ),
    std.extVar('overrides'),
  )
