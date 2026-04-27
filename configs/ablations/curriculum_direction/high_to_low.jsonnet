// Ablation: curriculum sampler with attack-ratio ramp 10.0 → 1.0 (i.e.
// start imbalanced 10:1 normal:attack, end balanced). Random within-tier
// ordering. Direction matches DCL (Wang et al. ICCV 2019, Table 4) and
// the Minority Initial Drop mechanism (Francazi et al. ICML 2023) which
// both argue for imbalanced→balanced over balanced→imbalanced for
// extreme-imbalance binary classification. See
// ~/plans/curriculum-imbalance-direction-ablation.md for the full case.
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
      run_dir=std.native('paths.run_dir')(dataset, 'curriculum_direction', 'high_to_low', seed),
      conv_type=conv_type, loss_fn=loss_fn,
      sampler='curriculum',
      curriculum_start_ratio=10.0,
      curriculum_end_ratio=1.0,
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
