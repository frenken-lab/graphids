// Ablation: fusion stage, locked fusion_method='bandit'.
local stage = import '../../stages/fusion.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'fusion', 'bandit', seed),
      fusion_method='bandit',
      ckpt_path=ckpt_path,
    ),
    std.extVar('overrides'),
  )
