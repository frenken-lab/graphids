// Ablation: supervised stage, locked loss_fn='weighted_ce'.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type,
  sampler='default',
  ckpt_path=null,
)
  std.mergePatch(
    stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=std.native('paths.run_dir')(dataset, 'gat_loss', 'weighted_ce', seed),
    conv_type=conv_type, sampler=sampler,
    loss_fn='weighted_ce',
    ckpt_path=ckpt_path,
  ), std.extVar('overrides'))
