// Ablation: supervised stage, locked sampler='default'.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=std.native('paths.run_dir')(dataset, 'gat_sampling', 'none', seed),
    conv_type=conv_type, loss_fn=loss_fn,
    sampler='default',
    ckpt_path=ckpt_path,
  ), std.extVar('overrides'))
