// Ablation: supervised stage, locked conv_type='gat'.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, loss_fn=pd.loss_fn,
  sampler='default',
  ckpt_path=null,
)
  std.mergePatch(
    stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=std.native('paths.run_dir')(dataset, 'conv_type', 'gat', seed),
    conv_type='gat', loss_fn=loss_fn, sampler=sampler,
    ckpt_path=ckpt_path,
  ), std.extVar('overrides'))
