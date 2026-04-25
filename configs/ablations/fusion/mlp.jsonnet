// Ablation: fusion stage, locked fusion_method='mlp'.
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
    run_dir=std.native('paths.run_dir')(dataset, 'fusion', 'mlp', seed),
    fusion_method='mlp',
    ckpt_path=ckpt_path,
  ) + {
    data+: { init_args+: {
      cached_states_dir: std.native('paths.states_dir')(dataset, seed),
    } },
  }, std.extVar('overrides'))
