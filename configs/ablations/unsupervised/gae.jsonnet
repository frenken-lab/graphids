// Ablation: autoencoder stage, locked model_type='vgae', variational=false (plain GAE).
local stage = import '../../stages/autoencoder.jsonnet';
local paths = import '../_paths.libsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  lake_root='/fs/ess/PAS1266/graphids/dev/rf15',
  scale=pd.scale, conv_type=pd.conv_type,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=paths.run_dir(lake_root, dataset, 'unsupervised', 'gae', seed),
    conv_type=conv_type,
    model_type='vgae', variational=false,
    trainer_overrides=trainer_overrides,
    stage_overrides=stage_overrides,
    ckpt_path=ckpt_path,
  )
