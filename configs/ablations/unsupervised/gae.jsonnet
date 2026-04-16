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
    // See vgae.jsonnet — same LR + T_max bump rationale. GAE breaks through
    // slightly earlier than VGAE (no KL noise) but still plateaus ~40% of
    // training pre-breakthrough.
    trainer_overrides={ 'trainer.max_epochs': 1200 } + trainer_overrides,
    stage_overrides={ 'model.init_args.lr': 0.006 } + stage_overrides,
    ckpt_path=ckpt_path,
  )
