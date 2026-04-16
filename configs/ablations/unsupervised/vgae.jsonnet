// Ablation: autoencoder stage, locked model_type='vgae', variational=true.
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
    run_dir=paths.run_dir(lake_root, dataset, 'unsupervised', 'vgae', seed),
    conv_type=conv_type,
    model_type='vgae', variational=true,
    // Post-standardization (cache v9.0.0) training still plateaus at the
    // old ~2800 floor for ~50% of epochs before breaking through to unit
    // scale. Bump peak LR 0.002 → 0.006 and T_max 300 → 600 so the
    // optimizer has more push during the stuck phase and more runway
    // after the breakthrough.
    trainer_overrides={ 'trainer.max_epochs': 1200 } + trainer_overrides,
    stage_overrides={ 'model.init_args.lr': 0.006 } + stage_overrides,
    ckpt_path=ckpt_path,
  )
