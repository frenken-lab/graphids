// Ablation: autoencoder stage, locked model_type='vgae', variational=true.
//
// Bumps peak LR 0.002 → 0.006: post-standardization (cache v9.0.0) training
// plateaus at the ~2800 floor for ~50% of epochs before breaking through.
// Higher LR pushes through the stuck phase faster.
// (max_epochs=1200 + precision='32-true' moved into the autoencoder stage —
// they apply to every VGAE run, not just this preset.)
local stage = import '../../stages/autoencoder.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'unsupervised', 'vgae', seed),
      conv_type=conv_type,
      model_type='vgae', variational=true,
      ckpt_path=ckpt_path,
    ) + { model+: { init_args+: { lr: 0.006 } } },
    std.extVar('overrides'),
  )
