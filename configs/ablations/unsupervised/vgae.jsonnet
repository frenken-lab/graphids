// Ablation: autoencoder stage, locked model_type='vgae'.
// lr override (0.002→0.006) dropped 2026-04-28 — workaround for pre-#43 plateau bug, no longer needed.
// `variational` TLA dropped 2026-04-28 — locked at always-on after the mask-recon synthesis.
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
      model_type='vgae',
      ckpt_path=ckpt_path,
    ),
    std.extVar('overrides'),
  )
