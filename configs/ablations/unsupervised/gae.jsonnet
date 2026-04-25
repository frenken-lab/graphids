// Ablation: autoencoder stage, locked model_type='vgae', variational=false (plain GAE).
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
    run_dir=std.native('paths.run_dir')(dataset, 'unsupervised', 'gae', seed),
    conv_type=conv_type,
    model_type='vgae', variational=false,
    // See vgae.jsonnet — same LR + T_max bump rationale. GAE breaks through
    // slightly earlier than VGAE (no KL noise) but still plateaus ~40% of
    // training pre-breakthrough.
    ckpt_path=ckpt_path,
    // precision='32-true' — see vgae.jsonnet for rationale (cache v10
    // z_benign post-standardization magnitudes vs fp16 range).
  ) + { trainer+: { max_epochs: 1200, precision: '32-true' }, model+: { init_args+: { lr: 0.006 } } }, std.extVar('overrides'))
