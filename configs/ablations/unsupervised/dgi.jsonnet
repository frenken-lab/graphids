// Ablation: autoencoder stage, locked model_type='dgi' (Deep Graph Infomax).
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
    run_dir=std.native('paths.run_dir')(dataset, 'unsupervised', 'dgi', seed),
    conv_type=conv_type,
    model_type='dgi',
    // DGI contrastive task converges ~3x slower than VGAE reconstruction:
    // val_loss=1.37 (random) until epoch ~165, then descends to 0.75 still
    // dropping at epoch 299. CosineAnnealingLR is near-zero by then. Bump
    // T_max so peak LR persists through the breakthrough phase.
    ckpt_path=ckpt_path,
    // precision='32-true' — see vgae.jsonnet for rationale (cache v10
    // z_benign post-standardization magnitudes vs fp16 range).
  ) + { trainer+: { max_epochs: 800, precision: '32-true' } }, std.extVar('overrides'))
