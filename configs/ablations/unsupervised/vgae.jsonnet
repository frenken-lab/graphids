// Ablation: autoencoder stage, locked model_type='vgae', variational=true.
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
    // Post-standardization (cache v9.0.0) training still plateaus at the
    // old ~2800 floor for ~50% of epochs before breaking through to unit
    // scale. Bump peak LR 0.002 → 0.006 and T_max 300 → 600 so the
    // optimizer has more push during the stuck phase and more runway
    // after the breakthrough.
    ckpt_path=ckpt_path,
    // precision='32-true' (overrides default '16-mixed'): cache v10's
    // z_benign scaler divides by benign-only stddev (smaller than the
    // benign+attack stddev v9 used), so post-standardization attack-row
    // magnitudes are larger than v9 saw. fp16 overflow risk that was
    // previously hcrl_sa-only may now extend to other datasets. Override
    // back to 16-mixed via --set trainer.precision='16-mixed' once
    // validated for a given (dataset, scaler_strategy) pair.
  ) + { trainer+: { max_epochs: 1200, precision: '32-true' }, model+: { init_args+: { lr: 0.006 } } }, std.extVar('overrides'))
