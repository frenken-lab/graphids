// Ablation: GAT supervised stage with z-score scaler fit on benign-only rows.
// Default behavior post-#43 fix; explicit arm so the scaler axis is testable
// against robust_benign (median+IQR over benigns). The previous z_joint arm
// (scaler fit on benign+attack mixture) was removed — joint scaling bakes
// the training-attack distribution into the coordinate system and attenuates
// novel-attack discrimination. See ~/plans/scaler-design-supervised-ood.md.
//
// Cache rebuild trigger: scaler_strategy is part of CANBusSource.cache_key
// (sc:{strategy}), so each arm builds + persists its own cache + scaler
// estimator under {LAKE_ROOT}/cache/{...}/canbus|.../sc:z_benign/.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  sampler='default',
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'scaler', 'z_benign', seed),
      conv_type=conv_type, loss_fn=loss_fn,
      sampler=sampler,
      ckpt_path=ckpt_path,
    ) + {
      data+: { init_args+: { dataset+: { init_args+: {
        scaler_strategy: 'z_benign',
      } } } },
    },
    std.extVar('overrides'),
  )
