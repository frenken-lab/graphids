// Ablation: GAT supervised stage with RobustScaler (median + IQR) fit on
// benign-only rows. CAN benigns are heavy-tailed (entropy spikes during
// diagnostic broadcasts; bursts on power events) so median+IQR may dampen
// outlier influence on the input frame vs mean+std. Secondary recommendation
// in ~/plans/scaler-design-supervised-ood.md §5.
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
      run_dir=std.native('paths.run_dir')(dataset, 'scaler', 'robust_benign', seed),
      conv_type=conv_type, loss_fn=loss_fn,
      sampler=sampler,
      ckpt_path=ckpt_path,
    ) + {
      data+: { init_args+: { dataset+: { init_args+: {
        scaler_strategy: 'robust_benign',
      } } } },
    },
    std.extVar('overrides'),
  )
