// Ablation: GAT trained with focal loss + Kang ICLR 2020 τ-norm applied
// to the classifier head at fit-end. Tests whether the GAT's final logit
// layer is imbalance-bottlenecked even when downstream fusion consumes a
// hybrid representation+logit input. See ~/plans/curriculum-imbalance-direction-ablation.md
// §"K1 — τ-norm on GAT's classifier head".
//
// τ defaults to 0.5 (Kang's typical sweet spot on long-tailed image bench).
// Sweep with --tla 'tau=0.0' (identity / no-op control) or 'tau=1.0' (full
// unit-norm) for direction confirmation.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type,
  sampler='default',
  tau=0.5,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'gat_loss', 'tau_norm', seed),
      conv_type=conv_type, sampler=sampler,
      loss_fn='focal',
      ckpt_path=ckpt_path,
    ) + {
      // Insert τ-norm callback under a key that sorts BEFORE 'mlflow'
      // so its on_fit_end runs first — MLflow then SHA256-tags the
      // τ-normed ckpt. defaults.libsonnet builds trainer.callbacks via
      // `[$.callbacks[k] for k in std.objectFields($.callbacks)]`,
      // which uses lexicographic order. 'kang_tau_norm' < 'mlflow'.
      callbacks+: {
        kang_tau_norm: {
          class_path: 'graphids.core.callbacks.TauNormCallback',
          init_args: { tau: tau },
        },
      },
    },
    std.extVar('overrides'),
  )
