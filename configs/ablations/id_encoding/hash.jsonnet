// Ablation arm 3/3: k-probe hash embedding — primary evidence-backed
// treatment from the 2021-2026 industrial-recsys literature.
// Every arb_id (seen or unseen) deterministically maps to k buckets;
// attack-injected novel IDs hit trained buckets by construction, so
// no special-case OOV slot is needed. Shape follows Coleman et al.
// 2023 [NeurIPS Spotlight] and Yan et al. 2021 [CIKM]. k=2 and bucket
// count next_pow2(num_buckets_factor * num_ids) = 4x vocab is a
// standard sweet spot between collision rate and parameter count.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  sampler='default',
  k=2, num_buckets_factor=4,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'id_encoding', 'hash', seed),
      conv_type=conv_type, loss_fn=loss_fn, sampler=sampler,
      ckpt_path=ckpt_path,
    ) + {
      model+: { init_args+: {
        id_encoder_class_path: 'graphids.core.models.id_encoding.HashIdEncoder',
        id_encoder_kwargs: { k: k, seed: seed, num_buckets_factor: num_buckets_factor },
      } },
    },
    std.extVar('overrides'),
  )
