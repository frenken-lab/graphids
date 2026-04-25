// Ablation arm 1/3: lookup embedding with untrained UNK slot.
// Baseline — documents the silent-failure mode where unseen arb_ids
// map to a never-gradient-updated row. Uses the module defaults
// (id_encoder_class_path=LookupIdEncoder, p_unk_drop=0.0) so the
// override block is intentionally empty — this preset's value is
// the distinct run_dir, not a config difference. See
// ~/plans/oov-embedding-handling.md §Stage 3 ablation arm.
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
    run_dir=std.native('paths.run_dir')(dataset, 'id_encoding', 'lookup', seed),
    conv_type=conv_type, loss_fn=loss_fn, sampler=sampler,
    ckpt_path=ckpt_path,
  ), std.extVar('overrides'))
