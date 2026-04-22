// Ablation arm 1/3: lookup embedding with untrained UNK slot.
// Baseline — documents the silent-failure mode where unseen arb_ids
// map to a never-gradient-updated row. Uses the module defaults
// (id_encoder_class_path=LookupIdEncoder, p_unk_drop=0.0) so the
// override block is intentionally empty — this preset's value is
// the distinct run_dir, not a config difference. See
// ~/plans/oov-embedding-handling.md §Stage 3 ablation arm.
local stage = import '../../stages/supervised.jsonnet';
local paths = import '../_paths.libsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  lake_root='/fs/ess/PAS1266/graphids/dev/rf15',
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  sampler='default',
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=paths.run_dir(lake_root, dataset, 'id_encoding', 'lookup', seed),
    conv_type=conv_type, loss_fn=loss_fn, sampler=sampler,
    trainer_overrides=trainer_overrides,
    stage_overrides=stage_overrides,
    ckpt_path=ckpt_path,
  )
