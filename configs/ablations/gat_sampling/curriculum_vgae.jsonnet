// Ablation: supervised stage with curriculum sampler + VGAEScorer.
// Upstream VGAE ckpt path is auto-derived from (dataset, seed) —
// matches the convention of unsupervised/vgae.jsonnet.
local stage = import '../../stages/supervised.jsonnet';
local paths = import '../_paths.libsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  lake_root='/fs/ess/PAS1266/graphids/dev/rf15',
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  canid_weight=0.1,
  curriculum_start_ratio=1.0, curriculum_end_ratio=10.0,
  curriculum_max_epochs=300, num_tiers=10,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=paths.run_dir(lake_root, dataset, 'gat_sampling', 'curriculum_vgae', seed),
    conv_type=conv_type, loss_fn=loss_fn,
    sampler='curriculum',
    vgae_ckpt_path=paths.vgae_ckpt(lake_root, dataset, seed),
    canid_weight=canid_weight,
    curriculum_start_ratio=curriculum_start_ratio,
    curriculum_end_ratio=curriculum_end_ratio,
    curriculum_max_epochs=curriculum_max_epochs,
    num_tiers=num_tiers,
    trainer_overrides=trainer_overrides,
    stage_overrides=stage_overrides,
    ckpt_path=ckpt_path,
  )
