// Ablation: fusion stage, locked fusion_method='bandit'.
// cached_states_dir (shared across all 4 fusion methods for a seed) is
// auto-derived from (dataset, seed) — produced by extract-fusion-states.
local stage = import '../../stages/fusion.jsonnet';
local paths = import '../_paths.libsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  lake_root='/fs/ess/PAS1266/graphids/dev/rf15',
  scale=pd.scale,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=paths.run_dir(lake_root, dataset, 'fusion', 'bandit', seed),
    fusion_method='bandit',
    trainer_overrides=trainer_overrides,
    stage_overrides=stage_overrides,
    ckpt_path=ckpt_path,
  ) + {
    data+: { init_args+: {
      cached_states_dir: paths.states_dir(lake_root, dataset, seed),
    } },
  }
