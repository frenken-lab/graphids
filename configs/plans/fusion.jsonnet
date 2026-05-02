// Fusion archetype — 1 extract row + 4 methods (bandit/dqn/mlp/weighted_avg) fit/test.
// The extract row produces the cached fusion features; fit/test rows consume
// them via FusionDataModule. Submit via:
//
//   graphids run configs/plans/fusion.jsonnet --dataset hcrl_sa --seed 42 -o plan.json
//   EXTRACT_JID=$(jq -c '.[0]' plan.json | xargs -I {} graphids submit --row {} --cluster pitzer)
//   for r in $(jq -c '.[1:][]' plan.json); do
//     graphids submit --row "$r" --cluster pitzer --depends-on-afterok "$EXTRACT_JID"
//   done

local g = import '../index.libsonnet';

function(dataset, seed)
  local extract_dir = std.native('paths.states_dir')(dataset, seed);
  local vgae_ckpt = std.native('paths.best_ckpt')(dataset, 'unsupervised', 'vgae', seed);
  local gat_ckpt = std.native('paths.best_ckpt')(dataset, 'gat_loss', 'focal', seed);

  local meta(variant) = {
    group: 'fusion', variant: variant,
    dataset: dataset, seed: seed,
    model_type: 'fusion', scale: 'small',
  };
  local fuse(variant, model) = g.compose.fusion(
    model  = model,
    method = variant,
    meta   = meta(variant),
  );

  local bandit       = fuse('bandit',       g.models.fusion.bandit());
  local dqn          = fuse('dqn',          g.models.fusion.dqn());
  local mlp          = fuse('mlp',          g.models.fusion.mlp());
  local weighted_avg = fuse('weighted_avg', g.models.fusion.weighted_avg());

  [
    g.row.extract(
      name='extract_fusion',
      dataset=dataset,
      extractor_ckpts={ vgae: vgae_ckpt, gat: gat_ckpt },
      output_dir=extract_dir,
      seed=seed,
    ),
    g.row.fit('bandit',       bandit),       g.row.test('bandit',       bandit),
    g.row.fit('dqn',          dqn),          g.row.test('dqn',          dqn),
    g.row.fit('mlp',          mlp),          g.row.test('mlp',          mlp),
    g.row.fit('weighted_avg', weighted_avg), g.row.test('weighted_avg', weighted_avg),
  ]
