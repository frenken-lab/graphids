// Fusion archetype — all 4 methods (bandit/dqn/mlp/weighted_avg) fit/test.

local g = import '../index.libsonnet';

function(dataset, seed)
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
    g.row.fit('bandit',       bandit),       g.row.test('bandit',       bandit),
    g.row.fit('dqn',          dqn),          g.row.test('dqn',          dqn),
    g.row.fit('mlp',          mlp),          g.row.test('mlp',          mlp),
    g.row.fit('weighted_avg', weighted_avg), g.row.test('weighted_avg', weighted_avg),
  ]
