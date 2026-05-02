// Supervised archetype — GAT + focal loss, single-preset spike.

local g = import '../index.libsonnet';

function(dataset, seed)
  local focal = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(source=g.data.source.can_bus(dataset, seed)),
    loss  = g.losses.focal(),
    meta  = {
      group: 'gat_loss', variant: 'focal',
      dataset: dataset, seed: seed,
      model_type: 'gat', scale: 'small',
    },
    trainer_overrides = { max_epochs: 200 },
  );

  [
    g.row.fit('focal', focal),
    g.row.test('focal', focal),
  ]
