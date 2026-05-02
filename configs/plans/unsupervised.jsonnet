// Unsupervised archetype — VGAE + DGI fit/test.

local g = import '../index.libsonnet';

function(dataset, seed)
  local meta(variant, mt) = {
    group: 'unsupervised', variant: variant,
    dataset: dataset, seed: seed,
    model_type: mt, scale: 'small',
  };
  local data = g.data.datamodule.graph(
    source       = g.data.source.can_bus(dataset, seed),
    label_filter = 'benign',
  );

  local vgae = g.compose.unsupervised(
    model   = g.models.autoencoder.vgae(),
    data    = data,
    loss    = g.losses.vgae_task(),
    monitor = 'val_discrimination_ratio',
    meta    = meta('vgae', 'vgae'),
    trainer_overrides = { max_epochs: 600, precision: '32-true' },
  );
  local dgi = g.compose.unsupervised(
    model   = g.models.autoencoder.dgi(),
    data    = data,
    monitor = 'val_dgi_loss',
    meta    = meta('dgi', 'dgi'),
    trainer_overrides = { max_epochs: 400 },
  );

  [
    g.row.fit('vgae', vgae),  g.row.test('vgae', vgae),
    g.row.fit('dgi',  dgi),   g.row.test('dgi',  dgi),
  ]
