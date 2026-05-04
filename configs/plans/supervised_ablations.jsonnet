// Supervised archetype stress test — loss variation (focal/ce) + upstreams
// variation (curriculum_vgae reads VGAE ckpt by row order).

local g = import '../index.libsonnet';

function(dataset, seed)
  local source = g.data.source.can_bus(dataset, seed);
  local meta(group, variant) = {
    group: group, variant: variant,
    dataset: dataset, seed: seed,
    model_type: 'gat', scale: 'small',
  };

  local focal = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(source=source),
    loss  = g.losses.focal(),
    meta  = meta('gat_loss', 'focal'),
    trainer_overrides = { max_epochs: 200 },
  );
  local ce = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(source=source),
    loss  = g.losses.ce(),
    meta  = meta('gat_loss', 'ce'),
  );

  // curriculum_vgae — single-source the upstream ckpt path so the datamodule
  // difficulty wiring AND the row's `_upstreams` lineage can never drift.
  // Loss-end curriculum via CurriculumWeightedLoss reading per-graph
  // difficulty + in_scope attached by GraphDataModule.setup.
  local vgae_ckpt = std.native('paths.best_ckpt')(dataset, 'unsupervised', 'vgae', seed);
  local curriculum_vgae = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(
      source     = source,
      difficulty = {
        class_path: 'graphids.core.data.preprocessing.curriculum.score_vgae',
        init_args: { ckpt_path: vgae_ckpt },
      },
    ),
    loss = g.losses.curriculum(g.losses.focal()),
    meta = meta('gat_sampling', 'curriculum_vgae'),
    upstreams = [
      { role: 'vgae', ckpt_path: vgae_ckpt, ckpt_tla: 'vgae_ckpt_path' },
    ],
  );

  [
    g.row.fit('focal',           focal),
    g.row.fit('ce',              ce),
    g.row.fit('curriculum_vgae', curriculum_vgae),
  ]
