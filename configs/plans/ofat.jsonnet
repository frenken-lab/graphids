// OFAT (one-factor-at-a-time) — sweeps every supervised ablation axis once,
// plus the unsupervised baseline (vgae). curriculum_vgae sits in the
// gat_sampling axis but reads the vgae upstream ckpt by row order.

local g = import '../index.libsonnet';

function(dataset, seed)
  // ----- helpers ---------------------------------------------------------
  local source(o={}) = g.data.source.can_bus(dataset, seed, overrides=o);
  local gat_meta(group, variant) = {
    group: group, variant: variant,
    dataset: dataset, seed: seed,
    model_type: 'gat', scale: 'small',
  };
  // Default GAT spec — model + plain graph datamodule + named loss.
  // Per-variant escape hatches: model_extra, data_overrides, source_overrides,
  // upstreams, callback_extras.
  local gat_spec(group, variant, loss,
                 model_extra={}, data_overrides={},
                 source_overrides={},
                 upstreams=[], callback_extras={},
                 trainer_overrides={}) =
    g.compose.supervised(
      model = g.models.supervised.gat() + model_extra,
      data  = g.data.datamodule.graph(
        source    = source(source_overrides),
        overrides = data_overrides,
      ),
      loss              = loss,
      meta              = gat_meta(group, variant),
      upstreams         = upstreams,
      callback_extras   = callback_extras,
      trainer_overrides = trainer_overrides,
    );

  // ----- unsupervised baseline (produces vgae ckpt) ----------------------
  local vgae = g.compose.unsupervised(
    model   = g.models.autoencoder.vgae(),
    data    = g.data.datamodule.graph(source=source(), label_filter='benign'),
    loss    = g.losses.vgae_task(),
    monitor = 'val_discrimination_ratio',
    meta    = {
      group: 'unsupervised', variant: 'vgae',
      dataset: dataset, seed: seed,
      model_type: 'vgae', scale: 'small',
    },
    trainer_overrides = { max_epochs: 600, precision: '32-true' },
  );

  // ----- gat_loss axis ---------------------------------------------------
  local focal       = gat_spec('gat_loss', 'focal',       g.losses.focal(),
                               trainer_overrides={ max_epochs: 200 });
  local ce          = gat_spec('gat_loss', 'ce',          g.losses.ce());
  local weighted_ce = gat_spec('gat_loss', 'weighted_ce', g.losses.weighted_ce(weights=[1.0, 5.0]));

  // ----- gat_sampling axis ----------------------------------------------
  local none = gat_spec('gat_sampling', 'none', g.losses.focal());

  // Loss-end curriculum: the datamodule attaches per-graph difficulty +
  // in_scope at setup; CurriculumWeightedLoss masks per-example focal
  // contribution at each training step via the LinearRampSchedule.
  local curriculum_random = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(
      source     = source(),
      difficulty = {
        class_path: 'graphids.core.data.preprocessing.curriculum.score_random',
        init_args: { seed: seed },
      },
    ),
    loss = g.losses.curriculum(g.losses.focal()),
    meta = gat_meta('gat_sampling', 'curriculum_random'),
  );

  // curriculum_vgae — single-source vgae ckpt path, threaded into BOTH the
  // datamodule difficulty wiring AND the row's `_upstreams` lineage.
  local vgae_ckpt = std.native('paths.best_ckpt')(dataset, 'unsupervised', 'vgae', seed);
  local curriculum_vgae = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(
      source     = source(),
      difficulty = {
        class_path: 'graphids.core.data.preprocessing.curriculum.score_vgae',
        init_args: { ckpt_path: vgae_ckpt },
      },
    ),
    loss = g.losses.curriculum(g.losses.focal()),
    meta = gat_meta('gat_sampling', 'curriculum_vgae'),
    upstreams = [
      { role: 'vgae', ckpt_path: vgae_ckpt, ckpt_tla: 'vgae_ckpt_path' },
    ],
  );

  // ----- scaler axis ----------------------------------------------------
  local z_benign      = gat_spec('scaler', 'z_benign',      g.losses.focal(),
                                 source_overrides={ scaler_strategy: 'z_benign' });
  local robust_benign = gat_spec('scaler', 'robust_benign', g.losses.focal(),
                                 source_overrides={ scaler_strategy: 'robust_benign' });

  // ----- id_encoding axis ----------------------------------------------
  local lookup = gat_spec('id_encoding', 'lookup', g.losses.focal(), model_extra={
    model+: { init_args+: {
      id_encoder_class_path: 'graphids.core.models.id_encoding.lookup.LookupIdEncoder',
    } },
  });
  local hash = gat_spec('id_encoding', 'hash', g.losses.focal(), model_extra={
    model+: { init_args+: {
      id_encoder_class_path: 'graphids.core.models.id_encoding.hash_embedding.HashIdEncoder',
      id_encoder_kwargs: { num_buckets: 2048 },
    } },
  });

  // ----- emit rows -------------------------------------------------------
  [
    // baseline
    g.row.fit('vgae',              vgae),              g.row.test('vgae',              vgae),
    // gat_loss
    g.row.fit('focal',             focal),             g.row.test('focal',             focal),
    g.row.fit('ce',                ce),                g.row.test('ce',                ce),
    g.row.fit('weighted_ce',       weighted_ce),       g.row.test('weighted_ce',       weighted_ce),
    // gat_sampling
    g.row.fit('none',              none),              g.row.test('none',              none),
    g.row.fit('curriculum_random', curriculum_random), g.row.test('curriculum_random', curriculum_random),
    g.row.fit('curriculum_vgae',   curriculum_vgae),   g.row.test('curriculum_vgae',   curriculum_vgae),
    // scaler
    g.row.fit('scaler_z',          z_benign),          g.row.test('scaler_z',          z_benign),
    g.row.fit('scaler_robust',     robust_benign),     g.row.test('scaler_robust',     robust_benign),
    // id_encoding
    g.row.fit('id_lookup',         lookup),            g.row.test('id_lookup',         lookup),
    g.row.fit('id_hash',           hash),              g.row.test('id_hash',           hash),
  ]
