// Fusion archetype composer — bandit / dqn / mlp / weighted_avg.
// All 4 share: FusionDataModule, mode='cpu', monitor='val_acc/max',
// upstreams = [vgae, focal] (auto-derived from meta).

local trainer    = import '../_kit/trainer.libsonnet';
local datamodule = import '../data/datamodule/fusion.libsonnet';
local callbacks  = import '../_kit/callbacks.libsonnet';
local v          = import '../_kit/validate.libsonnet';

function(model, method, meta,
         monitor='val_acc', mode='max',
         trainer_overrides={},
         patience=200,
         batch_size=128, episode_sample_size=20000,
         callback_extras={})

  // Standard fusion lineage — vgae + focal upstreams keyed off (dataset, seed).
  local upstreams = [
    {
      role: 'vgae',
      ckpt_path: std.native('paths.best_ckpt')(meta.dataset, 'unsupervised', 'vgae', meta.seed),
      ckpt_tla: 'vgae_ckpt_path',
    },
    {
      role: 'focal',
      ckpt_path: std.native('paths.best_ckpt')(meta.dataset, 'gat_loss', 'focal', meta.seed),
      ckpt_tla: 'gat_ckpt_path',
    },
  ];

  v.spec(
    trainer
    + model
    + datamodule(dataset=meta.dataset, seed=meta.seed, method=method,
                 batch_size=batch_size, episode_sample_size=episode_sample_size)
    + callbacks(monitor=monitor, mode=mode, patience=patience, extras=callback_extras)
    + {
      seed_everything: meta.seed,
      trainer+: {
        default_root_dir: std.native('paths.run_dir')(
          meta.dataset, meta.group, meta.variant, meta.seed
        ),
        // Fusion archetype trainer overrides. `accelerator` is derived
        // from `_resources.mode` by the row builder (single source of truth).
        precision: '32-true',         // RL methods need fp32
        gradient_clip_val: null,      // RL loss scales aren't gradient-clip-friendly
        max_epochs: 1500,
        log_every_n_steps: 10,
      } + trainer_overrides,
      _meta: meta,
      // Archetype-fixed: fusion-on-cached-states is CPU-bound (no GPU benefit).
      // Length is row-level.
      _resources: {mode: 'cpu'},
      _upstreams: upstreams,
    }
  )
