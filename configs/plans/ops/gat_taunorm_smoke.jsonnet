// Lightning-migration step 5+6 SLURM smoke — GAT + focal + TauNormCallback.
//
// Closes the open runtime gates from `~/plans/lightning-migration-spike.md`:
//   step 5 — bf16 autocast actually engages, MLflow per-epoch metrics +
//             LoggedModel land, Sha256ModelCheckpoint writes the .sha256 sidecar
//   step 6 — TauNormCallback resolves `fc_layers.<N>.weight` against a real
//             Lightning-saved ckpt and rewrites the best ckpt with normed head.
//
// max_epochs=2 keeps walltime ≪ 10 min on Pitzer V100.

local g = import '../../index.libsonnet';

function(dataset, seed)
  local gat_smoke = g.compose.supervised(
    model = g.models.supervised.gat(),
    data  = g.data.datamodule.graph(source=g.data.source.can_bus(dataset, seed)),
    loss  = g.losses.focal(),
    meta  = {
      group: 'lightning_migration_smoke', variant: 'gat_taunorm',
      dataset: dataset, seed: seed,
      model_type: 'gat', scale: 'small',
    },
    trainer_overrides = { max_epochs: 2 },
    callback_extras = {
      kang_tau_norm: {
        class_path: 'graphids.core.callbacks.TauNormCallback',
        init_args: { tau: 0.5 },
      },
    },
  );

  [
    g.row.fit('gat_taunorm', gat_smoke),
    g.row.test('gat_taunorm', gat_smoke),
  ]
