// Single-import namespace mirroring graphids/core/ structure.
// Plans use `local g = import '../index.libsonnet';` and reach primitives via
// `g.models.autoencoder.vgae(...)`, `g.compose.supervised(...)`, etc.
//
// Layout mirrors graphids/core/:
//   g.models.{autoencoder,supervised,fusion}.<arch>  ↔ graphids/core/models/...
//   g.losses.<name>                                  ↔ graphids/core/losses/...
//   g.data.{source,datamodule}.<name>                ↔ graphids/core/data/...
//
// `g.compose.<archetype>` and `g.row` have no graphids/ counterpart — they
// are jsonnet-only glue (archetype composers + plan row builder).

{
  models: {
    autoencoder: {
      vgae: import 'models/autoencoder/vgae.libsonnet',
      dgi:  import 'models/autoencoder/dgi.libsonnet',
    },
    supervised: {
      gat: import 'models/supervised/gat.libsonnet',
    },
    fusion: {
      bandit:       import 'models/fusion/bandit.libsonnet',
      dqn:          import 'models/fusion/dqn.libsonnet',
      mlp:          import 'models/fusion/mlp.libsonnet',
      weighted_avg: import 'models/fusion/weighted_avg.libsonnet',
    },
  },
  losses: {
    focal:       import 'losses/focal.libsonnet',
    ce:          import 'losses/ce.libsonnet',
    weighted_ce: import 'losses/weighted_ce.libsonnet',
    vgae_task:   import 'losses/vgae_task.libsonnet',
    curriculum:  import 'losses/curriculum.libsonnet',
  },
  data: {
    source: {
      can_bus: import 'data/source/can_bus.libsonnet',
    },
    datamodule: {
      graph:  import 'data/datamodule/graph.libsonnet',
      fusion: import 'data/datamodule/fusion.libsonnet',
    },
  },
  compose: {
    unsupervised: import 'compose/unsupervised.libsonnet',
    supervised:   import 'compose/supervised.libsonnet',
    fusion:       import 'compose/fusion.libsonnet',
  },
  row: import '_kit/row.libsonnet',
}
