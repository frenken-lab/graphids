// Fusion dispatch table — exposes base + methods indexed by method name
// so `stages/fusion.jsonnet` can pick a method with `fusion.methods[m]`.

{
  base: import 'fusion/base.libsonnet',
  methods: {
    bandit: import 'fusion/methods/bandit.libsonnet',
    dqn: import 'fusion/methods/dqn.libsonnet',
    mlp: import 'fusion/methods/mlp.libsonnet',
    weighted_avg: import 'fusion/methods/weighted_avg.libsonnet',
  },
}
