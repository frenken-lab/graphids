# Topology

Import-time coherence check for the jsonnet config tree (every model
family has a libsonnet, every fusion method has a method libsonnet,
every stage has a ``.jsonnet``), plus path helpers and the dataset
catalog loader. Failures raise at package load, not at sbatch time.

## `graphids.config.topology`

::: graphids.config.topology
