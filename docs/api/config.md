# Config

Typed configuration follows module ownership:

- `graphids.exp.config` defines experiments, resolved runs, resources, outputs,
  and the fit/test payload (`FitRunPayload`).
- `graphids.data.config.CANEventDataConfig` defines event data loading and
  chronological stream batching.
- `graphids.models.config.EventGraphMambaConfig` defines the graph-Mamba model
  and its training objectives.
- `graphids.data.representation.CANEventRepresentationConfig` defines the
  message-preserving event contract.

Experiments may extend one relative base YAML. See
[configuration composition](../reference/config-architecture.md) for merge
rules, expanded config snapshots, and standalone compatibility.

`graphids.exp.runtime` resolves the built-in YAML types `can_event_dm` and
`event_graph_mamba` directly to their config classes. Custom components and
callbacks use `class_path` with optional `init_args`. Config builders and direct
data/model constructors share validation; checkpoint reconstruction uses the
same model settings.

Python imports use the domain packages above. The old `graphids.core` and
`graphids.primitives*` modules have been removed without forwarding aliases.
YAML type names, config fields, and serialized artifact schemas are unchanged.

::: graphids.exp.config
