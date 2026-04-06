KD-GAT/
│
├── configs/ # declarative only, no Python
│ ├── \_lib/ # already exists — keep
│ │ ├── base.libsonnet
│ │ ├── slurm.libsonnet # absorbs ResourceSpec defaults
│ │ ├── training.libsonnet # stage-specific training defaults
│ │ └── utils.libsonnet
│ ├── datasets/
│ │ └── dataset_registry.json # already exists — keep
│ ├── stages/ # already exists — migrate YAMLs → jsonnet
│ │ ├── pretrain.jsonnet # autoencoder stage
│ │ ├── supervised.jsonnet # GAT stage
│ │ └── fusion.jsonnet # fusion stage (was fusion.libsonnet?)
│ ├── models/
│ │ ├── unsupervised.libsonnet  # was vgae + dgi
│ │ ├── supervised.libsonnet    # was gat
│ │ └── fusion/
│ │ ├── dqn.libsonnet
│ │ └── bandit.libsonnet
│ ├── resources/ # already exists — migrate → JSON
│ │ └── job_profiles.json # GAT gets x walltime etc → pure JSON
│ └── envs/
│ └── cluster.libsonnet # cluster-specific paths
│
├── graphids/
│ │
│ ├── config/ # Pydantic schemas — centralized by asset
│ │ ├── **init**.py # exports all configs
│ │ ├── shared.py # SlurmConfig, PathContext (already exists)
│ │ ├── pretrain.py # PretrainConfig (autoencoder stage)
│ │ ├── supervised.py # SupervisedConfig (GAT stage)
│ │ ├── fusion.py # FusionConfig (fusion stage)
│ │ ├── dataset.py # DatasetConfig — loaded from registry
│ │ └── jsonnet.py # render_config() — already exists, keep here
│ │
│ ├── core/
│ │ ├── contracts/ # TrainingSpec + envelope helpers — keep here
│ │ │ # these are runtime specs, not config schemas
│ │ ├── models/ # unchanged
│ │ ├── artifacts/ # unchanged
│ │ └── preprocessing/ # unchanged
│ │
│ ├── orchestrate/           # decomposed into subpackages (2026-04-06)
│ │ ├── contracts/           # TrainingSpec, build_tla_dict
│ │ ├── dagster/             # assets, checks, component, resources, runtime
│ │ ├── planning/            # planner, recipes, shared (StageConfig)
│ │ ├── resolve/             # resolver, cross_field
│ │ ├── ops/                 # entrypoint, catalog, finalize, status
│ │ ├── analysis.py
│ │ └── definitions.py
│ │
│ ├── slurm/                 # decomposed (2026-04-06)
│ │ ├── env.py               # centralized SLURM env var reads
│ │ ├── core/                # accounting, submit
│ │ ├── ops/                 # profile, staging
│ │ └── pipeline.py          # GraphIDS-specific spec plumbing
│ │
│ └── train_entrypoint.py # thin — calls render_config → validate → instantiate
