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
│ ├── models/ # already exists — migrate → jsonnet
│ │ ├── gat.libsonnet
│ │ ├── dgi.libsonnet
│ │ ├── vgae.libsonnet
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
│ ├── orchestrate/
│ │ ├── assets.py # Dagster assets — unchanged structure
│ │ ├── definitions.py # unchanged
│ │ ├── planning.py # enumerate_assets (StageConfig lives in graphids/config/shared.py)
│ │ ├── resolve.py # cross-field logic in config/schemas.py
│ │ ├── checks.py # Dagster asset checks — keep
│ │ ├── analysis.py # keep
│ │ └── component.py # keep
│ │
│ ├── slurm/
│ │ ├── __init__.py
│ │ ├── resources.py
│ │ └── slurm.py
│ │ # ResourceSpec MOVES to graphids/config/shared.py
│ │
│ └── train_entrypoint.py # thin — calls render_config → validate → instantiate
