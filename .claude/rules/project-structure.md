# KD-GAT Project Structure

## 4-Layer Hierarchy

```
graphids/               # Top-level package — __getattr__ lazy gateway for core/, pipeline/, storage/
  __init__.py          # Lazy gateway: PipelineConfig, resolve, checkpoint_path (eager); core/pipeline/storage (lazy)
  api.py               # Programmatic facade: train(), evaluate(), orchestrate() — for notebooks/Dagster
  logging.py           # structlog configure_logging(): processor pipeline, stdlib bridge, JSON/console renderers
  storage/              # Layer 0: Infrastructure (no imports from config/, pipeline/, or core/)
    __init__.py         # Re-exports: StorageGateway, ArtifactMapper, open_gateway, lake path primitives, contracts, manifest, catalog
    gateway.py          # NFS-safe I/O: atomic writes (tmpfile+fsync+rename), advisory locking (fcntl.flock), path resolution
    mapper.py           # Domain-aware serialization: checkpoints, configs, eval artifacts, cache, pickle (lazy domain imports)
    paths.py            # Lake path layout: lake_run_dir, lake_cache_dir, lake_raw_dir, etc.
    manifest.py         # _manifest.json writer/reader + SHA-256 checksum verification
    catalog.py          # DuckDB catalog rebuild from manifests + status query
    contracts.py        # Artifact validation: StageArtifact, TrainingArtifact, EvaluationArtifact, PreprocessingArtifact
  config/               # Layer 1: Inert, declarative (imports storage/ for path primitives, no pipeline/ or core/)
    _hydra_bridge.py    # resolve() via Hydra Compose API → PipelineConfig
    constants.py        # Project constants, load_pipeline_yaml(), topology (STAGES, STAGE_DEPENDENCIES, etc.)
    paths.py            # PipelineConfig-based path helpers (stage_dir, checkpoint_path), EnvironmentSettings (SLURM/MLflow). Lake primitives re-exported from storage/paths.py
    schema.py           # All Pydantic models: PipelineConfig (Literal model_type/scale), architectures, DatasetEntry, artifact contracts
    __init__.py         # Re-exports from all submodules. All external: `from graphids.config import X`
    pipeline.yaml       # Pipeline topology: model types, scales, stages, DAG dependencies
    datasets.yaml       # Dataset catalog (add entries here for new datasets)
    resources.yaml      # SLURM resource profiles + slurm_defaults + failure reactions
    conf/               # Hydra config groups (composed by _hydra_bridge.py)
      config.yaml       # Root config: defaults list, infrastructure, stages, variants
      model/            # model_type × scale (compound names, @package _global_)
        vgae_large.yaml, vgae_small.yaml, gat_large.yaml, gat_small.yaml, dqn_large.yaml, dqn_small.yaml
      auxiliary/         # Loss modifier config groups
        none.yaml, kd_standard.yaml
      dataset/          # Dataset identity config groups
        hcrl_sa.yaml, hcrl_ch.yaml, set_01.yaml, set_02.yaml, set_03.yaml, set_04.yaml
    search_spaces/      # HPO search space definitions (for Ray Tune)
      vgae.yaml, gat.yaml, dqn.yaml
  lake/                 # ESS data lake I/O (imports config/ only, no pipeline/ or core/)
    __init__.py         # Gateway: write_manifest, rebuild_catalog (lazy)
    catalog.py          # DuckDB catalog rebuild from _manifest.json
    manifest.py         # _manifest.json writer/reader + SHA-256 checksum verification
  pipeline/             # Layer 2: Orchestration (imports graphids.config/, graphids.storage/, lazy imports from graphids.core/)
    __init__.py         # Gateway: build_cli_cmd, STAGE_FNS
    cli.py              # Entry point: Hydra override grammar for training, argparse for subcommands; MLflow run context
    validate.py         # Config + environment validation utilities
    subprocess_utils.py # Shared CLI command builder for subprocess dispatch
    stages/             # Stage implementations
      training.py       # Training loop (autoencoder, curriculum, normal stages)
      evaluation.py     # Eval orchestrator + per-model evaluators + compute_metrics + probe_embedding_dim
      eval_types.py     # Frozen dataclasses: GATResult, VGAEResult, FusionResult
      eval_inference.py # Typed inference: run_gat/vgae/fusion_inference (batched via PyG DataLoader)
      fusion.py         # Multi-model fusion stage (DQN, MLP, weighted avg)
      temporal.py       # Temporal graph classification (GAT encoder + Transformer over time)
      data_loading.py   # Dataset loading + graph caching + training_preamble()
      batch_sizing.py   # Batch size resolution (safety_factor × configured batch_size)
      trainer_factory.py # Lightning Trainer + ModelCheckpoint + EarlyStopping + DeviceStatsMonitor + MLflow autolog
      modules.py        # Lightning modules: VGAEModule, GATModule, CurriculumDataModule + teacher offload helpers
      utils.py          # Re-exports from submodules (convenience imports)
    orchestration/      # Pipeline orchestration (Dagster + SLURM)
      __init__.py       # Gateway: ResourceSpec, PipesSlurmClient, SlurmJobFailed; lazy Dagster imports
      job.py            # Pydantic v2 frozen models: ResourceSpec (SLURM resource profiles)
      dagster_defs.py   # Dagster asset definitions + build_dag_topology() + fire_and_forget()
      pipes_slurm.py    # Dagster Pipes SLURM client (PipesClient + ConfigurableResource over NFS)
      slurm_primitives.py # SLURM primitives: sbatch gen, submit, poll, adaptive retry, resource profiles
      optuna_sweep.py   # Optuna HPO: run_sweep() + run_sweep_pipeline() (SQLite-backed resume)
  core/                 # Layer 3: Domain (models, data loading, preprocessing; imports graphids.config/)
    __init__.py         # Gateway: load_dataset, load_test_scenarios, get_model, process_dataset
    data.py             # Dataset loading with NFS-safe caching (was core/training/datamodules.py)
    graph_utils.py      # PyG graph utilities: get_batch_index(), graph_attack_type()
    models/             # Model architectures
      vgae.py           # Variational Graph Autoencoder (GraphAutoencoderNeighborhood)
      gat.py            # Graph Attention Network (GATWithJK)
      dqn.py            # DQN fusion agent (EnhancedDQNFusionAgent) + MLP/WeightedAvg baselines
      temporal.py       # Temporal model (GAT encoder + Transformer)
      fusion_features.py # Feature extraction for fusion models (Protocol-based)
      registry.py       # Model registry (type → class mapping + typed factory callables)
      _protocols.py     # Type contracts: GraphModel Protocol, StageMetrics TypedDict
      _utils.py         # Shared model utilities (checkpoint_conv)
    training/           # Backward-compat re-exports from core/data.py
    preprocessing/      # Graph construction from raw data
      dataset.py        # CollatedGraphDataset: collated tensor storage (zero-copy __getitem__)
      engine.py         # Preprocessing orchestration engine
      temporal.py       # TemporalGrouper (sliding window → temporal graphs)
      vocabulary.py     # Feature vocabulary building
      parallel.py       # Parallel preprocessing workers
      schema.py         # Preprocessing data schemas
      adapters/         # Data source adapters
        base.py         # Abstract base adapter
        can_bus.py      # CAN bus CSV → PyG graph adapter
data/
  ethernet/             # Network flow datasets (MachineLearningCSV, GeneratedLabelledFlows)
experimentruns/         # Legacy outputs (migrated to ESS data lake)
tests/
  conftest.py               # Shared fixtures: module-scoped DAG topology, asset lookup, resource factories
  test_resource_spec.py     # ResourceSpec Pydantic model: SLURM formatting, from_yaml, immutability
  test_slurm_primitives.py  # Script gen, resource profiles, adaptive retry, sacct parsing, polling
  test_dag_topology.py      # DAG construction, Dagster asset wiring, partition defs, Pipes client
  test_fire_and_forget.py   # Zero-daemon SLURM submission (monkeypatched, no real sbatch)
  test_pipes_slurm.py       # Pipes client integration, submit_no_poll, CLI subcommands
scripts/
  reproduce.sh          # Full reproduction script
  slurm/                # SLURM job scripts
    _preamble.sh            # Env setup (modules, venv, CUDA, MLflow)
    run_tests_slurm.sh  # Submit pytest to SLURM
    run_tests_parallel.sh # Parallel test runner
    sweep.sh            # Hyperparameter sweep submission
    job_epilog.sh       # Post-job cleanup
  data/                 # Data management scripts
    stage_data.sh       # Stage datasets from scratch/archive
    cleanup_orphans.sh  # Clean orphaned cache/output files
    push_experiments_to_hf.py # MLflow → Parquet → HF Dataset + ESS exports
  lake/                 # ESS data lake management
    setup_ess.sh        # Create ESS directory tree + layout files
    migrate_to_ess.sh   # rsync + restructure migration (adds seed subdirs)
  profiling/            # Profiling and benchmarking
    analyze_profile.py  # Profile analysis
    benchmark_orchestration.sbatch # Orchestration overhead benchmarks
    profile_conv_type.sbatch # Convolution type profiling
    run_pygod_baselines.py # PyGOD baseline comparisons
  dev/                  # Developer utilities
    setup_tmux.sh       # tmux session setup
    start_jupyter.sh    # Jupyter server launcher
notebooks/
  analysis/             # Analysis notebooks
    01_training_curves.ipynb    # Training performance visualization
    02_evaluation_results.ipynb # Evaluation metrics analysis
    03_analytics.ipynb          # DuckDB analytics exploration
    04_artifact_analysis.ipynb  # Model artifact analysis (embeddings, attention, CKA)
  prototyping/          # Prototyping and exploration
    playground.ipynb            # General experimentation
    deno_plot_template.ipynb    # Deno/Observable Plot template
    sample_bubble_chart.ipynb   # Bubble chart prototype
```

## File Count

~65 Python files under `graphids/` (storage: 7, config: 5, pipeline: 17, core: 24, top-level: 3).
