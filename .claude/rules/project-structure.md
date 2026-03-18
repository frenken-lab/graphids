# KD-GAT Project Structure

## 3-Layer Hierarchy

```
graphids/               # Top-level package — __getattr__ lazy gateway for core/, pipeline/, lake/
  __init__.py          # Lazy gateway: PipelineConfig, resolve, checkpoint_path (eager); core/pipeline/lake (lazy)
  api.py               # Programmatic facade: train(), evaluate(), orchestrate() — for notebooks/Dagster
  lake/                 # Layer 1b: ESS data lake (imports config/ only, no pipeline/ or core/)
    __init__.py         # Gateway: LakeConfig, write_manifest, cache_lock, rebuild_catalog (lazy)
    config.py           # LakeConfig Pydantic model + env var resolution ($KD_GAT_LAKE_ROOT)
    catalog.py          # DuckDB catalog rebuild from _manifest.json + config.json + metrics.json
    manifest.py         # _manifest.json writer/reader + SHA-256 checksum verification
    locking.py          # GPFS-safe advisory file locking (fcntl.flock) for cache writes
  config/               # Layer 1: Inert, declarative (no imports from pipeline/, core/, or lake/)
    handler.py          # ConfigHandler class + EnvironmentSettings (pydantic-settings) — YAML loading, resolution, paths
    schema.py           # All Pydantic models: PipelineConfig, architectures, DatasetEntry, artifact contracts
    __init__.py         # Singleton _api = ConfigHandler() + re-exports. All external: `from graphids.config import X`
    pipeline.yaml       # Pipeline topology + preprocessing constants + defaults + path defaults (single source of truth)
    datasets.yaml       # Dataset catalog (add entries here for new datasets)
    resources.yaml      # SLURM resource profiles + slurm_defaults + failure reactions
    models/             # Architecture × Scale YAML files (only overrides; Pydantic defaults are baseline)
      vgae/large.yaml, small.yaml
      gat/large.yaml, small.yaml
      dqn/large.yaml, small.yaml
    auxiliaries/        # Loss modifier YAML files (composable)
      kd_standard.yaml
    search_spaces/      # HPO search space definitions (for Ray Tune)
      vgae.yaml, gat.yaml, dqn.yaml
  pipeline/             # Layer 2: Orchestration (imports graphids.config/, lazy imports from graphids.core/)
    __init__.py         # Gateway: build_cli_cmd, STAGE_FNS
    cli.py              # Entry point + MLflow run context; lifecycle: _archive_previous, _log_stage_artifacts, _write_lake_manifest
    artifacts.py        # Artifact store: get/put/exists with cache → filesystem → MLflow fallback
    serve.py            # FastAPI inference server (/predict, /health)
    validate.py         # Config + environment validation utilities
    subprocess_utils.py # Shared CLI command builder for subprocess dispatch
    stages/             # Stage implementations
      training.py       # Training loop (autoencoder, curriculum, normal stages)
      evaluation.py     # Eval orchestrator + per-model evaluators (_evaluate_gat/vgae/fusion/temporal); batched inference via PyG DataLoader; torchmetrics
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
      dagster_resources.py # Retry state helpers (per-asset failure metadata)
      pipes_slurm.py    # SLURM sbatch/sacct wrapper: script gen, submit, poll, artifact validation
      sweep_pipeline.py # Hyperparameter sweep orchestration (SQLite-backed state)
      tune_config.py    # Ray Tune search space + OptunaSearch + ASHAScheduler
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
  conftest.py           # Shared fixtures (tiny architectures, temp dirs, E2E_OVERRIDES)
  test_layer_boundaries.py  # Import hierarchy + gateway enforcement (config ← pipeline ← core)
  test_dagster_orchestration.py # Dagster assets, fire-and-forget, resource profiles (62 tests)
  test_preprocessing.py     # Preprocessing unit tests
  test_registry.py          # Model registry tests
  test_fusion_extractors.py # Fusion feature extraction tests
  test_serve.py             # FastAPI serving tests
  test_training_smoke.py    # Training smoke tests (quick sanity checks)
  test_training_e2e.py      # End-to-end training tests (full pipeline, SLURM only)
  test_pipeline_integration.py # Pipeline integration tests
  test_new_features.py      # New feature regression tests
  test_lake.py              # Lake module tests (config, manifest, locking, catalog)
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
docs/
  ECOSYSTEM.md          # Dependency ecosystem documentation
  memory_optimization.md  # Memory optimization strategies (DeviceStatsMonitor + DynamicBatchSampler)
```

## File Count

57 Python files under `graphids/` (config: 3, pipeline: 15 incl. artifacts.py, core: 16, lake: 5, top-level: 2).
