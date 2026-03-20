# KD-GAT Project Structure

## 3-Layer Hierarchy

```
graphids/               # Top-level package — __getattr__ lazy gateway for core/, pipeline/
  __init__.py          # Lazy gateway: PipelineConfig, resolve (eager); core/pipeline (lazy)
  __main__.py          # Single CLI: @hydra.main for training/sweep, argparse for subcommands
  api.py               # Programmatic facade: train(), evaluate(), orchestrate()
  logging.py           # structlog configure_logging(): processor pipeline, stdlib bridge
  config/               # Layer 1: Inert, declarative (no pipeline/ or core/ imports)
    _hydra_bridge.py    # resolve() via Hydra Compose API → PipelineConfig
    constants.py        # Project constants, load_pipeline_yaml(), topology (STAGES, STAGE_DEPENDENCIES)
    paths.py            # Lake path primitives, EnvironmentSettings (SLURM), dataset catalog
    schema.py           # All Pydantic models: PipelineConfig, architectures, DatasetEntry
    __init__.py         # Re-exports from all submodules
    pipeline.yaml       # Pipeline topology: model types, scales, stages, DAG dependencies
    datasets.yaml       # Dataset catalog (add entries here for new datasets)
    resources.yaml      # SLURM resource profiles + failure reactions
    conf/               # Hydra config groups (composed by _hydra_bridge.py)
      config.yaml       # Root config: defaults, infrastructure, callbacks, checkpoints, hydra dirs
      model/            # model_type × scale (compound names, @package _global_)
      auxiliary/         # Loss modifier config groups (none, kd_standard)
      dataset/          # Dataset identity config groups
  pipeline/             # Layer 2: Orchestration (imports graphids.config/, lazy graphids.core/)
    __init__.py         # Re-exports
    executor.py         # execute_stage() — single entry point for all pipeline paths
    validate.py         # Config + environment validation
    stages/             # Stage implementations
      training.py       # Training loop (autoencoder, curriculum, normal stages)
      evaluation.py     # Eval orchestrator + per-model evaluators + compute_metrics
      eval_types.py     # Frozen dataclasses: GATResult, VGAEResult, FusionResult
      eval_inference.py # Typed inference: run_gat/vgae/fusion_inference (batched)
      fusion.py         # Multi-model fusion stage (DQN, MLP, weighted avg)
      temporal.py       # Temporal graph classification (GAT encoder + Transformer)
      data_loading.py   # Dataset loading + graph caching + training_preamble()
      batch_sizing.py   # Batch size resolution (safety_factor × configured batch_size)
      trainer_factory.py # Lightning Trainer factory, model loading, KD teacher prep
      callbacks.py      # EvalArtifactCallback, RunMetadataCallback
      modules.py        # Lightning modules: VGAEModule, GATModule, CurriculumDataModule
      cka.py            # CKA computation
    orchestration/      # Pipeline orchestration (submitit + SLURM)
      dag.py            # DAG topology + run_dag() via graphlib + submitit
      job.py            # ResourceSpec (SLURM resource profiles)
      slurm.py          # submitit executor factory
  core/                 # Layer 3: Domain (models, preprocessing; imports graphids.config/)
    __init__.py         # Gateway: load_dataset, load_test_scenarios, get_model
    graph_utils.py      # PyG graph utilities: get_batch_index(), graph_attack_type()
    models/             # Model architectures
      vgae.py           # Variational Graph Autoencoder (GraphAutoencoderNeighborhood)
      gat.py            # Graph Attention Network (GATWithJK)
      dqn.py            # DQN fusion agent (EnhancedDQNFusionAgent) + MLP/WeightedAvg
      temporal.py       # Temporal model (GAT encoder + Transformer)
      fusion_features.py # Feature extraction for fusion models (Protocol-based)
      registry.py       # Model registry (type → class mapping + factory callables)
      _protocols.py     # Type contracts: GraphModel Protocol, StageMetrics TypedDict
      _utils.py         # Shared model utilities
    preprocessing/      # Graph construction from raw data
      _cache.py         # Dataset loading with intelligent caching
      _cache_metadata.py # Cache metadata and graph statistics
      _dataset.py       # CollatedGraphDataset: collated tensor storage
      _engine.py        # Preprocessing orchestration engine
      _parallel.py      # Parallel preprocessing workers
      _schema.py        # Feature manifests and data schemas
      _vocabulary.py    # Entity vocabulary (CAN ID mapping)
      adapters/         # Data source adapters
        _base.py        # Abstract base adapter
        _can_bus.py     # CAN bus CSV → PyG graph adapter
scripts/
  slurm/                # SLURM job scripts
    _preamble.sh        # Env setup (modules, venv, CUDA)
    _epilog.sh          # Post-job cleanup + HF push
    run_tests_slurm.sh  # Submit pytest to SLURM
    run_tests_parallel.sh # Parallel test runner
    launch_dagster.sh   # Dagster launcher (legacy)
  data/                 # Data management scripts
    push_experiments_to_hf.py # metrics.csv → Parquet → HF Dataset
    export_paper_data.py # Paper-ready data export
    generate_attack_type_mapping.py # Attack type mapping JSON
  lake/                 # ESS data lake management
    setup_ess.sh        # Create ESS directory tree
    migrate_to_ess.sh   # Migration script
  profiling/            # Profiling and benchmarking
    profile_conv_type.sbatch # Convolution type profiling
    run_pygod_baselines.py # PyGOD baseline comparisons
tests/
  conftest.py           # Shared fixtures
  test_*.py             # Unit and integration tests
```

## File Count

~50 Python files under `graphids/` (config: 5, pipeline: 15, core: 20, top-level: 4).
