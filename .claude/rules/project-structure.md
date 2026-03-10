# KD-GAT Project Structure

## 3-Layer Hierarchy

```
graphids/               # Top-level package (pyproject.toml: packages = ["graphids"])
  config/               # Layer 1: Inert, declarative (no imports from pipeline/ or core/)
    schema.py           # Pydantic v2 frozen models: PipelineConfig, VGAEArchitecture, etc.
    resolver.py         # YAML composition: defaults → model_def → auxiliaries → CLI
    paths.py            # Path layout: {dataset}/{model_type}_{scale}_{stage}[_{aux}]
    catalog.py          # Data catalog: dataset registry + validation
    constants.py        # Domain/infrastructure constants (window sizes, feature counts, etc.)
    __init__.py         # Re-exports: from graphids.config import PipelineConfig, resolve, checkpoint_path, ...
    defaults.yaml       # Global baseline config values
    datasets.yaml       # Dataset catalog (add entries here for new datasets)
    models/             # Architecture × Scale YAML files
      vgae/large.yaml, small.yaml
      gat/large.yaml, small.yaml
      dqn/large.yaml, small.yaml
      fusion/dqn.yaml, mlp.yaml, weighted_avg.yaml
    auxiliaries/        # Loss modifier YAML files (composable)
      none.yaml, kd_standard.yaml
  pipeline/             # Layer 2: Orchestration (imports graphids.config/, lazy imports from graphids.core/)
    cli.py              # Entry point + MLflow run context + artifact logging + archive restore on failure
    serve.py            # FastAPI inference server (/predict, /health)
    validate.py         # Config + environment validation utilities
    stages/             # Stage implementations
      training.py       # Training loop (autoencoder, curriculum, normal stages)
      evaluation.py     # Multi-model eval; captures embeddings.npz + dqn_policy.json
      fusion.py         # Multi-model fusion stage (DQN, MLP, weighted avg)
      temporal.py       # Temporal graph classification (GAT encoder + Transformer over time)
      data_loading.py   # Dataset loading + graph caching + training_preamble()
      batch_sizing.py   # Batch size resolution (safety_factor × configured batch_size)
      trainer_factory.py # Lightning Trainer + ModelCheckpoint + EarlyStopping + DeviceStatsMonitor + MLflow autolog
      modules.py        # Lightning modules: VGAEModule, GATModule, CurriculumDataModule + teacher offload helpers
      loss_landscape.py # Loss landscape visualization (standalone analysis tool)
      utils.py          # Re-exports from submodules (convenience imports)
    orchestration/      # Ray orchestration
      ray_pipeline.py   # Config-driven variant pipeline, subprocess dispatch, benchmark mode
      ray_slurm.py      # SLURM ↔ Ray bridge
      sweep_pipeline.py # Hyperparameter sweep orchestration
      tune_config.py    # Ray Tune search space + OptunaSearch + ASHAScheduler
  core/                 # Layer 3: Domain (models, training, preprocessing; imports graphids.config/)
    models/             # Model architectures
      vgae.py           # Variational Graph Autoencoder (GraphAutoencoderNeighborhood)
      gat.py            # Graph Attention Network (GATWithJK)
      dqn.py            # DQN fusion agent (EnhancedDQNFusionAgent) + MLP/WeightedAvg baselines
      temporal.py       # Temporal model (GAT encoder + Transformer)
      fusion_features.py # Feature extraction for fusion models
      registry.py       # Model registry (type → class mapping + feature extractors)
      _utils.py         # Shared model utilities (checkpoint_conv)
    training/           # Data management
      datamodules.py    # Lightning DataModule: dataset loading, splits, DataLoader construction
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
  automotive/           # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  ethernet/             # Network flow datasets (MachineLearningCSV, GeneratedLabelledFlows)
  cache/                # Preprocessed graph cache (.pt, .pkl, metadata)
  mlflow/               # MLflow SQLite backend (mlflow.db + artifacts/)
experimentruns/         # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json
tests/
  conftest.py           # Shared fixtures (tiny architectures, temp dirs, E2E_OVERRIDES)
  test_layer_boundaries.py  # Import hierarchy enforcement (config ← pipeline ← core)
  test_preprocessing.py     # Preprocessing unit tests
  test_registry.py          # Model registry tests
  test_fusion_extractors.py # Fusion feature extraction tests
  test_serve.py             # FastAPI serving tests
  test_training_smoke.py    # Training smoke tests (quick sanity checks)
  test_training_e2e.py      # End-to-end training tests (full pipeline, SLURM only)
  test_pipeline_integration.py # Pipeline integration tests
  test_new_features.py      # New feature regression tests
scripts/
  reproduce.sh          # Full reproduction script
  slurm/                # SLURM job scripts
    ray_slurm.sbatch        # Ray + SLURM pipeline submission
    run_tests_slurm.sh  # Submit pytest to SLURM
    run_tests_parallel.sh # Parallel test runner
    sweep.sh            # Hyperparameter sweep submission
    job_epilog.sh       # Post-job cleanup
  data/                 # Data management scripts
    stage_data.sh       # Stage datasets from scratch/archive
    cleanup_orphans.sh  # Clean orphaned cache/output files
    push_experiments_to_hf.py # MLflow → Parquet → HF Dataset
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

48 Python files under `graphids/` (down from 55 after Mar 2026 simplification).
