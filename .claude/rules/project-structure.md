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
    cli.py              # Entry point + W&B init + lakehouse sync + archive restore on failure
    serve.py            # FastAPI inference server (/predict, /health)
    validate.py         # Config + environment validation utilities
    errors.py           # Custom exception classes
    tracking.py         # Memory monitoring utilities
    export.py           # Datalake/filesystem → static JSON/Parquet export for Quarto reports
    memory.py           # GPU memory management (static, measured, trial-based batch sizing)
    lakehouse.py        # Datalake Parquet append (fire-and-forget)
    sweep_export.py     # Ray Tune results → datalake + HF Dataset
    stages/             # Stage implementations
      training.py       # Training loop (autoencoder, curriculum, normal stages)
      evaluation.py     # Multi-model eval; captures embeddings.npz + dqn_policy.json + explanations.npz
      fusion.py         # Multi-model fusion stage (DQN, MLP, weighted avg)
      temporal.py       # Temporal graph classification (GAT encoder + Transformer over time)
      data_loading.py   # Dataset loading + graph caching pipeline
      batch_sizing.py   # Dynamic batch size optimization
      trainer_factory.py # Lightning Trainer construction
      callbacks.py      # Training callbacks (checkpointing, early stopping, etc.)
      modules.py        # Shared Lightning module base classes
      utils.py          # Stage utility functions
    orchestration/      # Ray orchestration (ray_pipeline, ray_slurm, tune_config)
  core/                 # Layer 3: Domain (models, training, preprocessing; imports graphids.config/)
    models/             # Model architectures
      vgae.py           # Variational Graph Autoencoder
      gat.py            # Graph Attention Network
      dqn.py            # Deep Q-Network
      temporal.py       # Temporal model (GAT encoder + Transformer)
      fusion_features.py # Feature extraction for fusion models
      registry.py       # Model registry (type → class mapping)
      _utils.py         # Shared model utilities
    explain.py          # GNNExplainer integration (feature importance analysis)
    training/           # Data management
      datamodules.py    # Lightning DataModule: dataset loading, splits, DataLoader construction
    preprocessing/      # Graph construction from raw data
      dataset.py        # PyG InMemoryDataset wrapper
      engine.py         # Preprocessing orchestration engine
      temporal.py       # TemporalGrouper (sliding window → temporal graphs)
      vocabulary.py     # Feature vocabulary building
      parallel.py       # Parallel preprocessing workers
      schema.py         # Preprocessing data schemas
      adapters/         # Data source adapters
        base.py         # Abstract base adapter
        can_bus.py      # CAN bus CSV → PyG graph adapter
        network_flow.py # Network flow → PyG graph adapter
data/
  automotive/           # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  ethernet/             # Network flow datasets (MachineLearningCSV, GeneratedLabelledFlows)
  cache/                # Preprocessed graph cache (.pt, .pkl, metadata)
  datalake/             # Parquet structured storage (runs, metrics, configs, artifacts, training_curves/)
                        # queries/ (leaderboard.sql, kd_impact.sql)
experimentruns/         # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json, explanations.npz
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
    ray_slurm.sh        # Ray + SLURM pipeline submission
    run_tests_slurm.sh  # Submit pytest to SLURM
    run_tests_parallel.sh # Parallel test runner
    sweep.sh            # Hyperparameter sweep submission
    job_epilog.sh       # Post-job cleanup
  data/                 # Data management scripts
    stage_data.sh       # Stage datasets from scratch/archive
    build_test_cache.sh # Build preprocessed test cache
    cleanup_orphans.sh  # Clean orphaned cache/output files
  profiling/            # Profiling and benchmarking
    analyze_profile.py  # Profile analysis
    benchmark_orchestration.sh # Orchestration overhead benchmarks
    profile_conv_type.sh # Convolution type profiling
    run_pygod_baselines.py # PyGOD baseline comparisons
  dev/                  # Developer utilities
    generate_sweep.py   # Hyperparameter sweep config generator
    setup_tmux.sh       # tmux session setup
    start_jupyter.sh    # Jupyter server launcher
    s3_bucket_policy.json # S3 bucket policy template
    s3_cors.json        # S3 CORS config template
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
  memory_optimization.md  # Memory optimization strategies
reports/                # Quarto website — paper chapters + interactive dashboard (PRIMARY)
  _quarto.yml           # Project config (website, HTML + Typst + Revealjs)
  index.qmd             # Landing page
  dashboard.qmd         # Multi-page dashboard (Overview, Performance, Training, GAT & DQN, KD, Graph, Datasets, Staging)
  playground.qmd        # Visualization playground (SQL console, chart builder, scratch cells)
  slides.qmd            # Revealjs presentation
  pipeline_dag.svg      # Pipeline DAG visualization
  custom.scss           # Theme overrides
  references.bib        # BibTeX bibliography
  data/                 # Report data (Parquet + JSON from export pipeline, incl. graph_samples.json)
  figures/              # YAML specs for Mosaic vgplot (28 declarative chart definitions)
  _ojs/                 # Observable JS modules (mosaic-setup.js, mosaic-renderer.js, theme.js, table-renderer.js, query-utils.js)
  paper/                # Research paper (10 chapters with interactive Mosaic figures)
    index.qmd           # Paper introduction
    02-background.qmd through 09-conclusion.qmd  # Paper body
    10-appendix.qmd     # Appendix with model sizing details
    _metadata.yml       # Paper-specific metadata + shared _setup.qmd include
    _setup.qmd          # Shared Mosaic/vgplot + DuckDB-WASM init for figures
    references.bib      # Paper bibliography
    data/               # Paper-specific CSV data (ablation, datasets, model params)
  _site/                # Build output (.gitignored)
```
