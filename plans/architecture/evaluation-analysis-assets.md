# Plan: Evaluation + Analysis Artifacts as Dagster Assets

## Context

Training checkpoints are dagster assets, but evaluation and analysis are manual CLI invocations. `pipeline.yaml:65-71` already defines an `evaluation` stage, and `Analyzer` (`core/artifacts/analyzer.py`) already generates paper-ready artifacts. This plan wires both into the dagster pipeline as first-class assets with SLURM submission, partitioned by dataset×seed, and IOManager-based checkpoint path handoff from upstream training.

## Approach: Extend existing patterns, two PRs

### PR 1 — Evaluation assets

**1. `slurm.py:43` — `generate_script()` gains `subcommand` param**

```python
def generate_script(config_files, resources, *, subcommand="fit", ...)
    parts = [f"python -m graphids {subcommand}"]
```

Only change to slurm.py. Stays dagster-free. Default `"fit"` preserves backward compat.

**2. `StageConfig` — add `subcommand` field**

```python
@dataclass(frozen=True)
class StageConfig:
    ...
    subcommand: str = "fit"  # "fit" | "test" | "analyze"
```

**3. `enumerate_assets()` — emit evaluation StageConfigs**

Evaluation is already in `pipeline.yaml:65-71`. Currently `enumerate_assets()` iterates recipe stages and builds configs. Evaluation configs:
- `subcommand = "test"`
- `config_files`: same as upstream fusion (or upstream training stage)
- `upstream_ckpt_flags`: `{upstream_asset: "--ckpt_path"}` (Lightning test needs the checkpoint)
- `identity`: inherits from upstream (same identity_keys, minus fusion-specific ones)

**4. `_make_asset()` — dispatch on subcommand**

Currently the `_train` inner function builds `--model.init_args.*` CLI args. For `test`:
- Skip model overrides (model is loaded from checkpoint)
- Pass `--ckpt_path {upstream_ckpt}` instead
- Return `str(rd_path)` (run dir with metrics.csv) instead of checkpoint path

Extract shared SLURM submission boilerplate (submit, poll, observe, metadata) into `_submit_slurm()` helper used by both training and evaluation code paths.

**5. `ablation.yaml` — add evaluation to default stages**

```yaml
defaults:
  stages: [autoencoder, curriculum, fusion, evaluation]
```

This matches `pipeline.yaml:82` `default_stages`.

**6. `resources.yaml` — evaluation profile**

`resources.yaml:187-194` already has a `test` profile (cpu, 30min, 16G). Map `evaluation` stage → `test` profile. Options:
- a) Add `eval` model type mirroring `test` profile
- b) Map `evaluation` stage to existing `test` resource lookup in `get_resources()`

Prefer (a) — explicit entry, follows existing convention.

**7. IOManager — no change needed**

Evaluation assets are leaf nodes (nothing downstream consumes their output). They consume upstream checkpoint paths via IOManager `load_input()` but their own return value (metrics dir path) doesn't need sidecar storage. If DuckDB catalog is added later, it can read metrics.csv files directly from the filesystem.

### PR 2 — Analysis artifacts

**8. `pipeline.yaml` — three analysis stages**

```yaml
analyze_vgae:
  learning_type: analysis
  model: vgae
  mode: gpu_eval
  depends_on:
    - { model: vgae, stage: autoencoder }
  identity_keys: [scale, conv_type, variational]

analyze_gat:
  learning_type: analysis
  model: gat
  mode: gpu_eval
  depends_on:
    - { model: gat, stage: curriculum }
  identity_keys: [scale, conv_type, loss_fn, variational]

analyze_fusion:
  learning_type: analysis
  model: dqn
  mode: gpu_eval
  depends_on:
    - { model: vgae, stage: autoencoder }
    - { model: gat, stage: curriculum }
    - { model: dqn, stage: fusion }
  identity_keys: [scale, gat_stage, loss_fn, method, conv_type, variational]
```

Three stages, not one, because dependencies differ (VGAE analysis needs only autoencoder ckpt; fusion analysis needs all three).

**9. `_make_asset()` — analyze subcommand**

For `analyze`:
- CLI: `python -m graphids analyze --config stages/analyze_{model_type}.yaml`
- `--analyzer.ckpt_path={upstream_ckpt}` (primary model)
- `--analyzer.dataset={dataset}`
- `--analyzer.output_dir={run_dir}/artifacts/`
- For fusion: `--analyzer.vgae_ckpt_path=...` + `--analyzer.gat_ckpt_path=...`
- For CKA (KD configs): `--analyzer.cka_teacher_ckpt={teacher_ckpt}`
- Return `str(rd_path / "artifacts")`

**10. Recipe-level opt-in**

```yaml
# ablation.yaml
defaults:
  stages: [autoencoder, curriculum, fusion, evaluation]
  analysis: [analyze_vgae, analyze_gat, analyze_fusion]  # opt-in
```

`enumerate_assets()` only emits analysis StageConfigs when `analysis` key is present. Omit to skip expensive artifacts.

**11. `resources.yaml` — analysis profiles**

```yaml
vgae:
  small:
    analyze:
      partition: gpu
      gres: "gpu:1"
      time: "01:00:00"
      mem: "24G"
      cpus_per_task: 2
      num_workers: 0
  # ... large scale with more time for landscape
```

Analysis needs GPU (landscape + embeddings) but less time than training. Loss landscape (51×51×500 graphs) is the bottleneck — ~30-60 min on V100.

**12. Upstream checkpoint flag mapping**

New mapping for analyze subcommands:
```python
_ANALYZE_CKPT_FLAG = {
    "vgae": "--analyzer.ckpt_path",
    "gat": "--analyzer.ckpt_path",
    "dqn": "--analyzer.ckpt_path",
}
_ANALYZE_EXTRA_FLAGS = {
    # For fusion analysis — needs all three upstream ckpts
    "analyze_fusion": {
        "vgae": "--analyzer.vgae_ckpt_path",
        "gat": "--analyzer.gat_ckpt_path",
    }
}
```

## Files to modify

| File | Change | Size |
|------|--------|------|
| `orchestrate/slurm.py:43` | `subcommand` param on `generate_script()` | 2 lines |
| `orchestrate/component.py` | `StageConfig.subcommand`, `enumerate_assets()` eval+analysis, `_make_asset()` dispatch, `_submit_slurm()` helper | ~80 lines |
| `config/pipeline.yaml` | 3 analysis stage definitions | ~25 lines |
| `config/resources.yaml` | eval + analysis resource profiles | ~30 lines |
| `config/recipes/ablation.yaml` | evaluation in defaults, analysis toggle | 2 lines |

No new files. No new dependencies. No torch imports at definition time.

## Verification

1. `dg check defs` — definitions load with new stages
2. `python -m graphids.orchestrate validate` — all config chains parse (including evaluation + analysis)
3. `python -m graphids.orchestrate smoke --dry-run` — dry-run shows correct CLI commands for test and analyze subcommands
4. Count assets: expect ~32 training + ~32 evaluation + analysis assets (exact count depends on recipe analysis toggle)
5. Verify `generate_script(subcommand="test")` produces `python -m graphids test ...`
6. Verify `generate_script(subcommand="analyze")` produces `python -m graphids analyze ...`
