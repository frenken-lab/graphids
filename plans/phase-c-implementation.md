# Phase C Implementation: Delete storage layer

**Date:** 2026-03-20
**Status:** Pending (revert broken attempt first)
**Parent plan:** `framework-consolidation.research.md`

---

## Pre-work: Revert broken Phase C attempt

The previous attempt deviated from the plan by introducing `stage_dir()` calls instead of using framework features. Revert all uncommitted changes back to commit `bbbfaeb` (Phase A+B done).

```bash
git checkout -- .
git clean -fd graphids/storage  # already deleted, nothing to restore
```

Then re-delete `graphids/storage/` and work through this plan file-by-file.

---

## Config changes (do first — everything depends on these)

### `config/conf/config.yaml`

**Add** OmegaConf interpolation for output paths and checkpoint locations. These replace all Python path functions for experiment storage.

```yaml
# Add after the infrastructure section:
_tier: dev/${oc.env:USER,unknown}
_output_base: ${lake_root}/${_tier}/${dataset}

checkpoints:
  vgae: ${_output_base}/vgae_${scale}_autoencoder/seed_${seed}/best_model.pt
  gat: ${_output_base}/gat_${scale}_curriculum/seed_${seed}/best_model.pt
  dqn: ${_output_base}/dqn_${scale}_fusion/seed_${seed}/best_model.pt
  temporal: ${_output_base}/gat_${scale}_temporal/seed_${seed}/best_model.pt

# Change hydra section:
hydra:
  job:
    chdir: true    # Hydra cd's to run.dir — Lightning writes to cwd
  output_subdir: null
  run:
    dir: ${_output_base}/${model_type}_${scale}_${stage}/seed_${seed}
```

**Why `chdir: true`:** Lightning's Trainer defaults to cwd for `default_root_dir`. With `chdir: true`, Hydra cd's to the output dir before calling the task function. No path passing needed — `trainer.log_dir`, `ModelCheckpoint`, `CSVLogger` all write to the right place automatically.

**Relative path concern:** `lake_root` defaults to `experimentruns` (relative). With `chdir: true`, relative paths resolve from the NEW cwd. Mitigation: on OSC, `KD_GAT_LAKE_ROOT` is always set to an absolute path (`/fs/ess/PAS1266/kd-gat`). For local dev, set the env var. Add a validation check in `__main__.py` that warns if `lake_root` is relative.

### `config/schema.py`

**Add** `checkpoints` field to PipelineConfig:

```python
# After the production field:
checkpoints: dict[str, str] = Field(default_factory=dict)
```

Dict, not Pydantic model — `cfg.checkpoints["vgae"]` works for both static and dynamic (variable model_type) access.

### `config/_hydra_bridge.py`

**Add** `_tier` and `_output_base` to `_HYDRA_ONLY_KEYS`:

```python
_HYDRA_ONLY_KEYS = frozenset({"stage", "_tier", "_output_base"})
```

### `config/paths.py`

**Keep:** `lake_cache_dir` (~5 lines), `lake_raw_dir` (~5 lines) — preprocessing cache paths are per-dataset, not per-run. Not managed by Hydra.

**Keep:** `lake_root_from_env()` — used by `__main__.py` lake subcommand.

**Keep:** `lake_catalog_path()` — used by lake subcommand (until Phase E).

**Delete:** `stage_dir()`, `checkpoint_path()`, `config_path()`, `run_id()`, `run_id_str()`, `_id_parts()`, `_config_hash()`, `sweep_result_path()`, `lake_run_dir()`. All replaced by `hydra.run.dir` template + `cfg.checkpoints` dict.

**Note:** `run_id()` is imported by `trainer_factory.py` for the persistent checkpoint root. With `chdir: true`, this is just cwd. Delete.

### `config/__init__.py`

**Remove re-exports:** `checkpoint_path`, `config_path`, `run_id`, `run_id_str`, `stage_dir`, `sweep_result_path`. Add: `lake_root_from_env`, `lake_catalog_path`.

### `graphids/__init__.py`

**Remove** `storage` from `_lazy_submodules`. Remove `checkpoint_path` from the re-export line if present.

---

## File-by-file: what to change and what framework feature replaces it

### `pipeline/executor.py`

**Current imports:** `from graphids.storage import open_gateway, write_manifest`

**What it uses storage for:**
- `gw, mapper = open_gateway(cfg)` → get gateway + mapper
- `sdir = gw.resolve(stage)` → output directory path
- `mapper.save_config(cfg, stage)` → write config.json
- `write_manifest(sdir, ...)` → write _manifest.json
- Archive/restore previous run

**Replace with:**
- Output directory: **cwd** (Hydra cd'd here). `sdir = Path(".")`.
- `mapper.save_config()`: **`save_hyperparameters()`** already saves config to checkpoint + `hparams.yaml`. For a standalone `config.json`, `cfg.save(Path("config.json"))` is one line. But consider: is `config.json` still needed? `hparams.yaml` has the same data. If nothing reads `config.json` except `load_frozen_cfg()`, update `load_frozen_cfg()` to read `hparams.yaml` instead. **Research needed:** does `save_hyperparameters()` write `hparams.yaml` to `trainer.log_dir` automatically? Yes — when CSVLogger is the logger, `trainer.log_dir` is the CSVLogger's `log_dir`, and `save_hyperparameters()` writes there.
- `write_manifest()`: **`RunMetadataCallback`** already writes `run_metadata.json` in `on_fit_end`. Delete manifest write.
- Archive/restore: **Delete.** With `chdir: true`, each run gets its own directory via `hydra.run.dir`. No need to archive — the old run is in its own dir.

**Exact changes:**
```python
# DELETE these imports:
from graphids.storage import open_gateway, write_manifest
import shutil
from datetime import datetime

# DELETE: gw, mapper = open_gateway(cfg) / sdir = gw.resolve(stage)
# DELETE: archive/restore block
# DELETE: mapper.save_config(cfg, stage)
# DELETE: write_manifest(...) block

# REPLACE sdir with:
sdir = Path.cwd()

# KEEP: validate, structlog context, timing, STAGE_FNS dispatch
# KEEP: cfg.save(sdir / "config.json") — until load_frozen_cfg is updated to use hparams.yaml
```

### `pipeline/stages/trainer_factory.py`

**Current imports:** `from graphids.storage import StorageGateway` (top-level), `from graphids.storage import ArtifactMapper` (lazy in `load_model`)

**What it uses storage for:**
1. `resolve_teacher_path()`: `StorageGateway(cfg=teacher_cfg)` → `gw.resolve(stage, "best_model.pt")` — find teacher checkpoint
2. `load_frozen_cfg()`: `StorageGateway(cfg=cfg)` → `gw.require(stage, "config.json", model_type=mt)` — find frozen config
3. `load_model()`: `StorageGateway(cfg=cfg)` + `ArtifactMapper(gw)` → `mapper.load_checkpoint(stage, model_type=mt)` — load checkpoint
4. `make_trainer()`: `StorageGateway(cfg=cfg)` → `gw.ensure_dir(stage)` — create output dir + paths for CSVLogger/ModelCheckpoint

**Replace with:**
1. Teacher checkpoint: `Path(cfg.checkpoints[model_type])`. But wait — teacher uses a DIFFERENT scale (teacher_scale, usually "large"). The `cfg.checkpoints` dict uses `${scale}` from the current config, which might be "small" for the student. **Fix:** resolve a separate teacher config: `teacher_cfg = resolve(model_type, teacher_scale, ...)`, then `teacher_cfg.checkpoints[model_type]`.
2. Frozen config: `Path(cfg.checkpoints[mt]).parent / "config.json"`. The config.json is in the same directory as the checkpoint.
3. Load checkpoint: `torch.load(cfg.checkpoints[model_type], map_location="cpu", weights_only=True)`. Direct stdlib call.
4. Output dir: **cwd** (`Path(".")`). With `chdir: true`, the trainer writes to cwd. No `ensure_dir` needed — Hydra created the dir.

**Exact changes:**
```python
# DELETE:
from graphids.storage import StorageGateway
from graphids.config import run_id, stage_dir  # if present

# In resolve_teacher_path():
# REPLACE: gw = StorageGateway(cfg=teacher_cfg); path = gw.resolve(stage, "best_model.pt")
# WITH:    path = Path(teacher_cfg.checkpoints[model_type])

# In load_frozen_cfg():
# REPLACE: gw = StorageGateway(cfg=cfg); p = gw.require(stage, "config.json", model_type=mt)
# WITH:    p = Path(cfg.checkpoints[mt]).parent / "config.json"
#          if not p.exists(): raise FileNotFoundError(...)

# In load_model():
# DELETE:  from graphids.storage import ArtifactMapper
# DELETE:  gw = StorageGateway(cfg=cfg); mapper = ArtifactMapper(gw)
# REPLACE: model.load_state_dict(mapper.load_checkpoint(stage, model_type=model_type))
# WITH:    model.load_state_dict(torch.load(cfg.checkpoints[model_type], map_location="cpu", weights_only=True))

# In make_trainer():
# DELETE:  gw = StorageGateway(cfg=cfg); out = gw.ensure_dir(stage)
# REPLACE: out = Path.cwd()  # Hydra cd'd here
# DELETE:  persistent_root / run_id lines (cwd IS the persistent root)
# UPDATE:  CSVLogger(save_dir=".", name="", version="")
# UPDATE:  ModelCheckpoint(dirpath=".", ...)
```

### `pipeline/stages/training.py`

**Current imports:** `from graphids.storage import open_gateway`

**What it uses storage for:**
- `_save_and_cleanup()`: `_, mapper = open_gateway(cfg)` → `mapper.save_training_result(module.model, cfg, stage, trainer)`

**Replace with:** **Nothing.** ModelCheckpoint already saved the model. `save_hyperparameters()` saved the config. CSVLogger saved the metrics. `_save_and_cleanup()` becomes:

```python
def _save_and_cleanup(module, trainer, cfg, stage, label=None):
    ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "")
    metrics = {}
    if trainer.callback_metrics:
        metrics = {k: v.item() if hasattr(v, "item") else v
                   for k, v in trainer.callback_metrics.items()}
    log.info("training_complete", label=label or stage, checkpoint=ckpt)
    cleanup()
    return {"checkpoint": ckpt, "metrics": metrics}
```

**Also delete:** `from graphids.config import run_id` if only used for `_resume_ckpt_path`. Update `_resume_ckpt_path` to use cwd:

```python
# REPLACE: persistent_root = Path(cfg.lake_root) / run_id(cfg, stage)
# WITH:    persistent_root = Path.cwd()
auto_save = persistent_root / ".pl_auto_save.ckpt"
```

### `pipeline/stages/evaluation.py`

**Current imports:** `from graphids.storage import StorageGateway, open_gateway`

**What it uses storage for:**
1. `gw.exists(stage, "best_model.pt", model_type=X)` — check if model checkpoints exist (5 calls)
2. `gw.ensure_dir("evaluation")` — create eval output dir
3. `mapper.save_cka(...)` — CKA computation + save
4. `mapper.save_embeddings(...)` — save embeddings.npz
5. `mapper.save_attention(...)` — save attention_weights.npz
6. `mapper.save_dqn_policy(...)` — save dqn_policy.json
7. `mapper.load_checkpoint(stage, model_type=X)` — load fusion/temporal checkpoints (2 calls)

**Replace with:**
1. Existence checks: `Path(cfg.checkpoints["gat"]).exists()`. Direct.
2. Eval output dir: `Path.cwd()` — Hydra cd'd here when stage=evaluation.
3. CKA: move computation to `cka.py`, I/O is `np.savez_compressed()` to cwd.
4-6. Embeddings/attention/DQN policy: **`EvalArtifactCallback`** (created in Phase A). Wire it: set `cb.gat_result`, `cb.vgae_result`, `cb.fusion_result`, call save methods with `Path.cwd()`.
7. Load checkpoints: `torch.load(cfg.checkpoints["dqn"], ...)`. Direct.

**Exact changes:**
```python
# DELETE:
from graphids.storage import StorageGateway, open_gateway

# REPLACE existence checks:
# OLD: gw.exists("fusion", "best_model.pt", model_type="dqn")
# NEW: Path(cfg.checkpoints["dqn"]).exists()

# REPLACE eval output dir:
# OLD: gw.ensure_dir("evaluation")
# NEW: (not needed — Hydra created cwd)

# REPLACE artifact saves:
# OLD: mapper.save_embeddings(gat_result, vgae_result, "evaluation")
#      mapper.save_attention(gat_result, "evaluation")
#      mapper.save_dqn_policy(fusion_result, "evaluation")
# NEW: from .callbacks import EvalArtifactCallback
#      cb = EvalArtifactCallback()
#      cb.gat_result = gat_result
#      cb.vgae_result = vgae_result
#      cb.fusion_result = fusion_result
#      cb._save_embeddings(Path.cwd())
#      cb._save_attention(Path.cwd())
#      cb._save_dqn_policy(Path.cwd())

# REPLACE CKA:
# OLD: mapper.save_cka(cfg, val_data, device, num_ids, in_ch, "evaluation")
# NEW: from .cka import compute_and_save_cka
#      compute_and_save_cka(cfg, val_data, device, num_ids, in_ch, Path.cwd())

# REPLACE checkpoint loads:
# OLD: mapper_f.load_checkpoint("fusion", model_type="dqn")
# NEW: torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True)
```

### `pipeline/stages/fusion.py`

**Current imports:** `from graphids.storage import open_gateway`

**What it uses storage for:**
1. `gw.ensure_dir("fusion")` — create output dir
2. `gw.resolve("fusion")` — get output path
3. `gw.exists("fusion", "best_model.pt")` — check checkpoint exists
4. `mapper.save_dqn_checkpoint(...)` — save DQN checkpoint
5. `mapper.save_checkpoint(...)` — save MLP/WeightedAvg checkpoint
6. `mapper.save_config(cfg, "fusion")` — save config.json

**Replace with:**
1-2. Output dir: `Path.cwd()` — Hydra cd'd here.
3. Exists: `Path("best_model.pt").exists()` — checking in cwd.
4-5. Save checkpoint: `torch.save(state_dict, "best_model.pt")` — direct to cwd.
6. Config: `cfg.save(Path("config.json"))` — or rely on `save_hyperparameters()`.

**Note:** DQN fusion doesn't use Lightning Trainer (custom RL loop). So `save_hyperparameters()` / ModelCheckpoint don't apply. Direct `torch.save()` is correct here.

**Exact changes:**
```python
# DELETE:
from graphids.storage import open_gateway

# ADD helper for DQN checkpoint save:
def _save_dqn_ckpt(agent) -> None:
    torch.save({
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
    }, "best_model.pt")

# In _train_dqn_fusion():
# REPLACE: mapper.save_dqn_checkpoint({...}, "fusion")
# WITH:    _save_dqn_ckpt(agent)
# REPLACE: gw.exists("fusion", "best_model.pt")
# WITH:    Path("best_model.pt").exists()

# In _make_fusion_trainer():
# REPLACE: gw, _ = open_gateway(cfg); out = gw.ensure_dir("fusion")
# WITH:    (delete — Hydra created cwd)
# UPDATE:  pl.Trainer(default_root_dir=".", ...)
# UPDATE:  ModelCheckpoint(dirpath=".", ...)
# UPDATE:  CSVLogger(save_dir=".", name="", version="")

# In _train_mlp_fusion() / _train_weighted_avg_fusion():
# REPLACE: mapper.save_checkpoint({...}, "fusion")
# WITH:    torch.save({...}, "best_model.pt")

# In train_fusion():
# DELETE:  gw, mapper = open_gateway(cfg); gw.ensure_dir("fusion")
# REPLACE: gw.resolve("fusion", "best_model.pt")
# WITH:    Path("best_model.pt")
# REPLACE: mapper.save_config(cfg, "fusion")
# WITH:    cfg.save(Path("config.json"))
```

### `pipeline/stages/temporal.py`

**Current imports:** `from graphids.storage import open_gateway`

**What it uses storage for:**
- `gw.ensure_dir("temporal")` — create output dir
- `mapper.save_config(cfg, "temporal")` — save config
- `mapper.save_checkpoint(temporal_model.state_dict(), "temporal")` — save model

**Replace with:**
- Output dir: `Path.cwd()` — Hydra cd'd here.
- Config: `cfg.save(Path("config.json"))`.
- Checkpoint: `torch.save(temporal_model.state_dict(), "best_model.pt")`.

**Exact changes:**
```python
# DELETE:
from graphids.storage import open_gateway

# DELETE: gw, mapper = open_gateway(cfg); gw.ensure_dir("temporal")
# REPLACE: mapper.save_config(cfg, "temporal")
# WITH:    cfg.save(Path("config.json"))
# REPLACE: best_ckpt = mapper.save_checkpoint(temporal_model.state_dict(), "temporal")
# WITH:    torch.save(temporal_model.state_dict(), "best_model.pt")
#          best_ckpt = Path("best_model.pt")
```

### `pipeline/stages/data_loading.py`

**Current imports:** `from graphids.storage import StorageGateway` (lazy, inside `compute_node_budget`)

**What it uses storage for:**
- `StorageGateway(cfg=cfg)` → `gw.read_json(metadata_path)` — read cache_metadata.json

**Replace with:** `json.loads(metadata_path.read_text())`. Direct stdlib.

**Exact changes:**
```python
# In compute_node_budget():
# DELETE: from graphids.storage import StorageGateway
# DELETE: gw = StorageGateway(cfg=cfg)
# REPLACE: meta = gw.read_json(metadata_path)
# WITH:    import json
#          meta = json.loads(metadata_path.read_text())
```

### `pipeline/validate.py`

**Current imports:** `from graphids.storage import StorageGateway` (lazy, inside `_artifact_exists`)

**What it uses storage for:**
- `StorageGateway(cfg=cfg)` → `gw.exists(stage, name, model_type=model_type)` — check artifacts exist

**Replace with:** `Path(cfg.checkpoints[model_type]).exists()`.

**Exact changes:**
```python
# DELETE entire _artifact_exists() function

# REPLACE calls to _artifact_exists(cfg, stage, name, model_type):
# WITH:    Path(cfg.checkpoints[model_type]).exists()
```

### `core/preprocessing/_cache.py`

**Current imports:** `from graphids.storage import StorageGateway, ArtifactMapper` (lazy, 2 locations)

**What it uses storage for:**
- `StorageGateway(lake_root=".", dataset="cache", ...)` — dummy gateway just for `lock()` and mapper helpers
- `gw.lock(cache_file.parent)` — NFS advisory locking
- `mapper.save_collated(graphs, cache_file)` — atomic collated tensor save
- `mapper.save_pickle(id_mapping, id_mapping_file)` — atomic pickle save

**Replace with:** Inline the I/O. The `lock()` is `fcntl.flock`. The saves are `torch.save` + `pickle.dump` with `os.fsync` + `os.rename` for NFS safety.

**Exact changes:**
```python
# DELETE: from graphids.storage import StorageGateway, ArtifactMapper
# DELETE: gw = StorageGateway(lake_root=".", ...); mapper = ArtifactMapper(gw)

# REPLACE gw.lock(path) with:
import fcntl
lock_path = cache_file.parent / ".lock"
with open(lock_path, "w") as lock_fd:
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    # ... save operations ...

# REPLACE mapper.save_collated(graphs, cache_file) with:
from torch_geometric.data import InMemoryDataset
tmp = cache_file.with_suffix(".tmp")
torch.save(InMemoryDataset.collate(graphs), tmp)
os.fsync(open(tmp, "rb").fileno())  # NFS safety
tmp.rename(cache_file)

# REPLACE mapper.save_pickle(obj, path) with:
import pickle
tmp = path.with_suffix(".tmp")
with open(tmp, "wb") as f:
    pickle.dump(obj, f)
    f.flush()
    os.fsync(f.fileno())
tmp.rename(path)
```

**Note:** Check the existing `save_collated` implementation in `mapper.py` (now deleted) for the exact collation call. It may use `InMemoryDataset.collate` or a custom approach.

### `core/preprocessing/_cache_metadata.py`

**Current imports:** `from graphids.storage import StorageGateway` (lazy)

**What it uses storage for:**
- `StorageGateway(lake_root=".", ...)` → `gw.write_json(metadata_file, metadata)` — atomic JSON write

**Replace with:** `json.dump` with fsync for NFS safety:

```python
# DELETE: from graphids.storage import StorageGateway
# DELETE: gw = StorageGateway(lake_root=".", ...)

# REPLACE: gw.write_json(metadata_file, metadata)
# WITH:
import json
tmp = metadata_file.with_suffix(".tmp")
with open(tmp, "w") as f:
    json.dump(metadata, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
tmp.rename(metadata_file)
```

### `__main__.py`

**Current imports:** `from graphids.storage.paths import lake_catalog_path, lake_root_from_env` (lazy, in `_lake()`)

**Already partially fixed** — lake subcommand was stubbed with a warning. Keep the stub until Phase E when catalog is rebuilt on metrics.csv.

**Exact changes:**
```python
# REPLACE: from graphids.storage.paths import lake_catalog_path, lake_root_from_env
# WITH:    from graphids.config.paths import lake_root_from_env
# (lake_catalog_path not needed until Phase E)
```

---

## `chdir: true` impact analysis

When `hydra.job.chdir: true`, Hydra cd's to `hydra.run.dir` before calling the task function. All relative paths resolve from the new cwd.

**Safe — these use absolute paths:**
- `cfg.lake_root` when `KD_GAT_LAKE_ROOT` is set (always on OSC: `/fs/ess/PAS1266/kd-gat`)
- `cfg.checkpoints.*` — resolved from `_output_base` which uses `lake_root`
- `cache_dir(cfg)` — returns `{lake_root}/cache/...` (absolute if lake_root is)
- `data_dir(cfg)` — returns `{lake_root}/raw/...` or `data/automotive/...`

**Breaks if `lake_root` is relative (`experimentruns`):**
- `data_dir(cfg)` fallback: `Path("data") / "automotive" / cfg.dataset` — relative to cwd, which is now the run dir, not the project root
- Fix: `data_dir()` should resolve relative to `PROJECT_ROOT` (from constants.py), not cwd

**Breaks in Compose API path (tests, notebooks):**
- `chdir: true` only applies to `@hydra.main`. Compose API doesn't chdir.
- `Path.cwd()` in `executor.py` etc. would be wrong when called via Compose API.
- Fix: `executor.py` needs a fallback. If `HydraConfig` is not available (Compose path), compute the output dir from config fields. This is the ONE place where a path computation stays — but it's a 3-line fallback, not a layer.

```python
try:
    sdir = Path.cwd()  # @hydra.main path — Hydra cd'd here
    assert sdir != Path.home()  # Sanity check — cwd should be run dir, not ~
except (AssertionError, Exception):
    # Compose API fallback — compute from config
    from graphids.config.paths import lake_run_dir
    sdir = lake_run_dir(cfg.lake_root, cfg.dataset, cfg.model_type, cfg.scale, stage,
                         aux=cfg.auxiliaries[0].type if cfg.auxiliaries else "",
                         seed=cfg.seed, production=cfg.production)
    sdir.mkdir(parents=True, exist_ok=True)
```

---

## Execution order

1. Config changes (config.yaml, schema.py, _hydra_bridge.py, paths.py, __init__.py)
2. `executor.py` — central dispatcher, everything flows through here
3. `trainer_factory.py` — make_trainer + model loading
4. `training.py` — simplest, just remove save_training_result
5. `fusion.py` — DQN uses direct torch.save, MLP/WeightedAvg use Trainer
6. `temporal.py` — same pattern as training
7. `evaluation.py` — most complex (existence checks + artifact saves + checkpoint loads)
8. `validate.py` — existence checks only
9. `data_loading.py` — json.load only
10. `core/preprocessing/_cache.py` — inline lock + atomic save
11. `core/preprocessing/_cache_metadata.py` — inline atomic json write

## Verification

After all files are changed:

1. `python -c "import graphids"` succeeds
2. `grep -r "from graphids.storage" graphids/` returns nothing
3. `grep -r "StorageGateway\|ArtifactMapper\|open_gateway\|write_manifest" graphids/` returns nothing
4. `ls graphids/storage/` fails (directory doesn't exist)
5. Net lines are negative
