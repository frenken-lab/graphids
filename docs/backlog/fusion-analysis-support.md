# Fusion analysis support

## Problem

Fusion is excluded from `ANALYSIS_SUPPORTED_MODELS` so no `artifacts/` or `.analyze_complete` for any fusion run. Two blockers prevent a simple add:

### 1. `safe_load_checkpoint("fusion", ckpt)` always loads BanditFusionModule

`_MODULE_PATHS` in `_training.py` maps `"fusion"` → `BanditFusionModule`. Loading an MLP, DQN, or WeightedAvg checkpoint as Bandit would crash. The orchestrator uses `model_type="fusion"` for all fusion methods — the fusion_method is only in the identity hash, not the model_type.

Fix options:
- A) Add per-method entries to `_MODULE_PATHS` (`"mlp_fusion"`, `"weighted_avg"`, etc.) and pass fusion_method through the analysis spec
- B) Read `class_path` from the checkpoint's saved hyperparameters and load dynamically
- C) Store `class_path` in the AnalysisSpec and use it directly

### 2. `fusion_policy` needs upstream checkpoint paths

`run_fusion_policy()` requires `vgae_ckpt_path` and `gat_ckpt_path`. `build_analysis_spec()` doesn't receive these — it only gets the fusion checkpoint path. The upstream paths are available in the orchestrator's `StageConfig` (from resolved upstream assets) but aren't passed through.

### 3. `best_model.ckpt` never saved for RL fusion (Bandit/DQN)

Both `BanditFusionModule` and `DQNFusionModule` set `self.automatic_optimization = False`.
ModelCheckpoint silently never fires with manual optimization — `val_acc` is logged and
reaches 0.96 (Bandit on set_01) but `best_model.ckpt` is never written. MLP and WeightedAvg
use automatic optimization and save correctly.

Evidence: `bandit.py:67` sets `automatic_optimization = False`. SLURM log
`fusion_82437173_set_01_s42_46270145` shows training accuracy 97.4%, val_acc oscillating
0.42–0.96, but only `last.ckpt` exists. Test phase falls back to `last.ckpt` (epoch 49,
val_acc=0.42) → degenerate test metrics (auc=0.5).

Fix: either manually call `self.log("val_acc", ..., on_epoch=True)` in a way that
ModelCheckpoint can see, or manually call `self.trainer.save_checkpoint()` in
`on_validation_epoch_end` when using manual optimization.

## Scope

- `graphids/core/models/_training.py` — `_MODULE_PATHS`
- `graphids/core/models/fusion/bandit.py` — `automatic_optimization`, checkpoint saving
- `graphids/core/models/fusion/dqn.py` — same pattern
- `graphids/orchestrate/analysis.py` — `build_analysis_spec`, `analysis_flags_for`
- `graphids/core/contracts/analysis.py` — `AnalysisSpec` (already has the fields)
