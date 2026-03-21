# Evaluation Decomposition: All-Lightning Modules

**Date:** 2026-03-21
**Status:** Draft
**Depends on:** DataModule restructure (done), num_ids config fix (done)
**Reference:** `plans/lightning-api-reference.md` (662 lines, 26 verified sources)

---

## Context

`evaluation.py` is a 265-line monolith with 3 incompatible eval patterns:
- GAT/VGAE: `trainer.test()` via Lightning (correct)
- Fusion DQN/bandit: manual inference loop + inline torchmetrics (should be Lightning)
- Temporal: manual inference loop + inline torchmetrics (should be Lightning)

Adding a new model requires understanding all 3 patterns. The goal is one pattern: every model evaluates via `trainer.test(module, dataloaders=...)`.

Additionally, fusion DQN/bandit training uses a custom RL loop. Lightning officially supports DQN via `IterableDataset` + manual optimization (source: [Lightning DQN tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html), verified in `plans/lightning-api-reference.md` §2).

---

## Current state (from codebase inventory)

### Already Lightning
| Component | Module | Train | Eval |
|-----------|--------|-------|------|
| VGAE | `VGAEModule(LightningModule)` in `modules.py` | `training_step` → loss | `test_step` → metrics via MetricCollection |
| GAT | `GATModule(LightningModule)` in `modules.py` | `training_step` → loss | `test_step` → metrics via MetricCollection |
| Temporal | `TemporalLightningModule(LightningModule)` in `temporal.py` | `training_step` → loss | **Missing** — eval uses manual loop in `_evaluate_temporal()` |
| MLP fusion | `MLPFusionModule(LightningModule)` in `fusion_baselines.py` | `training_step` → loss | **Missing** — eval uses `_evaluate_fusion()` manual loop |
| WeightedAvg fusion | `WeightedAvgModule(LightningModule)` in `fusion_baselines.py` | `training_step` → loss | **Missing** — eval uses `_evaluate_fusion()` manual loop |

### Not Lightning
| Component | Current location | Train | Eval |
|-----------|-----------------|-------|------|
| DQN fusion | `EnhancedDQNFusionAgent` in `dqn.py` | Custom RL loop in `_train_dqn_fusion()` | Manual loop in `_evaluate_fusion()` |
| Bandit fusion | `NeuralLinUCBAgent` in `bandit.py` | Custom loop in `_train_bandit_fusion()` | Manual loop in `_evaluate_fusion()` |

---

## Design

### Principle: one eval pattern for all models

Every model gets a `test_step()`. Eval becomes:
```python
trainer.test(module, dataloaders=loader)
metrics = extract_metrics(module)
```

### Phase 1: Add test_step to existing Lightning modules that lack it

**TemporalLightningModule** — already has `training_step` and `validation_step`. Add `test_step`:

```python
# temporal.py — TemporalLightningModule
def test_step(self, batch, batch_idx):
    graph_sequences, labels = batch
    logits = self(graph_sequences)
    preds = logits.argmax(dim=1)
    self.test_metrics.update(preds, labels)
```

Source: `test_step` signature from `lightning-api-reference.md` §1 — returns None, logs via `self.log_dict()`.

**MLPFusionModule** and **WeightedAvgModule** — already have `validation_step`. Add `test_step`:

```python
# fusion_baselines.py — MLPFusionModule
def test_step(self, batch, batch_idx):
    states, labels = batch
    logits = self.model(states).squeeze(-1)
    preds = (torch.sigmoid(logits) > 0.5).long()
    self.test_metrics.update(preds, labels)
```

These modules already exist in `graphids/core/models/fusion_baselines.py`. Need to add `test_metrics = MetricCollection(...)` to `__init__` and `test_step` method.

### Phase 2: Wrap DQN and Bandit agents in LightningModules

#### DQN Training — `DQNFusionModule(LightningModule)`

The DQN agent has:
- Its own optimizer (`self.optimizer = AdamW(...)`)
- Its own replay buffer (`self._buffer = TensorReplayBuffer`)
- Multi-step gradient updates per episode (`cfg.fusion.gpu_training_steps`)
- Epsilon decay

Lightning pattern (from `lightning-api-reference.md` §2, official DQN example):

```python
class DQNFusionModule(pl.LightningModule):
    def __init__(self, cfg, agent: EnhancedDQNFusionAgent):
        super().__init__()
        self.automatic_optimization = False  # manual optimization for RL
        self.agent = agent
        self.cfg = cfg
        self.test_metrics = MetricCollection({...})

    def training_step(self, batch, batch_idx):
        states, labels = batch
        opt = self.optimizers()

        # Environment interaction
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=True)
        preds = (alphas > 0.5).long()
        rewards = self.agent.reward_calc.compute(preds, labels, norm_states, alphas)
        self.agent.store_experiences_batch(norm_states, actions, rewards)

        # Gradient steps from replay buffer
        if self.agent.buffer_size_current >= self.cfg.dqn.batch_size:
            for _ in range(self.cfg.fusion.gpu_training_steps):
                loss = self.agent.train_step()  # uses its own optimizer internally
            self.log("train_loss", loss)

        # Epsilon decay
        self.agent.epsilon = max(
            self.agent.min_epsilon,
            self.agent.epsilon * self.agent.epsilon_decay
        )

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.validate_batch(states, labels)
        self.log("val_acc", result["accuracy"])

    def test_step(self, batch, batch_idx):
        states, labels = batch
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=False)
        anomaly_scores, gat_probs = self.agent.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > 0.5).long()
        self.test_metrics.update(preds, labels)

    def configure_optimizers(self):
        # Agent manages its own optimizer; return a dummy or the agent's
        return self.agent.optimizer

    def train_dataloader(self):
        # Cached predictions as a TensorDataset
        return DataLoader(TensorDataset(self._train_states, self._train_labels),
                          batch_size=self.cfg.fusion.episode_sample_size, shuffle=True)
```

**Key decision:** The DQN agent already manages its own optimizer internally in `train_step()`. Two options:

- **Option A:** Keep agent's internal optimizer. `configure_optimizers()` returns it for Lightning's scheduler hooks, but `self.automatic_optimization = False` means Lightning won't call `.step()`. Agent calls it in `train_step()`.
- **Option B:** Extract the optimizer from the agent, pass it to Lightning. Agent's `train_step()` becomes a pure loss computation, Lightning handles backward/step.

**Option A is simpler** — minimal changes to `dqn.py`. The agent's `train_step()` already does zero_grad/backward/step/clip internally. Manual optimization mode means Lightning won't interfere.

Source: `lightning-api-reference.md` §2 — "With `automatic_optimization=False`, Lightning handles only: accelerator setup, precision scaling, strategy (DDP). User handles everything else."

#### Bandit Training — `BanditFusionModule(LightningModule)`

The bandit agent uses Sherman-Morrison updates (closed-form, no gradient descent) plus periodic backbone retraining. This maps to manual optimization:

```python
class BanditFusionModule(pl.LightningModule):
    def __init__(self, cfg, agent: NeuralLinUCBAgent):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self.test_metrics = MetricCollection({...})

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.train_episode(states, labels)
        self.log_dict({k: v for k, v in result.items() if isinstance(v, (int, float))})

    def test_step(self, batch, batch_idx):
        states, labels = batch
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=False)
        anomaly_scores, gat_probs = self.agent.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > 0.5).long()
        self.test_metrics.update(preds, labels)

    def configure_optimizers(self):
        return self.agent.backbone_optimizer
```

### Phase 3: Decompose evaluation.py

Replace the monolith with a dispatcher + per-model eval functions.

```python
# evaluation.py — after refactor

EVAL_ORDER = ["gat", "vgae", "fusion", "temporal"]

def evaluate(cfg) -> dict:
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup()
    dm.populate_config(cfg)

    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics = {}
    test_metrics = {}
    artifacts = {}

    for model_name in EVAL_ORDER:
        ckpt = cfg.checkpoints.get(model_name)
        if not ckpt or not Path(ckpt).exists():
            continue
        result = EVAL_FNS[model_name](cfg, val_data, test_scenarios, dm)
        all_metrics[model_name] = result["val_metrics"]
        if result.get("test_metrics"):
            test_metrics[model_name] = result["test_metrics"]
        if result.get("artifacts"):
            artifacts[model_name] = result["artifacts"]

    # CKA (KD runs only)
    if any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        _try_cka(cfg, val_data, dm)

    # Persist artifacts
    _save_all_artifacts(artifacts, Path.cwd())

    if test_metrics:
        all_metrics["test"] = test_metrics
    return {"metrics": all_metrics}
```

Each `eval_*` function follows the same pattern:

```python
def eval_gat(cfg, val_data, test_scenarios, dm) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, "gat", "curriculum", device)
    module = GATModule(cfg)
    module.model = model

    val_metrics = test_model(module, val_data)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tdata)

    artifacts = capture_gat_artifacts(model, val_data, device)

    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": artifacts}
```

```python
def eval_fusion(cfg, val_data, test_scenarios, dm) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", "curriculum", device)
    models = {"vgae": vgae, "gat": gat}

    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)
    frozen_cfg = load_frozen_cfg(cfg, "fusion")
    method = frozen_cfg.fusion.method if hasattr(frozen_cfg, "fusion") else cfg.fusion.method

    # Load agent + wrap in Lightning module
    agent = _load_fusion_agent(method, frozen_cfg, cfg, device)
    module = _make_fusion_eval_module(agent, method)

    # Eval via trainer.test
    loader = DataLoader(TensorDataset(val_cache["states"], val_cache["labels"]), batch_size=256)
    val_metrics = test_model(module, loader)  # unified path

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
            tl = DataLoader(TensorDataset(tc["states"], tc["labels"]), batch_size=256)
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tl)

    artifacts = run_fusion_inference(agent, val_cache)  # for dqn_policy.json

    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": artifacts}
```

**Note:** `test_model()` currently takes a list of graphs and creates its own DataLoader. For fusion, it receives a pre-built DataLoader of tensor batches. The function needs a small overload or the signature changes to accept either.

### Phase 4: Wrap DQN/Bandit training in Lightning

Replace `_train_dqn_fusion()` and `_train_bandit_fusion()` with:

```python
def _train_dqn_fusion(cfg, train_cache, val_cache, device) -> float:
    agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))
    module = DQNFusionModule(cfg, agent, train_cache, val_cache)
    trainer = _make_fusion_trainer(cfg)
    trainer.fit(module)
    _save_dqn_ckpt(agent)
    return trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
```

The episode loop moves into `DQNFusionModule.training_step()`. Lightning handles:
- Epoch counting (`self.current_epoch`)
- Validation loop (`validation_step` called automatically)
- Checkpointing (ModelCheckpoint callback)
- Logging (CSVLogger)
- GPU management
- Progress bar

---

## Files to modify/create

### Create

| File | Lines | Purpose |
|------|-------|---------|
| `graphids/pipeline/stages/eval_gat.py` | ~30 | `eval_gat()` — load model, test, capture artifacts |
| `graphids/pipeline/stages/eval_vgae.py` | ~40 | `eval_vgae()` — load model, threshold search, test, capture artifacts |
| `graphids/pipeline/stages/eval_fusion.py` | ~50 | `eval_fusion()` — cache predictions, load agent, test via Lightning |
| `graphids/pipeline/stages/eval_temporal.py` | ~30 | `eval_temporal()` — load model, group sequences, test via Lightning |

Or: keep them all in `evaluation.py` as named functions (no new files). Decision: **keep in one file** — they're small functions, and splitting 4 × 30-line functions into 4 files violates "does this file need to exist?"

### Modify

| File | Change |
|------|--------|
| `modules.py` | Add `test_metrics` + `test_step` to any module missing them (check what's already there) |
| `temporal.py` | Add `test_step` + `test_metrics` to `TemporalLightningModule` |
| `fusion_baselines.py` | Add `test_metrics` + `test_step` to `MLPFusionModule` and `WeightedAvgModule` |
| `fusion.py` | Create `DQNFusionModule`, `BanditFusionModule`. Refactor `_train_dqn_fusion` and `_train_bandit_fusion` to use them |
| `evaluation.py` | Rewrite as dispatcher + per-model eval functions |
| `eval_inference.py` | Adapt `test_model()` to accept DataLoader or list. Keep artifact capture functions |

### Delete

Nothing deleted — code moves, doesn't disappear.

---

## Execution order

1. Add `test_step` + `test_metrics` to `TemporalLightningModule` (temporal.py)
2. Add `test_step` + `test_metrics` to `MLPFusionModule` + `WeightedAvgModule` (fusion_baselines.py)
3. Create `DQNFusionModule` + `BanditFusionModule` in `fusion.py` (eval path only first)
4. Adapt `test_model()` in `eval_inference.py` to accept DataLoader
5. Rewrite `evaluation.py` as dispatcher + per-model eval functions
6. Verify all eval paths produce same metrics as before (regression test)
7. Wrap DQN training in `DQNFusionModule` (training path)
8. Wrap Bandit training in `BanditFusionModule` (training path)
9. Delete `_train_dqn_fusion()` and `_train_bandit_fusion()` standalone functions

Steps 1-6 are the eval decomposition. Steps 7-9 are the training migration — can be a separate session.

---

## Verification

1. **Import check (login node):**
   ```bash
   python -c "from graphids.pipeline.stages.evaluation import evaluate; print('OK')"
   ```

2. **Regression test:** Run eval on a completed experiment, compare metrics output before and after. Metrics must be identical (same models, same data, same threshold).

3. **Smoke test (gpudebug):** Full pipeline smoke: train autoencoder → train curriculum → eval. Verify all 4 model sections produce metrics.

---

## Sources referenced

| Claim | Source | Verified? |
|-------|--------|-----------|
| `test_step` with `dataloader_idx` for multi-DL | `lightning-api-reference.md` §1, §3 | Yes — fetched from Lightning docs |
| Manual optimization for DQN | `lightning-api-reference.md` §2 | Yes — official DQN example |
| IterableDataset as replay buffer | `lightning-api-reference.md` §2 | Yes — official DQN example |
| MetricCollection.clone(prefix=...) | `lightning-api-reference.md` §5 | Yes — torchmetrics docs |
| on_test_epoch_end fires once after all DLs | `lightning-api-reference.md` §6 | Yes — Lightning hooks docs |
| PyG DataBatch in training_step | `lightning-api-reference.md` §7 | Yes — PyG batching docs |
| TemporalLightningModule already exists | Codebase inventory: `temporal.py` | Yes — read by explore agent |
| MLPFusionModule/WeightedAvgModule exist | Codebase inventory: `fusion_baselines.py` | Yes — read by explore agent |
| DQN agent manages own optimizer | Codebase inventory: `dqn.py` | Yes — read by explore agent |
