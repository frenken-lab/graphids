# PyTorch Lightning API Reference for KD-GAT Pipeline Redesign

Compiled 2026-03-21. All content fetched from official docs (URLs cited inline).

---

## 1. LightningModule Core Methods

Source: [LightningModule API](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

### training_step

```python
LightningModule.training_step(*args, **kwargs) -> Union[Tensor, Mapping[str, Any], None]
```

- **Return type**: A loss `Tensor`, a dict containing a `'loss'` key, or `None` (skip batch).
- **What Lightning does with the return**: In automatic optimization, Lightning uses the returned loss inside a closure: calls `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
- **Returning None**: Skips to the next batch. Not supported for multi-GPU, TPU, or DeepSpeed.

### validation_step

```python
LightningModule.validation_step(*args, **kwargs) -> Union[Tensor, Mapping[str, Any], None]
```

With multiple val dataloaders, signature includes `dataloader_idx=0`.

### test_step

```python
LightningModule.test_step(*args, **kwargs) -> Union[Tensor, Mapping[str, Any], None]
```

With multiple test dataloaders, signature includes `dataloader_idx=0`:

```python
def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
    ...
```

### predict_step

```python
LightningModule.predict_step(*args, **kwargs) -> Any
```

- By default runs `self.forward()`. Override to customize.
- Includes `dataloader_idx=0` with multiple dataloaders.
- Return value is collected and returned by `trainer.predict()`.

### configure_optimizers

```python
LightningModule.configure_optimizers() -> Union[
    Optimizer,
    Sequence[Optimizer],
    tuple[Sequence[Optimizer], Sequence[Union[LRScheduler, ReduceLROnPlateau, LRSchedulerConfig]]],
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
    Sequence[OptimizerConfig],
    Sequence[OptimizerLRSchedulerConfig],
    None
]
```

6 supported return patterns:
1. Single optimizer
2. List/tuple of optimizers (for manual optimization / GAN)
3. Two lists: `[optimizers], [schedulers]`
4. Dict with `'optimizer'` and optional `'lr_scheduler'` keys
5. Sequence of such dicts
6. `None` (no optimization)

LR scheduler config dict keys: `scheduler`, `interval` ("step" or "epoch"), `frequency`, `monitor`, `strict`, `name`.

### self.log() and self.log_dict()

Source: [Logging docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)

```python
self.log(
    name: str,
    value,
    on_step: Optional[bool] = None,   # auto-determined by hook
    on_epoch: Optional[bool] = None,   # auto-determined by hook
    prog_bar: bool = False,
    logger: bool = True,
    reduce_fx = torch.mean,
    enable_graph: bool = False,
    sync_dist: bool = False,
    add_dataloader_idx: bool = True,
    batch_size: Optional[int] = None,
    rank_zero_only: bool = False,
)
```

**Default behavior by hook:**

| Hook | on_step default | on_epoch default |
|------|-----------------|------------------|
| training_step / training batch hooks | True | False |
| training epoch hooks | False | True |
| before/after backward hooks | True | False |
| validation_step / validation batch hooks | False | True |
| validation epoch hooks | False | True |
| test_step / test hooks | False | True |

**Key constraint**: Setting both `on_step=True` and `on_epoch=True` creates two keys per metric with suffixes `_step` and `_epoch`.

`self.log_dict(dictionary, **kwargs)` — same parameters as `self.log()`, applied to all entries. "Everything explained below applies to both log() or log_dict() methods."

**add_dataloader_idx**: When True (default) and using multiple dataloaders, metrics are suffixed with `/dataloader_idx_0`, `/dataloader_idx_1`, etc. Set to False for custom naming.

### automatic_optimization = False

Source: [Manual Optimization docs](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html)

What Lightning still handles:
- Accelerator logic (GPU placement)
- Precision (fp16/bf16 scaling)
- Strategy logic (DDP sync)

What Lightning stops handling:
- `optimizer.zero_grad()` — you call it
- `loss.backward()` — you call `self.manual_backward(loss)` instead
- `optimizer.step()` — you call it
- Gradient clipping — use `self.clip_gradients()` manually
- LR scheduler stepping — call `sch.step()` manually
- `lr_scheduler_config` keys (`"frequency"`, `"interval"`) are **ignored**

---

## 2. Manual Optimization for RL (DQN)

Source: [Lightning DQN Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html)

### ReplayBuffer + IterableDataset Pattern

```python
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Wraps replay buffer as an IterableDataset for Lightning DataLoader."""
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
```

### DQNLightning Module (key methods only)

```python
class DQNLightning(LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.save_hyperparameters()
        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)
        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.populate(self.hparams.warm_start_steps)  # fill buffer with random steps

    def get_epsilon(self, start, end, frames) -> float:
        """Epsilon decay using self.global_step."""
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch, nb_batch):
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # Step environment (side effect: updates replay buffer)
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # Compute loss from replay buffer batch
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Target network sync every N steps
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({"reward": reward, "train_loss": loss})
        self.log("total_reward", self.total_reward, prog_bar=True)
        return loss  # automatic optimization handles backward + step

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        return DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
```

**Key patterns from the official DQN example:**
- Uses **automatic** optimization (not manual) — returns loss, Lightning handles backward
- `self.global_step` drives epsilon decay
- Environment step happens **inside** training_step (side effect that updates buffer)
- IterableDataset wraps replay buffer — DataLoader samples from it
- Target network sync uses `self.global_step % sync_rate == 0`
- No multiple gradient steps per training_step in this example

### Manual Optimization Pattern (from GAN example)

Source: [Manual Optimization docs](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html)

```python
class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()

    # Multiple optimizers:
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        # ... discriminator step with d_opt ...
        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()
        # ... generator step with g_opt ...
        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()
        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt
```

**Constraint**: Call `optimizer.zero_grad()` **before** `self.manual_backward(loss)` — wrong order can prevent convergence.

Available helpers in manual mode:
- `self.optimizers()` — returns one or list of `LightningOptimizer`
- `self.manual_backward(loss)` — replaces `loss.backward()`
- `self.clip_gradients(opt, gradient_clip_val=..., gradient_clip_algorithm=...)`
- `self.toggle_optimizer(opt)` / `self.untoggle_optimizer(opt)` — freeze/unfreeze param groups
- `self.lr_schedulers()` — returns one or list; call `sch.step()` manually

---

## 3. Trainer.test() with Multiple Dataloaders

Source: [Evaluation (intermediate)](https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html)

### How it iterates

When `test_dataloader()` returns a list, Lightning iterates through **each dataloader sequentially** (not interleaved). All batches from dataloader 0 are processed, then all from dataloader 1, etc.

### dataloader_idx flow

```python
def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('test_loss', loss)  # auto-suffixed with /dataloader_idx_N
    return loss
```

**The `dataloader_idx` parameter is required** when using multiple test dataloaders.

### Per-dataloader metric aggregation

**Option A — Automatic suffixing (default):**
```python
def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
    self.log('test_loss', loss, add_dataloader_idx=True)  # default
    self.log('test_acc', acc, add_dataloader_idx=True)
    # Creates: test_loss/dataloader_idx_0, test_loss/dataloader_idx_1, ...
```

**Option B — Custom naming:**
```python
def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
    names = {0: "clean", 1: "noisy", 2: "adversarial"}
    name = names[dataloader_idx]
    self.log(f'test_loss_{name}', loss, add_dataloader_idx=False)
    self.log(f'test_acc_{name}', acc, add_dataloader_idx=False)
```

**Option C — Manual collection + on_test_epoch_end:**
```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.test_outputs = {}

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {'predictions': [], 'targets': []}
        self.test_outputs[dataloader_idx]['predictions'].append(y_hat)
        self.test_outputs[dataloader_idx]['targets'].append(y)
        return loss

    def on_test_epoch_end(self):
        for idx, outputs in self.test_outputs.items():
            preds = torch.cat(outputs['predictions'], dim=0)
            targets = torch.cat(outputs['targets'], dim=0)
            acc = (preds.argmax(dim=1) == targets).float().mean()
            self.log(f'test_acc_dl{idx}', acc)
        self.test_outputs.clear()
```

### Retrieving results

```python
results = trainer.test(model, dataloaders=[dl1, dl2, dl3])
# results is a list of dicts, one per dataloader
for i, result in enumerate(results):
    print(f"Dataloader {i}: {result}")
```

### Accessing datamodule from inside module

```python
self.trainer.datamodule  # access the LightningDataModule
self.trainer.test_dataloaders  # list of test DataLoaders (available during test)
```

---

## 4. LightningDataModule

Source: [LightningDataModule docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)

### setup(stage) — valid stages

```python
def setup(self, stage: str):
    # stage is one of: "fit", "validate", "test", "predict"
    if stage == "fit":
        full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.train_ds, self.val_ds = random_split(full, [55000, 5000])
    if stage == "test":
        self.test_ds = MNIST(self.data_dir, train=False, transform=self.transform)
    if stage == "predict":
        self.predict_ds = MNIST(self.data_dir, train=False, transform=self.transform)
```

`setup()` is called on **all processes** after `prepare_data()` completes. `prepare_data()` runs on a **single process** only (for downloads).

### test_dataloader returning a list

```python
def test_dataloader(self):
    return [
        DataLoader(self.clean_test, batch_size=32),
        DataLoader(self.noisy_test, batch_size=32),
        DataLoader(self.adversarial_test, batch_size=32),
    ]
```

This maps directly to `dataloader_idx` in `test_step`: index 0 = first loader, etc.

### Lifecycle order

1. `prepare_data()` — single process, CPU
2. `setup(stage)` — all processes
3. `{train,val,test,predict}_dataloader()` — creates loaders
4. Batch transfer hooks: `on_before_batch_transfer()` -> `transfer_batch_to_device()` -> `on_after_batch_transfer()`
5. `teardown(stage)` — cleanup

### Usage

```python
dm = MyDataModule()
model = MyModel()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)       # reuses model from fit
trainer.predict(datamodule=dm)
```

---

## 5. MetricCollection (torchmetrics)

Source: [torchmetrics overview](https://github.com/lightning-ai/torchmetrics/blob/master/docs/source/pages/overview.rst) and [torchmetrics Lightning integration](https://github.com/lightning-ai/torchmetrics/blob/master/docs/source/pages/lightning.rst)

### Creating a MetricCollection

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

metrics = MetricCollection([
    MulticlassAccuracy(num_classes=N),
    MulticlassPrecision(num_classes=N),
    MulticlassRecall(num_classes=N),
])
```

Or with a dict for custom names:

```python
metrics = MetricCollection({
    'accuracy': MulticlassAccuracy(num_classes=3, average='micro'),
    'precision_macro': MulticlassPrecision(num_classes=3, average='macro'),
    'recall_macro': MulticlassRecall(num_classes=3, average='macro'),
    'f1_weighted': MulticlassF1Score(num_classes=3, average='weighted'),
})
```

### clone(prefix=...) for train/val/test splits

```python
class MyModule(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes),
            MulticlassPrecision(num_classes),
            MulticlassRecall(num_classes),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
```

**Key**: `clone()` creates deep copies with independent state. The `prefix` is prepended to all metric names in the returned dict.

### clone(prefix=...) for per-dataloader metrics

For multiple test dataloaders, clone once per dataloader:

```python
self.test_metrics = nn.ModuleList([
    metrics.clone(prefix=f'test_{name}_')
    for name in ["clean", "noisy", "adversarial"]
])
```

### update / compute / reset lifecycle

```python
# In training_step — calling the collection directly updates + returns batch values
batch_values = self.train_metrics(preds, targets)  # equivalent to update() + compute() on batch
self.log_dict(batch_values)

# In validation_step — update only, compute at epoch end
self.valid_metrics.update(logits, y)

# In on_validation_epoch_end — compute accumulated, log, reset
output = self.valid_metrics.compute()
self.log_dict(output)
self.valid_metrics.reset()
```

**Constraint**: You must call `reset()` between epochs, otherwise metrics accumulate across epochs. When using `self.log()` with torchmetrics `Metric` objects directly (not MetricCollection), Lightning handles `compute()` and `reset()` automatically. With MetricCollection + `log_dict`, you manage the lifecycle manually.

---

## 6. Callbacks

Source: [Callback API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html), [Hooks](https://lightning.ai/docs/pytorch/stable/common/hooks.html)

### Callback method signatures

All callback hooks have the same base signature:

```python
def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
```

Batch-level hooks add `outputs`, `batch`, `batch_idx`, `dataloader_idx`:

```python
def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None: ...
```

### Execution order — Training (trainer.fit())

1. `on_fit_start()` — callbacks, then LightningModule
2. Sanity check validation (if enabled)
3. `on_train_start()`
4. Per epoch:
   - `on_train_epoch_start()`
   - Per batch: `on_train_batch_start()` -> `training_step()` -> backward -> optimizer step -> `on_train_batch_end()`
   - `on_train_epoch_end()` — non-monitoring callbacks, then LightningModule, then monitoring callbacks
5. `on_train_end()`
6. **`on_fit_end()`** — fires during teardown, after on_train_end

### Execution order — Testing (trainer.test())

1. `on_test_start()`
2. `on_test_epoch_start()`
3. Per batch: `on_test_batch_start()` -> `test_step()` -> `on_test_batch_end()`
4. **`on_test_epoch_end()`** — fires after ALL test batches (across all dataloaders)
5. `on_test_end()`

**Gotcha**: `on_test_epoch_end` fires **once** after all dataloaders have been iterated, not once per dataloader. If you need per-dataloader aggregation, do it inside `test_step` or track outputs manually.

### When to use on_fit_end for artifact saving

```python
class ArtifactSaver(L.Callback):
    def on_fit_end(self, trainer, pl_module):
        # Save embeddings, attention weights, config, etc.
        # trainer.checkpoint_callback.best_model_path is available here
        # pl_module is the trained model
        save_artifacts(pl_module, trainer.log_dir)
```

---

## 7. PyG + Lightning Integration

Source: [PyG Introduction](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html), [PyG Advanced Batching](https://github.com/pyg-team/pytorch_geometric/blob/master/docs/source/advanced/batching.rst), [LightningDataset API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.lightning.LightningDataset.html)

### PyG DataLoader creates a Batch object, not stacked tensors

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch is a DataBatch, NOT a tuple of stacked tensors
    # batch.x       — [total_nodes, num_features] (concatenated, not stacked)
    # batch.edge_index — [2, total_edges] (indices offset per graph)
    # batch.y       — [num_graphs] (graph-level labels, concatenated)
    # batch.batch   — [total_nodes] (maps each node -> its graph index)
    # batch.num_graphs — number of graphs in the batch
    pass
```

Example output:
```python
DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])
batch.num_graphs  # 32
```

### The batch vector

`batch.batch` is a column vector mapping each node to its graph:
```
batch.batch = [0, 0, ..., 0, 1, 1, ..., 1, ..., 31, 31, ..., 31]
```

Used for graph-level pooling:
```python
from torch_geometric.nn import global_mean_pool
x = global_mean_pool(data.x, data.batch)  # [num_graphs, num_features]
```

### Edge index adjustment

During batching, `edge_index` tensors are **offset by the cumulative node count** from preceding graphs. The DataLoader creates a sparse block-diagonal adjacency matrix. This is handled automatically — no manual adjustment needed.

### How this differs from standard PyTorch batching

| Aspect | Standard PyTorch | PyG |
|--------|-----------------|-----|
| Batch creation | Stack tensors (all same shape) | Concatenate (variable shape per graph) |
| Feature tensor | `[batch, features]` | `[total_nodes, features]` — need `batch` vector to separate |
| Edge connectivity | N/A | `edge_index` offset automatically |
| Graph identity | Implicit (batch dim) | Explicit via `batch.batch` vector |

### Gotchas for Lightning integration

1. **`batch` in training_step is a `DataBatch` object**, not a tuple `(x, y)`. Access attributes directly: `batch.x`, `batch.y`, `batch.edge_index`, `batch.batch`.

2. **PyG's DataLoader is a subclass of `torch.utils.data.DataLoader`** — it accepts all standard args (`num_workers`, `shuffle`, `pin_memory`, etc.).

3. **`follow_batch`** parameter: If you need per-attribute batch vectors (e.g., for heterogeneous data), pass `follow_batch=['x_s', 'x_t']` to DataLoader.

4. **LightningDataset** wraps PyG datasets into a LightningDataModule:
   ```python
   from torch_geometric.data.lightning import LightningDataset

   dm = LightningDataset(
       train_dataset=train_ds,
       val_dataset=val_ds,
       test_dataset=test_ds,
       batch_size=32,
       num_workers=4,
   )
   trainer.fit(model, datamodule=dm)
   ```
   Only supports `SingleDeviceStrategy` and `DDPStrategy`.

5. **No special integration needed** for basic use. Just use `torch_geometric.loader.DataLoader` in your `train_dataloader()` / `test_dataloader()` methods and handle the `DataBatch` object in your steps. Lightning doesn't interfere with PyG's collation.

---

## Verification Table

| # | Topic | Source URL | Fetched and read? | Method |
|---|-------|-----------|-------------------|--------|
| 1a | training_step signature + return | https://lightning.ai/docs/pytorch/stable/common/lightning_module.html | yes | web fetch |
| 1b | test_step, predict_step signatures | https://lightning.ai/docs/pytorch/stable/common/lightning_module.html | yes | web fetch |
| 1c | configure_optimizers return types | https://lightning.ai/docs/pytorch/stable/common/optimization.html | yes | web fetch |
| 1d | self.log / self.log_dict defaults | https://lightning.ai/docs/pytorch/stable/extensions/logging.html | yes | web fetch |
| 1e | automatic_optimization = False | https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html (via context7 /lightning-ai/pytorch-lightning) | yes | context7 + web fetch |
| 2a | Manual optimization pattern | https://github.com/lightning-ai/pytorch-lightning/blob/master/docs/source-pytorch/model/manual_optimization.rst | yes | context7 |
| 2b | DQN example (full code) | https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html | yes | web fetch |
| 2c | IterableDataset as replay buffer | https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html | yes | web fetch |
| 2d | Epsilon decay via global_step | https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html | yes | web fetch |
| 2e | GAN multi-optimizer manual pattern | https://github.com/lightning-ai/pytorch-lightning/blob/master/docs/source-pytorch/model/manual_optimization.rst | yes | context7 |
| 3a | trainer.test() with multiple DLs | https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html | yes | web fetch |
| 3b | dataloader_idx in test_step | https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html | yes | web fetch |
| 3c | Per-dataloader metric aggregation | https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html | yes | web fetch |
| 3d | trainer.datamodule access | https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html | yes | web fetch |
| 4a | test_dataloader returning list | https://lightning.ai/docs/pytorch/stable/data/datamodule.html | yes | web fetch |
| 4b | setup(stage) valid stages | https://lightning.ai/docs/pytorch/stable/data/datamodule.html | yes | web fetch |
| 5a | MetricCollection API | https://github.com/lightning-ai/torchmetrics/blob/master/docs/source/pages/overview.rst | yes | context7 |
| 5b | clone(prefix=...) pattern | https://github.com/lightning-ai/torchmetrics/blob/master/docs/source/pages/lightning.rst | yes | context7 |
| 5c | update/compute/reset lifecycle | https://github.com/lightning-ai/torchmetrics/blob/master/docs/source/pages/lightning.rst | yes | context7 |
| 6a | Callback signatures | https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html | yes | web fetch |
| 6b | Hook execution order | https://lightning.ai/docs/pytorch/stable/common/hooks.html | yes | web fetch |
| 6c | on_test_epoch_end timing | https://lightning.ai/docs/pytorch/stable/common/hooks.html | yes | web fetch |
| 7a | PyG DataLoader + Batch object | https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html | yes | web fetch + context7 |
| 7b | Batch vector, edge_index offset | https://github.com/pyg-team/pytorch_geometric/blob/master/docs/source/advanced/batching.rst | yes | context7 |
| 7c | LightningDataset API | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.lightning.LightningDataset.html | yes | web fetch |
| 7d | PyG + Lightning gotchas | Multiple sources (context7 + web) | yes | context7 + web search |
