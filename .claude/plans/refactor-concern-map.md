# GraphIDS Model Refactor: Concern Map

## M1: Threshold-Flavor Anomaly Detection (VGAE + DGI)

Both models implement score-based detection with threshold discovery at test time. This concern should be factored out into a shared mixin or utility module.

### 1.1: Methods Involved

#### VGAE

**`_score` (lines 372-392)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **State touched:** `self.mu_mean`, `self.mu_std` (per-graph Mahalanobis distance)
- **Callers:** `_per_graph_errors` (line 422), `fit_score_norm` (line 471), `extract_features` (line 403)
- **Model-coupled?** YES — directly accesses mu from encoder; uses model state buffers
- **Signature:** `(self, batch) -> (recon, mahal, kl, z)`
  - Computes: reconstruction MSE per-graph, Mahalanobis distance on μ, KL divergence per-graph
  - Returns 4-tuple of per-graph scores (all 1D tensors) and latent z

**`_per_graph_errors` (lines 415-427)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **State touched:** `self.score_norm_fitted` (check), `self.score_recon_mean/std`, `self.score_mahal_mean/std`, `self.score_kl_mean/std`
- **Callers:** `test_step` (line 489)
- **Model-coupled?** YES — reads calibration buffers, checks fitted flag
- **Signature:** `(self, batch) -> errors: (N,)`
  - Normalizes recon/mahal/kl scores using z-score (loc/scale)
  - Returns per-graph max-σ anomaly evidence

**`_per_graph_scores` (DGI, lines 205-214)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/dgi.py`
- **State touched:** `self.svdd_center`
- **Callers:** `test_step` (line 235), `predict_step` (line 243)
- **Model-coupled?** YES — depends on SVDD center buffer
- **Signature:** `(self, batch) -> scores: (N,)`
  - L2 distance from SVDD center in pooled-latent space
  - Raises if svdd_center is zero (not calibrated)

**`_pooled_latent` (DGI, lines 194-203)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/dgi.py`
- **State touched:** None (reads encoder params only)
- **Callers:** `_per_graph_scores` (line 213), `calibrate_svdd_center` (line 225)
- **Model-coupled?** PARTIALLY — uses encode() method, which is model-specific
- **Signature:** `(self, batch) -> pooled_z: (N, latent_dim)`
  - Forward pass through encoder + global_mean_pool
  - Pure latent extraction, no anomaly logic

**`_init_threshold_metrics` (base.py, lines 352-357)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/base.py`
- **State touched:** Initializes `self.roc_metric` (BinaryYoudenJThreshold), `self.test_threshold`
- **Callers:** Called from `__init__` of VGAE (line 83), DGI (line 62)
- **Model-coupled?** NO — pure initialization
- **Signature:** `(self) -> None`
  - Creates metric accumulator for score/label pairs
  - Initializes threshold to None (computed at test-epoch-end)

**`_log_thresholded_metrics` (base.py, lines 364-406)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/base.py`
- **State touched:** Reads `self.roc_metric.preds`, `self.roc_metric.target`, sets/reads `self.test_threshold`, rebuilds `self.test_metrics`
- **Callers:** `on_test_epoch_end` override in VGAE (line 494), DGI (line 240)
- **Model-coupled?** PARTIALLY — depends on pre-populated roc_metric.preds/target (set by test_step)
- **Signature:** `(self) -> None`
  - Computes Youden-J threshold from pooled scores/labels
  - Rebuilds test_metrics at the discovered threshold
  - Logs per-set metrics and operating points

**`on_save_checkpoint` (base.py, lines 408-410)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/base.py`
- **State touched:** Reads `self.test_threshold`, writes to checkpoint dict
- **Callers:** Trainer lifecycle hook (implicit, called by checkpoint callback)
- **Model-coupled?** NO — pure checkpoint persistence
- **Signature:** `(self, checkpoint: dict) -> None`
  - Persists threshold to checkpoint so reloaded models skip re-discovery

**`on_load_checkpoint` (base.py, lines 412-414)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/base.py`
- **State touched:** Reads from checkpoint, restores `self.test_threshold`
- **Callers:** Trainer checkpoint loader (implicit, called after state_dict load)
- **Model-coupled?** NO — pure checkpoint restoration
- **Signature:** `(self, checkpoint: dict) -> None`
  - Restores test_threshold from checkpoint if available

### 1.2: Test Lifecycle Invocation

**VGAE:**
- `trainer.test()` (trainer.py, line 231)
  - Line 249: Calls `fit_score_norm()` if not already fitted
  - Line 254: Calls `on_test_epoch_start()` (inherited from base)
  - Line 259: Calls `test_step()` for each batch
  - Line 261: Calls `on_test_epoch_end()` → VGAE.on_test_epoch_end (vgae.py:493) → `_log_thresholded_metrics()`

**DGI:**
- `trainer.test()` (trainer.py, line 231)
  - Line 238-241: Calls `calibrate_svdd_center()` if method exists
  - Line 254: Calls `on_test_epoch_start()`
  - Line 259: Calls `test_step()` for each batch
  - Line 261: Calls `on_test_epoch_end()` → DGI.on_test_epoch_end (dgi.py:239) → `_log_thresholded_metrics()`

### 1.3: State Management

**VGAE Buffers (registered in `_register_score_norm_buffers`, lines 92-98):**
- `score_recon_mean`, `score_recon_std` — per-component normalization (mean, scale)
- `score_mahal_mean`, `score_mahal_std` — Mahalanobis distance normalization
- `score_kl_mean`, `score_kl_std` — KL divergence normalization
- `mu_mean`, `mu_std` — latent space stats (latent_dim-dimensional)
- `score_norm_fitted` — boolean flag, set to True after `fit_score_norm()` completes

**DGI Buffers (registered in `__init__`, line 69):**
- `svdd_center` — (latent_dim,) centroid of training-normal pooled embeddings

**Shared Threshold State (initialized in `_init_threshold_metrics`):**
- `self.roc_metric` — BinaryYoudenJThreshold accumulates scores/labels across all test batches
- `self.test_threshold` — float, computed from roc_metric at epoch-end, persisted to checkpoint

**Shared Test Metrics (created in constructor, inherited in base):**
- `self.test_metrics` — MetricCollection (binary_test_metrics()) that is rebuilt at discovered threshold
- `self._test_buffers` — dict of {test_set_name: {"scores": [], "labels": [], "preds": []}}
- `self._per_set_metrics` — dict of per-set MetricCollections, cloned from test_metrics

### 1.4: test_step Call Sites

**VGAE.test_step (lines 488-491):**
```python
def test_step(self, batch, _idx, dataloader_idx=0):
    errors = self._per_graph_errors(batch)
    self.roc_metric.update(errors.detach(), batch.y.detach())
    self._record_test_batch(dataloader_idx, scores=errors, labels=batch.y)
```
- Computes anomaly score, buffers into roc_metric and _test_buffers

**DGI.test_step (lines 234-237):**
```python
def test_step(self, batch, _idx, dataloader_idx=0):
    scores = self._per_graph_scores(batch)
    self.roc_metric.update(scores.detach(), batch.y.detach())
    self._record_test_batch(dataloader_idx, scores=scores, labels=batch.y)
```
- Identical structure; only difference is score computation method

### 1.5: binary_test_metrics vs classification_test_metrics

**Used by:**
- `binary_test_metrics(threshold=0.5)` — VGAE (line 84), DGI (line 63), threshold-flavor models
- `classification_test_metrics(num_classes)` — GAT (line 76), fusion models, classifier-flavor

**Located:** `/users/PAS2022/rf15/graphids/graphids/core/models/_metrics.py`

**Instantiation:**
- VGAE: `self.test_metrics = binary_test_metrics()` (line 84)
- DGI: `self.test_metrics = binary_test_metrics()` (line 63)
- Rebuilt at epoch-end with threshold: `binary_test_metrics(threshold=self.test_threshold)` (base.py, line 377)

---

## M2: Post-Fit Calibration

### 2.1: Method Signatures and Invocation

**`VGAE.fit_score_norm` (lines 429-486)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **Invoked from:** `trainer.test()` (trainer.py, line 251)
- **Signature:** `@torch.no_grad() (self, val_loader, device: torch.device) -> None`
- **Loader contract:** Consumes `datamodule.val_dataloader()` (passed at line 251)
- **State mutated:**
  - `self.mu_mean` ← mean of encoder latent μ over benign val graphs
  - `self.mu_std` ← std of encoder latent μ over benign val graphs
  - `self.score_recon_mean` ← mean of masked recon error over benign val
  - `self.score_recon_std` ← std of masked recon error
  - `self.score_mahal_mean` ← mean of Mahalanobis distance
  - `self.score_mahal_std` ← std of Mahalanobis distance
  - `self.score_kl_mean` ← mean of KL divergence
  - `self.score_kl_std` ← std of KL divergence
  - `self.score_norm_fitted` ← True (filled at line 486)
- **Device contract:** Loader batches moved to `device` before forward (line 446)
- **Filter:** Two-pass calibration on benign-only subsets:
  1. First pass (lines 444-458): Collect μ from benign rows, compute stats
  2. Second pass (lines 465-475): Score benign rows, collect all three error components

**`DGI.calibrate_svdd_center` (lines 216-232)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/dgi.py`
- **Invoked from:** `trainer.test()` (trainer.py, line 241)
- **Signature:** `@torch.no_grad() (self, train_loader, device: torch.device) -> None`
- **Loader contract:** Consumes `datamodule.train_eval_dataloader()` (passed at line 241)
- **State mutated:**
  - `self.svdd_center` ← mean of pooled latents over ALL training graphs
- **Device contract:** Loader batches moved to `device` (line 224)
- **Filter:** NO benign filter — uses entire train_eval loader. Comment (line 67) explains: "deterministic statistic of (encoder weights, benign train data)"

### 2.2: Loader Sources

**VGAE:**
- Loader: `datamodule.val_dataloader()` (trainer.py, line 251)
- Setup trigger: `datamodule.setup("fit")` (line 250)
- Purpose: Collect benign validation graphs for score normalization

**DGI:**
- Loader: `datamodule.train_eval_dataloader()` (trainer.py, line 241)
- Setup trigger: `datamodule.setup("fit")` (line 240)
- Purpose: Collect training-normal graphs for SVDD centroid

### 2.3: Caller Chain

**fit_score_norm:**
1. `trainer.test()` (trainer.py:219-266)
2. Line 249: Check `if fit_score_norm is not None and not bool(getattr(model, "score_norm_fitted", False)):`
3. Line 250: `datamodule.setup("fit")`
4. Line 251: `fit_score_norm(datamodule.val_dataloader(), self._device)`

**calibrate_svdd_center:**
1. `trainer.test()` (trainer.py:219-266)
2. Line 238: Check `if calibrate is not None:`
3. Line 240: `datamodule.setup("fit")`
4. Line 241: `calibrate(datamodule.train_eval_dataloader(), self._device)`

### 2.4: State Checked Before Calibration

**fit_score_norm guard (trainer.py, line 249):**
```python
if fit_score_norm is not None and not bool(getattr(model, "score_norm_fitted", False)):
```
- Skips if `score_norm_fitted` is already True (e.g., reloaded from checkpoint)

**calibrate_svdd_center guard (trainer.py, line 238):**
```python
if calibrate is not None:
```
- No check for prior calibration; always re-fits
- Design note: Centroid is "deterministic statistic of (encoder weights, benign train data)" (dgi.py:67)
- Prior design stored svdd_center in state_dict, causing deadlocks (dgi.py:235)

---

## M3: Loss Math Leak — Methods on Model Classes vs graphids/core/losses/

### 3.1: Loss Math Methods

**VGAE.neighborhood_loss_negsampled (lines 202-224)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **Signature:** `@staticmethod neighborhood_loss_negsampled(logits, node_id, edge_index, num_ids, k_neg=32) -> Tensor`
- **State touched:** NONE (pure function; decorated @staticmethod)
- **Called from:**
  - VGAETaskLoss.forward (line 89 in autoencoder.py)
  - Called as: `VGAE.neighborhood_loss_negsampled(nbr_logits, batch.node_id, nbr_edge_index, self.num_ids, k_neg=self.k_neg)`
- **True loss math?** YES — negative-sampled contrastive loss (NCE loss variant)
  - Computes: `pos_loss = -log_sigmoid(pos_logits).mean()` + `neg_loss = -log_sigmoid(-neg_logits).mean()`
  - Could move to losses/autoencoder.py alongside VGAETaskLoss
- **Model-coupled?** NO — @staticmethod, takes explicit (logits, node_id, edge_index, num_ids, k_neg)
- **Migration notes:**
  - Current: Called from losses/autoencoder.py line 89, which imports VGAE class
  - Already in the migration path: VGAETaskLoss already calls back into VGAE.neighborhood_loss_negsampled
  - Recommended move: Convert to a plain function or @classmethod in losses/autoencoder.py, stop importing from VGAE

**DGI.dgi_loss (lines 169-174)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/dgi.py`
- **Signature:** `(self, pos_z, neg_z, summary, batch_idx) -> Tensor`
- **State touched:** NONE (reads self.discriminator_weight, but that's params not loss state)
- **Called from:**
  - DGI._step (line 182): `return self.dgi_loss(pos_z, neg_z, summary, batch.batch)`
  - DGI._training_step_inner (line 185): via _step
  - DGI.validation_step (line 191): direct call
- **True loss math?** YES — contrastive MI loss: `-(log(pos_score) + log(1 - neg_score))`
- **Model-coupled?** PARTIAL — reads self.discriminator_weight
  - discriminator_weight is a Parameter, not transient loss state
  - Could pass as argument or keep on-model
- **Migration notes:**
  - DGI has no separate loss_fn module (unlike VGAE/GAT which accept loss_fn in __init__)
  - Loss is currently intrinsic to DGI (design intentional: "No loss_fn kwarg: the contrastive MI loss is intrinsic" dgi.py:29)
  - If moving: Would need to instantiate a DGITaskLoss and pass discriminator_weight to it, or keep dgi_loss on-model

**Fusion.\_supervised_loss (fusion/base.py, lines 100-105)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/fusion/base.py`
- **Signature:** `(self, td, labels) -> (scores, loss)`
- **State touched:** NONE (pure BCE on forward_scores output)
- **Called from:**
  - FusionModuleBase.training_step (line 110): `_, loss = self._supervised_loss(td, labels)`
  - FusionModuleBase.validation_step (line 120): `scores, loss = self._supervised_loss(td, labels)`
- **True loss math?** YES — Binary cross-entropy
  - `loss = BCE(scores.clamp(1e-7, 1-1e-7), labels.float())`
  - Trivial to move: just `nn.BCELoss`
- **Model-coupled?** PARTIAL — calls `self.forward_scores(td)` which is model-specific
  - forward_scores composes learned weights + score tensors (fusion/base.py, lines 85-99)
  - So loss logic must stay on-model OR forward_scores must become a separate module
- **Migration notes:**
  - Simplest fix: Keep on-model (it's already minimal)
  - Alternatively: Fusion could use a standard BCELoss and just call self.forward_scores in the step methods

### 3.2: Input Transform Methods (Architectural, Must Stay on Model)

**VGAE.apply_random_mask (lines 191-200)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **Signature:** `(self, x, node_id, mask_rate=0.15) -> (x_masked, node_id_masked, mask)`
- **State touched:** `self.mask_token` (param), `self.mask_id` (int)
- **Called from:** `_masked_forward` (line 305)
- **True loss math?** NO — architectural transform
  - Sets 15% of nodes to mask_token and mask_id
  - Part of the "predict from neighbors" training objective
  - Must stay on-model because it modifies inputs in an architecture-specific way
- **Migration notes:** NOT A CANDIDATE FOR MOVE

**VGAE.\_per_graph_masked_recon (lines 313-320)**
- **File:** `/users/PAS2022/rf15/graphids/graphids/core/models/autoencoder/vgae.py`
- **Signature:** `(self, cont, x, mask, batch_idx) -> per_graph_recon: (N,)`
- **State touched:** NONE
- **Called from:**
  - validation_step (line 350): aggregates recon error for val loss
  - _score (line 388): aggregates recon error for test scoring
  - (Would also be called from VGAETaskLoss if mask-recon became a separate loss module)
- **True loss math?** NO — metric aggregation / pooling
  - Computes MSE per-node, scatter-pools by graph, averages over masked nodes
  - Similar to gather operations in loss modules (see line 318: `scatter(..., reduce="sum")`)
  - Could move to losses/ as a helper, but currently used by both loss and scoring
- **Migration notes:** CANDIDATE FOR MOVE AS UTILITY
  - Not loss math per se, but per-graph aggregation used by loss modules
  - Could go to graphids/core/losses/utils.py alongside other reduction ops

### 3.3: Current Invocation of neighborhood_loss_negsampled

**VGAETaskLoss.forward (lines 66-111, losses/autoencoder.py)**
- Line 87-95:
  ```python
  from graphids.core.models.autoencoder.vgae import VGAE
  
  nbr_loss = VGAE.neighborhood_loss_negsampled(
      nbr_logits, batch.node_id, nbr_edge_index, self.num_ids, k_neg=self.k_neg
  )
  ```
- **Current design:** VGAETaskLoss imports VGAE class to call @staticmethod
- **Problem:** Circular import risk (VGAE imports losses/build.py, which indirectly imports VGAETaskLoss)
- **Migration target:** Move neighborhood_loss_negsampled to a free function or method in VGAETaskLoss itself

### 3.4: VGAETaskLoss Already Calls Back into VGAE

**Current:**
- VGAETaskLoss.forward(student_outputs, batch, mask) receives:
  - student_outputs = (cont_out, canid_logits, nbr_logits, z, kl_per_node)
  - batch (with x, node_id, edge_index)
  - mask (optional node mask)
- Lines 74-95: Compute recon, canid, nbr_loss, kl
- Line 89: `VGAE.neighborhood_loss_negsampled(...)` — currently a static import

**Implications:**
- VGAETaskLoss can already call back into VGAE for the nbr loss computation
- No new dependencies introduced; circular import already exists and is managed
- Move is safe: Just relocate the method or replace the import with an internal implementation

### 3.5: Summary Table

| Method | File | True Loss Math? | Model-Coupled? | Candidate for Move? | Notes |
|--------|------|-----------------|-----------------|---------------------|-------|
| `neighborhood_loss_negsampled` | vgae.py:202 | YES (NCE) | NO (@staticmethod) | YES → losses/ | Already imported by VGAETaskLoss; move to losses/autoencoder.py |
| `apply_random_mask` | vgae.py:191 | NO (input transform) | YES | NO | Must stay on model; architectural |
| `_per_graph_masked_recon` | vgae.py:313 | NO (pooling/aggregation) | NO | MAYBE → losses/utils.py | Used by both loss and scoring; candidate for shared utility |
| `dgi_loss` | dgi.py:169 | YES (MI loss) | PARTIAL (reads discriminator_weight) | MAYBE | Intrinsic to DGI; would need DGITaskLoss wrapper if moved |
| `_supervised_loss` | fusion/base.py:100 | YES (BCE) | PARTIAL (calls forward_scores) | NO | Too tightly coupled to forward_scores; keep on model or simplify to use nn.BCELoss |

