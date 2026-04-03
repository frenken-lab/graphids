# Resource Profiling System — 7-Dimension Action Plan

## Context

Year-long OOM/low-utilization seesaw. The budget module (`node_budget()`) sizes
DynamicBatchSampler's `max_num`, but relies on a forward-only probe and two
heuristics: `_SAFETY_MARGIN=0.85` and `_GRAD_MULTIPLIER=2`. Seven blind spots
remain. This plan replaces heuristics with measurements and closes the
probe→train→calibrate feedback loop.

## The 7 Dimensions

| # | Dimension | Currently | Gap |
|---|-----------|-----------|-----|
| 1 | Edge variance | Budget sizes by nodes | GNN memory scales with edges; edge density varies per batch |
| 2 | Backward peak | `_GRAD_MULTIPLIER=2` heuristic | Real backward/forward ratio is model-dependent |
| 3 | Fragmentation | Not tracked | Allocator fragments over epochs; probe sees t=0 only |
| 4 | Worker RSS | Not tracked | Workers pickle-copy tensors; host RAM OOM looks like SLURM kill |
| 5 | Fusion setup peak | Not tracked | FusionDataModule.setup() loads VGAE+GAT simultaneously |
| 6 | Compile vs eager | Probe forces eager | torch.compile changes allocation patterns |
| 7 | KD teacher | Not tracked | teacher_on_device() puts teacher on GPU during _step() |

## What Goes Where

| Dimension | Probe (upfront) | Callback (runtime) | Post-analysis |
|-----------|----------------|--------------------|--------------| 
| 1. Edge variance | Read edge_count.p99/mean from cache_metadata.json, adjust margin | Log batch.num_edges + peak_vram per step | Edge→VRAM regression coefficient |
| 2. Backward peak | Run one forward+backward step, measure ratio | Log peak_vram per step | Validate probe ratio vs runtime peaks |
| 3. Fragmentation | — | Log reserved-allocated gap every N steps | Drift slope across epochs |
| 4. Worker RSS | — | Log host RSS periodically | Project RSS at epoch 100 vs SLURM --mem |
| 5. Fusion setup | VRAM pre-flight check after loading both models | — | — |
| 6. Compile delta | Record `is_compiled` in BudgetResult | — | Compare compiled vs eager run profiles |
| 7. KD teacher | Estimate teacher param footprint, subtract from free VRAM | Log peak_vram during KD step (includes teacher) | Validate estimate vs runtime |

---

## Phase 1: Training Callback + Edge Margin (ship first, zero risk)

### 1A. ResourceProfileCallback

**File:** `graphids/_lightning.py` (inline, ~60 lines)

Lightning Callback, registered as forced callback alongside checkpoint/early_stopping/device_stats.

```python
class ResourceProfileCallback(pl.Callback):
    """Per-step VRAM + batch stats → {run_dir}/resource_profile.csv"""
    
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every = log_every_n_steps
        self._writer = None
        self._step_start = None
    
    def on_fit_start(self, trainer, pl_module):
        # Open CSV writer at {trainer.default_root_dir}/resource_profile.csv
        # Fields: epoch, global_step, num_nodes, num_edges, num_graphs,
        #         cuda_allocated_mb, cuda_reserved_mb, cuda_peak_mb,
        #         host_rss_mb, step_time_ms
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = time.perf_counter()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every != 0:
            return
        # torch.cuda.max_memory_allocated() → cuda_peak_mb
        # torch.cuda.memory_allocated() → cuda_allocated_mb
        # torch.cuda.memory_reserved() → cuda_reserved_mb
        # torch.cuda.reset_peak_memory_stats()
        # resource.getrusage(RUSAGE_SELF).ru_maxrss → host_rss_mb
        # batch.num_nodes, batch.num_edges, batch.num_graphs from batch
        # step_time_ms from perf_counter delta
        # Write row to CSV
    
    def on_fit_end(self, trainer, pl_module):
        # Close CSV file handle
```

**Wiring in `_lightning.py`:**
```python
parser.add_lightning_class_args(ResourceProfileCallback, "resource_profile")
parser.set_defaults({"resource_profile.log_every_n_steps": 50})
```

**Overhead:** Non-logging steps: ~50ns (modulo check). Logging steps: ~0.3ms
(3 CUDA calls + getrusage + CSV write). Amortized: 0.006ms/step.

**Output:** `{run_dir}/resource_profile.csv` — one row per logged step.

### 1B. Node+Edge Cost Model (replaces edge_cv hack)

**File:** `graphids/core/preprocessing/budget.py`

GNN batch VRAM = A × nodes + B × edges. The current probe bakes in a single
batch's edge/node ratio as if it's universal — wrong when batches have
different graph densities.

**Probe change** (`_probe()`): build two batch pairs with different E/N ratios,
solve the 2×2 system:
```
vram_1 = A × N₁ + B × E₁
vram_2 = A × N₂ + B × E₂
```
Returns `(bytes_per_node, bytes_per_edge, gamma, alpha, beta)` — 5-tuple
(breaking change to _probe return, update all callers).

**Budget change** (`node_budget()`): read edge_count stats from
cache_metadata.json (already has p95, mean — written by can_bus.py:126),
compute conservative node-equivalent budget:
```python
edge_stats = stats["edge_count"]
node_stats = stats["node_count"]
edges_per_node_p95 = edge_stats["p95"] / node_stats["p95"]
effective_bpn = bytes_per_node + bytes_per_edge * edges_per_node_p95
mem_budget = int(free * _SAFETY_MARGIN / effective_bpn)
```

DynamicBatchSampler stays `mode="node"` — budget is pre-adjusted for edges.

**BudgetResult additions:** `bytes_per_edge: int | None`, `edges_per_node_p95: float | None`.

---

## Phase 2: Probe Expansion (replaces heuristics with measurements)

### 2A. Backward Pass Multiplier

**File:** `graphids/core/preprocessing/budget.py`, inside `_probe()`

After the existing forward-only measurement, add:

```python
# --- Backward pass peak (with gradients) ---
# step_fn already available (model._step or model.__call__)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
mem_before = torch.cuda.memory_allocated()

# Run one forward+backward
model.train()  # Enable dropout/batchnorm
loss = _extract_loss(step_fn(large_batch))
loss.backward()
torch.cuda.synchronize()

backward_peak = torch.cuda.max_memory_allocated() - mem_before
model.eval()
model.zero_grad(set_to_none=True)

backward_multiplier = backward_peak / max(fwd_peak, 1)
# Use measured multiplier instead of _GRAD_MULTIPLIER
bytes_per_node = fwd_per_node * backward_multiplier
```

Helper:
```python
def _extract_loss(output):
    """Handle _step() return formats: scalar, tuple, or dict."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        return output["loss"]
    raise TypeError(f"Cannot extract loss from {type(output)}")
```

Keep `_GRAD_MULTIPLIER` as fallback when backward measurement fails
(e.g., model without _step). Add `backward_multiplier: float | None` to
`BudgetResult`.

### 2B. KD Teacher VRAM Reservation

**File:** `graphids/core/preprocessing/budget.py`, inside `node_budget()`

```python
# After computing free VRAM (line 219):
teacher_vram = 0
if model is not None and hasattr(model, 'teacher') and model.teacher is not None:
    # Teacher param bytes (fp32 = 4 bytes per param)
    teacher_params = sum(p.numel() * p.element_size() for p in model.teacher.parameters())
    # Inference activations ~2x params (no gradients, but intermediate tensors)
    teacher_vram = int(teacher_params * 2.5)
    log.info("kd_teacher_vram", bytes=teacher_vram,
             params=teacher_params, mb=teacher_vram / 1e6)

effective_free = free - teacher_vram
mem_budget = int(effective_free * effective_margin / bytes_per_node)
```

Add `teacher_vram_bytes: int` to `BudgetResult`.

### 2C. Compile Status Tracking

**File:** `graphids/core/preprocessing/budget.py`

Minimal — just record whether the model was compiled when probed:

```python
is_compiled = hasattr(model, '_orig_mod') if model is not None else None
```

Add `is_compiled: bool | None` to `BudgetResult`. No behavioral change.
The backward probe (2A) automatically captures whichever mode the model uses.

### 2D. Fusion Setup Pre-flight

**File:** `graphids/core/preprocessing/datamodule.py`, in `FusionDataModule.setup()`

After loading both models (lines 315-317), before feature caching:

```python
if torch.cuda.is_available():
    combined = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_mem
    usage_pct = combined / total * 100
    if usage_pct > 85:
        log.warning("fusion_setup_vram_high",
                    allocated_mb=combined / 1e6,
                    total_mb=total / 1e6,
                    pct=round(usage_pct, 1))
```

Warning only — no budget change (fusion uses TensorDataset, not graph batching).

---

## Phase 3: Post-Campaign Analysis (closes the loop)

### 3A. Budget Calibration Analyzer Task

**File:** `graphids/core/artifacts/tasks.py`

New task following the existing `run_embeddings()` pattern:

```python
def run_vram_calibration(
    *,
    output_dir: Path,
    run_dir: Path,
    budget_result: dict | None = None,  # Probe predictions for this run
) -> None:
    """Compute calibration metrics from resource_profile.csv."""
    csv_path = run_dir / "resource_profile.csv"
    if not csv_path.exists():
        log.warning("no_resource_profile", run_dir=str(run_dir))
        return
    
    rows = _read_csv(csv_path)
    
    calibration = {
        # Edge → VRAM correlation
        "edge_vram_pearson_r": _pearson(rows, "num_edges", "cuda_peak_mb"),
        
        # Fragmentation: slope of (reserved - allocated) over steps
        "fragmentation_slope_mb_per_1k_steps": _linear_slope(
            rows, "global_step", lambda r: r["cuda_reserved_mb"] - r["cuda_allocated_mb"]
        ) * 1000,
        
        # Peak VRAM stats
        "peak_vram_p50_mb": _percentile(rows, "cuda_peak_mb", 0.50),
        "peak_vram_p95_mb": _percentile(rows, "cuda_peak_mb", 0.95),
        "peak_vram_p99_mb": _percentile(rows, "cuda_peak_mb", 0.99),
        
        # Worker RSS
        "host_rss_max_mb": max(r["host_rss_mb"] for r in rows),
        "host_rss_slope_mb_per_1k_steps": _linear_slope(
            rows, "global_step", lambda r: r["host_rss_mb"]
        ) * 1000,
        
        # Backward multiplier validation (if probe result available)
        "empirical_backward_mult": None,  # filled if budget_result provided
    }
    
    if budget_result and budget_result.get("bytes_per_node"):
        predicted_mb = budget_result["bytes_per_node"] * budget_result["mean_nodes"] / 1e6
        calibration["empirical_backward_mult"] = (
            calibration["peak_vram_p95_mb"] / max(predicted_mb, 1)
        )
    
    (output_dir / "vram_calibration.json").write_text(
        json.dumps(calibration, indent=2)
    )
```

**Wiring in `analyzer.py`:** Add `vram_calibration: bool = False` flag to
Analyzer class, call `run_vram_calibration()` when enabled.

### 3B. Cross-Run Aggregation

**File:** `graphids/commands/profile_budget.py`, new `--calibrate` flag

```bash
python -m graphids probe-budget --calibrate /fs/ess/PAS1266/kd-gat/dev/rf15/set_01/
```

Walks run directories, reads `artifacts/vram_calibration.json` from each,
aggregates per (model_type, scale, dataset):

- Recommended `_SAFETY_MARGIN` = 1 / (1 + p99_peak / predicted_budget)
- Recommended `backward_multiplier` = median across runs
- Worker RSS risk flag if slope > threshold

Writes `{lake_root}/reference/budget_calibration.csv`.

---

## Phase 4: Feedback Loop (future, after first calibrated campaign)

### 4A. Auto-Read Calibration in node_budget()

If `{lake_root}/reference/budget_calibration.csv` exists, `node_budget()` can
read the calibrated safety margin and backward multiplier for the current
(model_type, scale, dataset) combo. Falls back to constants if no calibration
data exists.

### 4B. Per-Model Worker Recommendations

From callback data: identify per-model optimal `num_workers` based on:
- cg_ratio from budget matrix (more workers help when cg_ratio >> 1)
- Worker RSS growth rate (diminishing returns vs memory cost)

Update `config/resources/profiles/*.yaml` worker counts.

---

## Implementation Order

| Priority | Work | Files | Status |
|----------|------|-------|--------|
| **P1** | ResourceProfileCallback | `_lightning.py` | **DONE** (session 14) |
| **P1** | Edge-aware margin | `budget.py` | **DONE** (session 14) |
| **P2** | Backward multiplier probe | `budget.py` | **DONE** (session 14) |
| **P2** | KD teacher reservation | `budget.py` | **DONE** (session 14) |
| **P2** | Fusion pre-flight warning | `datamodule.py` | **DONE** (session 14) |
| **P2** | Compile status in BudgetResult | `budget.py` | **DONE** (session 14) |
| **P3** | Calibration analyzer task | `tasks.py`, `analyzer.py` | Open — needs campaign data |
| **P3** | Cross-run aggregation | `profile_budget.py` | Open — needs campaign data |
| **P4** | Auto-read calibration | `budget.py` | Open — after P3 |
| **P4** | Worker recommendations | resource profile YAMLs | Open — after P3 |

**P1+P2 done.** P3 after running one campaign with the callback active.
P4 after reviewing calibration results.

## BudgetResult Additions

```python
@dataclass
class BudgetResult:
    # Existing fields (unchanged)
    budget: int
    mean_nodes: float
    mem_budget: int
    throughput_budget: int | None
    binding: str
    cg_ratio: float | None
    bytes_per_node: int | None = None
    gamma_us: float | None = None
    alpha_ms: float | None = None
    beta_us: float | None = None
    # New fields
    bytes_per_edge: int | None = None         # B: per-edge VRAM cost from 2×2 probe
    edges_per_node_p95: float | None = None   # p95 edge/node ratio from cache stats
    backward_multiplier: float | None = None  # measured fwd+bwd / fwd ratio
    teacher_vram_bytes: int = 0               # KD teacher estimated VRAM
    is_compiled: bool | None = None           # torch.compile status
```

## Verification

### After Phase 1:
- Submit any training job → confirm `resource_profile.csv` appears in run_dir
- Check CSV has expected columns and reasonable values
- Verify edge margin: `BudgetResult.edge_cv` populated, margin tighter for high-variance datasets

### After Phase 2:
- `scripts/submit.sh probe-budget` → backward_multiplier column in output
- Run KD training → confirm teacher_vram_bytes > 0 in budget log
- Run fusion → confirm warning if VRAM > 85%

### After Phase 3:
- `python -m graphids analyze --analyzer.vram_calibration true` → `vram_calibration.json` in artifacts/
- `python -m graphids probe-budget --calibrate <lake_dir>` → `budget_calibration.csv`

## Files Modified

| File | Changes |
|------|---------|
| `graphids/_lightning.py` | ResourceProfileCallback class + forced registration |
| `graphids/core/preprocessing/budget.py` | Edge margin, backward probe, KD teacher, compile flag, BudgetResult fields |
| `graphids/core/preprocessing/datamodule.py` | Fusion pre-flight warning |
| `graphids/core/artifacts/tasks.py` | `run_vram_calibration()` task |
| `graphids/core/artifacts/analyzer.py` | `vram_calibration` flag |
| `graphids/commands/profile_budget.py` | `--calibrate` flag |
| `tests/core/preprocessing/test_vram_budget.py` | Tests for new budget fields + backward probe |
| `graphids/config/resources/clusters.yaml` | (already done: gpu_vram section) |
