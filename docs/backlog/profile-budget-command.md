# profile-budget command

## Problem

The budget module (`graphids/core/preprocessing/budget.py`) computes batch sizes from an affine GPU cost model `T_gpu = α + β·N`, but α and β have never been measured on real hardware. The estimated probe values in `test_budget_matrix.py` are guesses from architecture sizes. Until we run the probe on GPU, we don't know:

1. Whether α > 0 (if not, the throughput ceiling never exists and the module reduces to VRAM / bytes_per_node)
2. Whether the affine model actually fits (R² of a multi-point probe)
3. What the real bytes_per_node values are per model type

## Proposed command

`python -m graphids profile-budget [--dataset X] [--model-type X] [--scale X]`

### What it does

For each (model_type, scale, dataset) combo:

1. Instantiate the model on GPU with random weights (activation shapes don't depend on trained weights)
2. Load the dataset from cache
3. Run `_probe()` — measures bytes_per_node, γ, α, β
4. Compute derived values: cg_ratio, mem_budget, throughput_budget, binding
5. Output structured results (table + JSON)

### Default: all combos

8 model configs × 4 datasets with cache = 32 probes. Each probe takes ~2s (warmup + BenchmarkTimer). Total: ~1-2 minutes on GPU.

### Output format

```
model_type  scale  dataset   bytes/node  γ(μs)  α(ms)   β(μs)  cg_ratio  mem_budget  tput_budget  binding
vgae        small  hcrl_ch   1847        62.3   1.82    0.11   8.42      4,891,203   2,341        throughput
vgae        small  set_01    1847        71.1   1.82    0.11   7.33      4,891,203   2,014        throughput
gat         large  set_02    3412        68.9   4.11    0.53   1.21      2,647,892   None         memory
...
```

Plus JSON for programmatic consumption.

### SLURM submission

`scripts/submit.sh profile-budget` — GPU partition, 1 GPU, ~10 min wall time, 8 CPUs.

### Implementation

Register as a command in `__main__.py _COMMAND_MODULES`. Module: `graphids/commands/profile_budget.py`. Instantiate models via jsonargparse from stage + scale YAMLs (same path as LightningCLI but without Trainer). Load datasets via `CANBusDataset` from cache.

### What to do with the results

1. Replace estimated values in `tests/core/preprocessing/test_budget_matrix.py` `MODEL_PROBES` dict with measured values
2. If α ≈ 0 for all models → delete throughput ceiling code, budget = VRAM / bytes_per_node
3. If affine model doesn't fit (add multi-point probe + R² check) → reconsider cost model
4. Log probe results to `docs/reference/` as a measured reference table
