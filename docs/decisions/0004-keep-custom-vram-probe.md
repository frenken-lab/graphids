# Research: Can Lightning's Profiler Replace `_probe_bytes_per_node()`?

> **Stale reference note (2026-04-06):** `graphids/core/preprocessing/datamodule.py` no longer exists. Datamodules moved to `graphids/core/data/datamodule/` (graph.py, fusion.py, can_bus.py). `_probe_bytes_per_node` and `vram_node_budget` are no longer present in the codebase.

> Last modified: 2026-03-30
> Verdict: **No. Keep the custom probe.**

## Why Not

**Lifecycle mismatch:** Our probe must run *before* `DynamicBatchSampler` is constructed. All Lightning profilers/callbacks run *after* — they wrap `training_step()`.

```
setup() → train_dataloader() → _build_loader() → vram_node_budget() → _probe_bytes_per_node()
                                                                              ↓
                                                                    DynamicBatchSampler(budget)
                                                                              ↓
                                                                    DataLoader built
                                                                              ↓
                                                            training_step() ← profilers start HERE
```

**No programmatic peak memory API:** None of the 4 profilers expose whole-step peak GPU memory consumable by code:

| Profiler | Measures | Peak GPU Memory? |
|----------|----------|-----------------|
| SimpleProfiler | Wall-clock time | No |
| AdvancedProfiler | cProfile CPU time | No |
| PyTorchProfiler | Per-operator allocation deltas | No (deltas, not peak) |
| DeviceStatsMonitor | `allocated_bytes.all.peak` | Logs to logger only — no programmatic access |
| BatchSizeFinder | OOM trial-and-error | No measurement — just crash/no-crash |

## Our Probe vs Alternatives

| Aspect | `_probe_bytes_per_node()` | Any profiler approach |
|--------|--------------------------|----------------------|
| When | During `_build_loader()` → `vram_node_budget()` — right time | After DataLoader built — too late |
| What | `torch.cuda.max_memory_allocated()` delta / node count | Would need custom extraction |
| Cost | 1 `_step()` call (~50ms) | ≥1 full training step |
| Code | 42 lines in `preprocessing/datamodule.py:36-77` | More code, not less |
| Grad accounting | `_GRAD_MULTIPLIER=2` heuristic | Real backward (more accurate but couples to loss fn) |

## Only Improvement Worth Considering

The probe already uses `model._step(batch)` when available (all our LightningModules expose `_step`), capturing KD teacher inference and auxiliary losses. The remaining inaccuracy is `_GRAD_MULTIPLIER=2`: it overestimates for KD (teacher backward doesn't exist) but errs on the safe side. If the 2× multiplier wastes too much VRAM headroom, measure actual backward cost by running `_step()` + `loss.backward()` with grad enabled. **Not recommended unless budget miscalculation is observed.**

## Sources

- `pytorch_lightning/profilers/{simple,advanced,pytorch,profiler}.py`
- `pytorch_lightning/callbacks/{batch_size_finder,device_stats_monitor}.py`
- `pytorch_lightning/accelerators/cuda.py:78` (`torch.cuda.memory_stats`)
- `graphids/core/preprocessing/datamodule.py:36-77` (`_probe_bytes_per_node`)
- `graphids/core/preprocessing/datamodule.py:80-128` (`vram_node_budget`)
- [PyTorch Lightning profiler docs](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html)
- [BatchSizeFinder API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html)
