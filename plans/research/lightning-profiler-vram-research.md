# Research: Can Lightning's Profiler Replace `_probe_bytes_per_node()`?

> Last modified: 2026-03-30
> Verdict: **No. Keep the custom probe.**

## Why Not

**Lifecycle mismatch:** Our probe must run *before* `train_dataloader()` constructs the `DynamicBatchSampler`. All Lightning profilers/callbacks run *after* — they wrap `training_step()`.

```
setup() → train_dataloader() → _probe_bytes_per_node() → DynamicBatchSampler(budget)
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
| When | During `train_dataloader()` — right time | After DataLoader built — too late |
| What | `torch.cuda.max_memory_allocated()` delta / node count | Would need custom extraction |
| Cost | 1 forward pass (~50ms) | ≥1 full training step |
| Code | 12 lines in datamodule.py:36-72 | More code, not less |
| Grad accounting | `_GRAD_MULTIPLIER=2` heuristic | Real backward (more accurate but couples to loss fn) |

## Only Improvement Worth Considering

If the 2× gradient multiplier proves wrong, change the probe to run `model.training_step()` + `loss.backward()` instead of `model(batch)` under `torch.no_grad()`. Still uses `max_memory_allocated()`, still runs in `train_dataloader()` — no profiler needed. **Not recommended unless budget miscalculation is observed.**

## Sources

- `pytorch_lightning/profilers/{simple,advanced,pytorch,profiler}.py`
- `pytorch_lightning/callbacks/{batch_size_finder,device_stats_monitor}.py`
- `pytorch_lightning/accelerators/cuda.py:78` (`torch.cuda.memory_stats`)
- [PyTorch Lightning profiler docs](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html)
- [BatchSizeFinder API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html)
