# VRAM Probe — KD-Aware Step-Based Measurement

> Date: 2026-03-30
> Branch: `dagster`
> Files changed: `graphids/core/preprocessing/datamodule.py` (probe + budget), `graphids/core/models/_training.py` (teacher_on_device), `graphids/core/models/vgae.py`, `graphids/core/models/gat.py`

## Problem

`_probe_bytes_per_node()` measured only `model.forward(batch)` (student-only). During KD training, each step also runs `teacher_on_device()` → teacher forward → CPU offload. The probe missed this, setting the `DynamicBatchSampler` budget too high. On first real KD step, peak VRAM = student activations + teacher weights (~3 MB) + teacher activations, exceeding the budget. `OOMSkipMixin` caught the OOM but wasted wall time on skipped batches.

Evidence: Run 004 job `autoencoder_8e6b9f70_kd` (SLURM 46156625) — budget collapsed from 506K→168K nodes after teacher loaded (`ablation-run-004-failures.md:52-53`).

## Root Cause (two bugs, now both fixed)

1. **Teacher auto-moved to GPU by Lightning.** `self.teacher = nn_module` triggered `nn.Module.__setattr__` → registered in `_modules` → Lightning's `.to(device)` moved teacher to GPU permanently, consuming VRAM before any training step.

2. **Probe measured `forward()` not the full training step.** Even after fixing bug 1, the probe would underestimate because `forward()` excludes the teacher forward pass that `_step()` executes via `teacher_on_device()`.

## Fix Applied

### Bug 1: Teacher device management (`_training.py`, `vgae.py`, `gat.py`)

Teacher stored via `self.__dict__["teacher"]` to bypass `nn.Module._modules` registration. Lightning's `.to(device)` no longer touches it. `teacher_on_device()` rewritten to unconditionally move teacher CPU→GPU for inference, then back to CPU.

### Bug 2: Probe scope (`datamodule.py`)

`vram_node_budget()` now auto-detects `model._step` via `getattr(model, "_step", None)` and passes it to `_probe_bytes_per_node()` as `step_fn`. The probe calls `(step_fn or model)(batch)` — running the full training step (including `teacher_on_device` + teacher forward + KD loss) instead of just `forward()`.

**Zero callsite changes** — both `CANBusDataModule._build_loader` and `CurriculumDataModule.train_dataloader` already pass the model to `vram_node_budget()`, which now does the right thing automatically.

## Options Evaluated

| Option | Approach | Verdict |
|--------|----------|---------|
| **A: Probe `_step` (chosen)** | `vram_node_budget` auto-detects `model._step`, probe runs full training step | Measures real peak, no KD-specific logic, zero callsite changes |
| B: Two-pass probe | Run student forward + teacher forward separately, sum peaks | Overestimates (peaks don't overlap), duplicates step logic, fragile to divergence |
| C: First-step calibration callback | Measure peak after first real step, adjust budget retroactively | First step may OOM, budget change at epoch boundary, only works with CurriculumDataModule rebuild |

## Caveats

### `_GRAD_MULTIPLIER` overestimates for KD

The multiplier (`=2`, `datamodule.py:33`) assumes backward-pass memory ≈ forward-pass memory. It's applied to the entire probe peak, including the teacher's activation peak. But the teacher is frozen (`requires_grad_(False)`) — no backward runs through it. This means:

```
probe_peak = student_fwd + teacher_weights + teacher_fwd
budget_estimate = probe_peak × 2     ← teacher_fwd doubled unnecessarily
real_peak = max(student_fwd + teacher_fwd,   # during teacher forward in _step
                student_retained + student_grad)  # during backward
```

For KD configs, the budget is ~1.3-1.5× conservative (teacher fwd doubled when it shouldn't be). This is the **safe direction** — slightly smaller batches, no OOM. The `_SAFETY_MARGIN=0.85` provides additional headroom.

**Validation plan:** Compare `probe_bytes_per_node` log (with `method="step_fn"`) against `DeviceStatsMonitor`'s `allocated_bytes.all.peak` from the first training step in Run 005. If budget is >40% conservative, consider splitting the multiplier: `student_bpn × 2 + teacher_bpn × 1`.

### Probe runs `_step` in eval mode under `no_grad`

The probe sets `model.eval()` + `torch.no_grad()` to avoid side effects. This means:
- Student dropout disabled → slightly less memory than training mode
- No autograd graph → activations not retained for backward
- Teacher runs identically to training (always eval + no_grad in `_step`)

The `_GRAD_MULTIPLIER` compensates for the no-autograd underestimate. Dropout mask memory is negligible relative to activation tensors.

### Per-step CPU↔GPU transfer cost for teacher

Teacher weights (~745K params × 4 bytes ≈ 3 MB) transfer each step:
- PCIe 3.0 (V100): ~12 GB/s → ~0.25 ms each way → ~0.5 ms per step
- Training step: ~10-25 ms (VGAE/GAT) → <5% overhead
- PyTorch caching allocator reuses freed GPU blocks after first step

### `_step` as a probe convention

The probe uses `getattr(model, "_step", None)` — a convention, not a contract. All graph LightningModules (`VGAEModule`, `GATModule`, `DGIModule`) expose `_step`. If a future model omits it, the probe falls back to `model.forward()` (original behavior). The structured log field `method="step_fn"|"forward"` makes it visible which path was taken.

## Verification

| Check | Result |
|-------|--------|
| `orchestrate validate` (18 configs) | OK |
| `pytest --collect-only` (81 tests) | OK |
| Module imports | OK |
| `_teacher_on_cpu` references removed | 0 matches in codebase |
| Teacher excluded from `state_dict` | Confirmed via unit test |
| Teacher excluded from `_modules` | Confirmed via unit test |

Full test suite requires SLURM submission (`scripts/slurm/run_tests_slurm.sh`). Real VRAM measurement requires a GPU job — first signal will come from Run 005 smoke test on `gpudebug`.
