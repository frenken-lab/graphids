# Ablation Runs 001 & 004 — Post-Mortem

> Consolidated: 2026-03-31 | Open items extracted to `open_issues.md`

## Run 001 (2026-03-24)

69-job ablation on V100. Three issues found:

1. **VRAM underutilization** — `batch_size=4096` too small for ~100K-param models, 33-42% V100 usage.
   Recommendation: 8192 for small VGAE/GAT/DGI. *Status: open (in open_issues.md as GPS batch_size).*
2. **GPS OOM** — `GPSConv` with `attn_type="multihead"` computes O(N^2) attention across mega-graph batch.
   155K nodes → 48.5GB attention matrix. *Status: open.*
3. **Data staging bottleneck** — full 86GB cache copied per job. *Status: open.*

## Run 004 (2026-03-30)

Dagster branch, 18 configs on set_01/set_02. Two orchestrator runs (cancelled + failed).

### Resolved

| Issue | Fix |
|-------|-----|
| SLURM RAM OOM (6 jobs, 24G) | `resources.yaml` bumped to 36G/4CPU for small/medium |
| Dagster subprocess crash (structlog kwargs to dagster logger) | Switched to f-string logging |
| Large GAT CUDA OOM (model-blind VRAM budget) | `vram_node_budget()` now probes real bytes/node via `_probe_bytes_per_node()` |
| Teacher VRAM auto-moved to GPU | Teacher stored via `self.__dict__` to bypass `nn.Module` registration |
| `profile_jobs.py` broken for dagster naming | Replaced by `orchestrate/profiler.py` (sacct `.batch` step, dagster name regex) |

### Open (extracted to `open_issues.md`)

- KD wall time unverified for retry path
- Dagster testing layers 0-3 missing
- ESS stale run dirs without `.complete` markers
