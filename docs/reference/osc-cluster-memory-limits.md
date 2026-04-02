# OSC Cluster Memory Limits

> Pitzer: queried via `scontrol show partition` 2026-04-02.
> Ascend/Cardinal: from OSC documentation (osc.edu), 2026-04-02.

SLURM enforces `MaxMemPerCPU` per partition. Total memory available to a job =
`cpus-per-task × MaxMemPerCPU`. Requesting more than this causes the job to be
rejected at submission.

## Pitzer

| Partition | CPUs/node | MaxMemPerCPU | Node RAM | GPUs | Notes |
|-----------|-----------|-------------|----------|------|-------|
| gpu | 40 | 9,292 MB (9.07 GB) | 363 GB | 2× V100 16GB | Primary training |
| gpudebug | 40 | 9,292 MB (9.07 GB) | 363 GB | 2× V100 16GB | 1h max |
| gpu-exp | 48 | 7,744 MB (7.56 GB) | 363 GB | 2× V100 32GB | Expanded nodes |
| gpu-quad | 48 | 15,872 MB (15.5 GB) | 744 GB | 4× V100 | 4 nodes only |
| cpu | 40 | 4,556 MB (4.45 GB) | 178 GB | — | CPU jobs |
| cpu-exp | 48 | 3,797 MB (3.71 GB) | 178 GB | — | Expanded CPU |
| debug-cpu | 48 | 4,556 MB (4.45 GB) | 178 GB | — | 1h max |

## Ascend

| Partition | CPUs/node | MemPerCPU | Node RAM | GPUs | Notes |
|-----------|-----------|----------|----------|------|-------|
| nextgen | 120 | 4,027 MB (3.93 GB) | 472 GB | 2× A100 40GB | Primary training |
| debug-nextgen | 120 | 4,027 MB (3.93 GB) | 472 GB | 2× A100 40GB | 1h max |
| quad | 88 | 10,724 MB (10.47 GB) | 922 GB | 4× A100 80GB | NVLink |
| debug-quad | 88 | 10,724 MB (10.47 GB) | 922 GB | 4× A100 80GB | 1h max |

Source: [OSC Ascend Batch Limit Rules](https://www.osc.edu/resources/technical_support/supercomputers/ascend/batch_limit_rules)

## Cardinal

| Partition | CPUs/node | MemPerCPU | Node RAM | GPUs | Notes |
|-----------|-----------|----------|----------|------|-------|
| gpu | 96 | 9,216 MB (9.0 GB) | 1 TB | 4× H100 94GB | NVLink |
| debug | 96 | 4,966 MB (4.85 GB) | 1 TB | 4× H100 94GB | 1h max, CPU+GPU |
| cpu | 96 | 4,966 MB (4.85 GB) | 512 GB | — | CPU jobs |
| longcpu | 96 | 4,966 MB (4.85 GB) | 512 GB | — | 14d max |
| hugemem | 96 | 19,891 MB (19.4 GB) | 2 TB | — | 886-1978G range |

Source: [OSC Cardinal Technical Specifications](https://www.osc.edu/resources/technical_support/supercomputers/cardinal/technical_specifications)

**Note:** Cardinal's GPU partition is `gpu`, not `batch`. Current `clusters.yaml`
maps Cardinal to `batch` — this is wrong and must be fixed.

## Cross-Cluster Validation (cpus: 8)

| Cluster | GPU Partition | MaxMemPerCPU | Max for 8 CPUs | Largest profile (52G) |
|---------|-------------|-------------|---------------|----------------------|
| Pitzer | gpu | 9,292 MB | 72.6 GB | OK |
| Ascend | nextgen | 4,027 MB | 31.5 GB | **FAIL** — need 14 CPUs |
| Ascend | quad | 10,724 MB | 83.8 GB | OK |
| Cardinal | gpu | 9,216 MB | 72.0 GB | OK |

**Ascend `nextgen` is the problem.** At 4,027 MB/CPU, 8 CPUs only gives 31.5 GB —
not enough for the 52G VGAE/DGI large profiles. Options:
- Request 14 CPUs on Ascend nextgen (14 × 3.93 = 55 GB)
- Use Ascend `quad` partition instead (10.47 GB/CPU)
- Per-cluster resource profile overrides in the config system

## CPU Partition Limits (Tier 3 — future CPU training)

| Cluster | CPU Partition | MemPerCPU | 8-CPU ceiling | 12-CPU ceiling |
|---------|-------------|----------|--------------|---------------|
| Pitzer | cpu | 4,556 MB | 35.6 GB | 53.4 GB |
| Ascend | nextgen (no GPU) | 4,027 MB | 31.5 GB | 47.2 GB |
| Cardinal | cpu | 4,966 MB | 38.8 GB | 58.2 GB |
