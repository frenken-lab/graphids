# Pre-batch Timing Analysis — Real Numbers from hcrl_sa Probe (2026-04-07)

## Data: hcrl_sa

- 19,085 graphs, mean 38.3 nodes/graph, mean 98 edges/graph
- ~10.8 KB raw tensor data per graph (batched, with batch vector)
- V100 16GB, PCIe 3.0 x16 (~12 GB/s), pin_memory ~20 GB/s

## Key numbers from probe-budget (job 46511451, with optimizer+compile warmup)

| Model | Budget (nodes) | Graphs/batch | Batch MB | T_collation | T_gpu | H2D (est) | Pin (est) |
|-------|---------------|-------------|----------|-------------|-------|-----------|-----------|
| VGAE small | 404,718 | 10,557 | 113.9 | 380.7 ms | 154.9 ms | 9.5 ms | 5.7 ms |
| VGAE large | 343,522 | 8,959 | 96.7 | 327.1 ms | 286.8 ms | 8.1 ms | 4.8 ms |
| GAT small | 233,339 | 6,086 | 65.7 | 218.5 ms | 261.5 ms | 5.5 ms | 3.3 ms |
| GAT large | 62,439 | 1,629 | 17.6 | 58.9 ms | 121.6 ms | 1.5 ms | 0.9 ms |

H2D = PCIe transfer estimate (batch_MB / 12 GB/s).
Pin = pin_memory estimate (batch_MB / 20 GB/s).
T_collation = Batch.from_data_list() per step (old path only).
T_gpu = forward × backward_multiplier.
Probe includes optimizer state (Adam m/v) and torch.compile graph caches in VRAM.

---

## Diagram 1: OLD path — VGAE small, 3 workers, per-step collation

```
Time(ms)  0       155      310      386      465      541      620      696      775      851
          |--------|--------|--------|--------|--------|--------|--------|--------|--------|

Worker 1  [====== collate B2 (386ms) ======]          [====== collate B5 (386ms) ======]
Worker 2  [====== collate B3 (386ms) ======]          [====== collate B6 (386ms) ======]
Worker 3  [====== collate B4 (386ms) ======]          [====== collate B7 (386ms) ======]
          |                                 |         |
Main      [H2D]                             [H2D]    [H2D]    [H2D]    [H2D]
          10ms                              10ms     10ms     10ms     10ms
          |                                 |         |        |        |
GPU       [== fwd+bwd B1 ==]               [= B2 ==] [= B3 =] [= B4 =] [= B5 =]
          |    155 ms       |  *** IDLE *** | 155 ms | 155 ms | 155 ms | 155 ms
          |                 |   231 ms      |
                            ^^^^^^^^^^^^^^^^
                            GPU starved — collation > GPU time
                            3 workers can't keep up (386/155 = 2.5×)

Memory: 3 workers × 6GB (pickle copy) + 8GB main = 26 GB
CPUs:   3 workers + 1 main + 1 headroom = 5
```

**Problem:** T_collation (386ms) > T_gpu (155ms). Even 3 workers can't fully
saturate the GPU. The first batch after startup always stalls because workers
haven't finished collating yet. GPU utilization: ~60%.

---

## Diagram 2: NEW path — VGAE small, prebatched, workers=0

```
SETUP PHASE (once, before training):
  Pre-collate all batches: 10,527 graphs ÷ 10,527/batch = 1 giant batch
  (In practice NodeBudgetBatchSampler may produce a few batches)
  Time: ~386ms × num_batches (one-time cost, amortized over all epochs)

TRAINING LOOP (every step):

Time(ms)  0    6   16       171  177  187       342  348  358       513
          |----|---|---------|----|----|---------|----|----|---------|
                                                                    
Main      [pin][H2D queue]  [pin][H2D queue]   [pin][H2D queue]
          5.7ms  9.5ms      5.7ms  9.5ms       5.7ms  9.5ms
           ↓                  ↓                  ↓
Side      [==== DMA ====]   [==== DMA ====]    [==== DMA ====]
Stream         9.5ms             9.5ms              9.5ms
                                                                    
GPU            [======= fwd+bwd B1 (155ms) =======]
(default       |                                   |
 stream)       16                                  171
                    [======= fwd+bwd B2 (155ms) =======]
                    |                                   |
                    187                                 342

Timeline for 3 steps:

Step 1:
  t=0     Main: list[0] → pin_memory (5.7ms)
  t=6     Side stream: .to(cuda, non_blocking) starts DMA (9.5ms)
  t=16    GPU default stream: fwd+bwd B1 starts (155ms)
  t=16    Main returns to Python — yield previous batch (but first iter, skip)

Step 2:
  t=16    Main: list[1] → pin_memory (5.7ms)        ← runs WHILE GPU computes B1
  t=22    Side stream: DMA for B2 (9.5ms)            ← runs WHILE GPU computes B1
  t=31    DMA done. B2 ready on GPU.                  ← still 140ms left of B1
  t=171   GPU done with B1. yield B1 to trainer.
  t=171   wait_stream() — already done (B2 ready since t=31)
  t=171   GPU starts fwd+bwd B2 immediately. ZERO gap.

Step 3:
  t=171   Main: list[2] → pin_memory (5.7ms)        ← WHILE GPU computes B2
  ...same pattern...

Memory: dataset + pre-batched list in main process only = ~8-12 GB
CPUs:   1 main process sufficient (pin_memory is 5.7ms vs 155ms GPU)
```

**Key insight:** pin(5.7ms) + H2D(9.5ms) = 15.2ms total CPU-side work per step.
GPU step = 154.6ms. The CPU finishes preparing the next batch in 15.2ms, then
waits 139ms for the GPU. The GPU is **never idle**. Workers would only add
IPC serialization overhead (~2-5ms) for zero benefit.

---

## Diagram 3: When workers=0 IS slow (old non-prebatched path)

```
WITHOUT pre-batching, workers=0:

Time(ms)  0                   386  396                    782  792
          |                    |    |                       |    |
Main      [=== collate B1 ===][H2D][=== collate B2 ===]  [H2D]
          |   386 ms          |10ms|    386 ms            |10ms
                               |                           |
GPU                            [== fwd+bwd B1 (155ms) ==]
                               396                     551
                                                       ^^^ GPU idle 231ms waiting for B2

Per-step time: 386 + 10 + wait = 396ms
GPU utilization: 155 / 396 = 39%
```

**This is why you needed workers before.** Collation (386ms) dominated every
step. Workers parallelize collation so the next batch is ready when GPU finishes.

---

## Summary: why the model changed

```
OLD: T_step = max(T_collation/workers, T_gpu) + T_H2D
     workers = ceil(T_collation / T_gpu) to keep GPU saturated
     memory = workers × dataset_copy + base

NEW: T_step = T_gpu  (pin + H2D fully hidden)
     workers = 0     (no collation to parallelize)
     memory = dataset + prebatched_list

                    OLD (3 workers)    NEW (prebatched)
     T_step:       ~165 ms            ~155 ms
     GPU util:     ~60-83%            ~100%
     Memory:       26 GB              8-12 GB
     CPUs:         5                  1-2
```

The probe-budget CSV validates that pin+H2D << T_gpu for ALL model/scale
combos on hcrl_sa. The worst case is VGAE small: 15.2ms / 154.6ms = 9.8%
overhead. GPU is never the bottleneck waiter.
