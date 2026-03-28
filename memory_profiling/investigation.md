
  The core problem: serial vs pipelined execution

  With PyG's variable-size graphs, each batch requires Batch.from_data_list() — which walks every graph, concatenates tensors, recomputes edge_index offsets, builds batch vector. This is pure CPU work.
  Meanwhile the VGAE forward/backward on small graphs is fast on V100.

  From your profiler data, we can estimate the ratio. With num_workers=2, GPU util was 30%, meaning:

  GPU utilization ≈ N_workers × T_gpu / T_collate
  0.30 ≈ 2 × T_gpu / T_collate
  → T_collate ≈ 6.7 × T_gpu

  Collation is ~7x slower than GPU compute per batch. The GPU finishes in milliseconds and sits idle waiting for the next batch.

  Timeline diagrams (Nsight Systems / PyTorch Profiler style)

  ═══════════════════════════════════════════════════════════════════════
  num_workers=0  (GPU util: ~22%, your baseline)
  ═══════════════════════════════════════════════════════════════════════

          ┌──collate──┐         ┌──collate──┐         ┌──collate──┐
  CPU:    │▓▓▓▓▓▓▓▓▓▓▓│         │▓▓▓▓▓▓▓▓▓▓▓│         │▓▓▓▓▓▓▓▓▓▓▓│
          └───────────┘         └───────────┘         └───────────┘
               ┌─H2D─┐              ┌─H2D─┐              ┌─H2D─┐
  xfer:        │░░░░░│              │░░░░░│              │░░░░░│
               └─────┘              └─────┘              └─────┘
                    ┌─fwd+bwd─┐         ┌─fwd+bwd─┐
  GPU:              │██████████│         │██████████│         ...
                    └──────────┘         └──────────┘

  time: ──|─────70ms──────|──10ms──|─────70ms──────|──10ms──|───
          ↑               ↑        ↑
          collate          GPU      idle waiting
          (main thread)    burst    for next collate


  Everything is serial. GPU idles during collate. Collate idles during GPU.
  Total per batch: T_collate + T_transfer + T_gpu ≈ 70 + 2 + 10 = 82ms
  GPU duty cycle: 10/82 = 12%  (nvidia-smi samples see ~22% due to overlap)


  ═══════════════════════════════════════════════════════════════════════
  num_workers=1, prefetch_factor=2  (GPU util: ~15-20%)
  ═══════════════════════════════════════════════════════════════════════

  W0:     │▓▓▓collate▓▓▓│▓▓▓collate▓▓▓│▓▓▓collate▓▓▓│▓▓▓collate▓▓▓│
          └─────────────┘└─────────────┘└─────────────┘└─────────────┘
                  ↓ queue(2)    ↓              ↓              ↓
  Main:        [pull]──H2D──[pull]──H2D──[pull]──H2D──[pull]──H2D──
                    ↓              ↓              ↓              ↓
  GPU:         [fwd+bwd]---gap---[fwd+bwd]---gap---[fwd+bwd]---gap---

  1 worker produces at rate 1/T_c. GPU consumes at 1/T_g.
  Since T_c ≈ 7×T_g, worker can't keep up. Prefetch buffer drains.
  After initial 2 batches, GPU waits ~60ms between bursts.
  Worse than num_workers=0 due to spawn overhead + IPC serialization.
  CPU: 1 core for worker + 1 main = 2 cores needed.


  ═══════════════════════════════════════════════════════════════════════
  num_workers=2, prefetch_factor=2  (GPU util: ~30%, your test)
  ═══════════════════════════════════════════════════════════════════════

  W0:     │▓▓collate▓▓│         │▓▓collate▓▓│         │▓▓collate▓▓│
  W1:          │▓▓collate▓▓│         │▓▓collate▓▓│         │▓▓collate▓▓│
          ─────────────────────────────────────────────────────────────
  queue:  [b0,b1,b2,b3]→[b2,b3,b4]→[b4,b5]→[b5,b6]→[b6,b7,b8]→ ...
          ─────────────────────────────────────────────────────────────
  Main:   [pull─H2D][pull─H2D][pull─H2D]  ...  [pull─H2D][pull─H2D]
                ↓         ↓         ↓                ↓         ↓
  GPU:    [fwd+bwd][fwd+bwd]--gap--[fwd+bwd]  [fwd+bwd][fwd+bwd]--gap--
          ████████ ████████        ████████    ████████ ████████

  2 workers produce at rate 2/T_c ≈ 2/70ms = 28.6 batches/sec
  GPU consumes at 1/T_g ≈ 1/10ms = 100 batches/sec
  Ratio: 28.6/100 = 28.6% → matches observed ~30%

  Queue fills during startup (4 batches), GPU blasts through them,
  then alternates between bursts and gaps as workers refill.
  CPU: 2 workers + 1 main = 3 cores needed.
  RAM: main + 2 × dataset copy ≈ 15G + 2×5.9G = 27G (hit 36G with overhead)


  ═══════════════════════════════════════════════════════════════════════
  num_workers=4, prefetch_factor=2  (predicted GPU util: ~57%)
  ═══════════════════════════════════════════════════════════════════════

  W0:     │▓▓collate▓▓│                   │▓▓collate▓▓│
  W1:       │▓▓collate▓▓│                   │▓▓collate▓▓│
  W2:         │▓▓collate▓▓│                   │▓▓collate▓▓│
  W3:           │▓▓collate▓▓│                   │▓▓collate▓▓│
          ─────────────────────────────────────────────────────────────
  queue:  [b0..b7]→[b5..b7,b8]→[b7,b8,b9,b10]→ ...  (deeper buffer)
          ─────────────────────────────────────────────────────────────
  GPU:    [fwd+bwd][fwd+bwd][fwd+bwd][fwd+bwd]--gap--[fwd+bwd][fwd+bwd]
          ████████ ████████ ████████ ████████         ████████ ████████

  4 workers produce at 4/70ms = 57.1 batches/sec
  GPU consumes at 100 batches/sec
  Ratio: 57.1/100 = 57% → longer burst runs before queue drains

  CPU: 4 workers + 1 main = 5 cores (fits in 6 allocated)
  RAM: 15G + 4×5.9G = 39G → need --mem=48G


  ═══════════════════════════════════════════════════════════════════════
  num_workers=7  (theoretical 100% GPU util — NOT feasible)
  ═══════════════════════════════════════════════════════════════════════

  7 workers × 1/70ms = 100 batches/sec = GPU consumption rate
  But: 7 workers + 1 main = 8 cores (exceeds 6 allocated)
  RAM: 15G + 7×5.9G = 56G
  Not practical. Diminishing returns past 4 workers.

  The resource tradeoffs

  workers  GPU util  CPU cores   RAM (set_02)   Wall time (est)
  ──────── ──────── ────────── ────────────── ─────────────────
     0       22%      1 core       15G          30+ min (∞)
     1       15%      2 cores      21G          worse (IPC overhead)
     2       30%      3 cores      27G (36G*)   18 min ✓
     4       57%      5 cores      39G (48G*)   ~11 min
     6       80%      7 cores      50G (54G*)   ~8 min (need more CPUs)
     7      100%      8 cores      56G          ~7 min (not feasible)

  * with OS/PyTorch/CUDA overhead

  Why it can never reach 100% on this workload

  The fundamental bottleneck is PyG collation cost relative to GPU compute time. With small graphs (30-50 nodes) and a small model (100K params), the GPU forward+backward is ~10ms while
  Batch.from_data_list() on a dynamic batch of ~8000 graphs takes ~70ms on a single CPU core.

  The only ways to close the gap:
  1. More workers (linear CPU/RAM cost, diminishing returns)
  2. Larger model (makes T_gpu longer, better ratio — this happens with GAT vs VGAE)
  3. Pre-collated batches (do collation offline, save as single tensors — big engineering lift)
  4. Reduce collation cost (smaller batches, but then GPU underutilizes VRAM)

  Recommendation for Run 004

  num_workers=4 with --mem=48G and --cpus-per-task=6 is the sweet spot. Gets ~57% GPU util, fits in the current CPU allocation, and cuts wall time roughly in half vs num_workers=2. Want me to update the
  config and resource profiles?


he deciding factor is: does the call site receive an InMemoryDataset (has _data + slices), or a plain list[Data]? Fast collate can only work on the former — there's no pre-collated tensor to slice from
  in a list.

  ┌─────┬─────────────────────────────────────┬─────────────────────────────────┬──────────────────────────────────┬───────────────────────────────┐
  │  #  │              Location               │           Input type            │            Hot loop?             │       Use fast_collate?       │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 3   │ datamodule._build_loader            │ CANBusDataset                   │ Yes — every batch, every epoch   │ Yes                           │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 5   │ curriculum.py:133 train loader      │ CANBusDataset                   │ Yes — every batch, every epoch   │ Yes                           │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 6   │ curriculum.py:160 val loader        │ CANBusDataset                   │ Yes — every val epoch            │ Yes                           │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 1   │ vgae.py:329 _score_difficulty       │ list[Data]                      │ No — once per curriculum rebuild │ No — no _data/slices to slice │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 2   │ temporal.py:113                     │ list[Data]                      │ No — once                        │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 4   │ datamodule.py:377 cache_predictions │ list[Data] (from list(dataset)) │ No — once at fusion setup        │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 7   │ vgae.py:373 score_graphs            │ list[Data]                      │ No — once at eval                │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 8   │ vgae.py:560 encode_dataset          │ list[Data]                      │ No — once at eval                │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 9   │ gat.py:196 encode_dataset           │ list[Data]                      │ No — once at eval                │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 10  │ dgi.py:237 encode_dataset           │ list[Data]                      │ No — once at eval                │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 11  │ _training.py:174 run_evaluation     │ list[Data] or pre-built loader  │ No — once at eval                │ No                            │
  ├─────┼─────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
  │ 12  │ loss_landscape.py:175               │ list[Data]                      │ No — once                        │ No
