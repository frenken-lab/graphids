# Deep Learning Training: CPU–GPU, DataLoader, SLURM, and GNN Batching

> A consolidated reference covering the hardware layer, PyTorch data pipeline, SLURM primitives, and GNN-specific batching strategies for small-graph workloads.

---

## Table of Contents

1. [The Hardware Layer](#1-the-hardware-layer)
2. [The PyTorch Layer](#2-the-pytorch-layer)
3. [The SLURM Layer](#3-the-slurm-layer)
4. [GNN Batching: Why It's Different](#4-gnn-batching-why-its-different)
5. [Budget-Based Batching](#5-budget-based-batching)
6. [Model-Aware Memory Estimation](#6-model-aware-memory-estimation)
7. [Fixing the Collate / Worker Bottleneck](#7-fixing-the-collate--worker-bottleneck)
8. [Bucketing for Collate Efficiency](#8-bucketing-for-collate-efficiency)
9. [Tuning Workflow](#9-tuning-workflow)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. The Hardware Layer

**CPU** is the *host*. It runs your Python process, your DataLoader workers, your training loop logic, and orchestrates everything. It has **cores** — independent execution units that can run threads/processes in parallel.

**GPU** is a *device* attached to the CPU via PCIe (or NVLink). It has thousands of tiny cores optimized for SIMD math (matrix ops). The CPU *launches* work onto the GPU; the GPU executes it asynchronously.

### Memory is Separate and Critical

| Location | Contains |
|---|---|
| CPU RAM (system memory) | Dataset, Python objects, DataLoader prefetch buffers |
| GPU VRAM (device memory) | Model weights, activations, gradients, optimizer states |

Data must be explicitly transferred CPU RAM → GPU VRAM via `.to(device)`. The **PCIe bus** is the bottleneck between them. Modern interconnects like NVLink (GPU↔GPU) and HBM (on-device) significantly increase bandwidth [[NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)].

### Pinned Memory and DMA

By default, CPU allocations use *pageable* memory that the OS can swap to disk. For GPU transfers, CUDA first copies pageable data into a temporary pinned staging buffer — adding a redundant copy. Setting `pin_memory=True` in the DataLoader allocates batches directly in **page-locked (pinned) memory**, allowing the GPU's DMA engine to read directly from RAM without the staging copy [[PyTorch Docs: pin_memory and non_blocking](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)].

Empirically, switching to pinned transfers has been shown to increase host-to-device bandwidth from ~17 GB/s to ~26 GB/s on PCIe Gen4 hardware, and enables **asynchronous** transfers: the CPU can prepare the next batch while DMA is in flight [[ThatSzucs, Pinned Memory Benchmark](https://thatszucs.github.io/pinned-std-vector/)].

> **Caveat**: Pinned pages cannot be swapped. On memory-constrained systems, excessive pinning can pressure the OS and degrade overall performance [[PyTorch Docs](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)].

---

## 2. The PyTorch Layer

### DataLoader Workers (`num_workers`)

Workers are **CPU processes** (not threads — PyTorch uses `multiprocessing`). Each worker:
1. Reads data from disk into CPU RAM
2. Runs `__getitem__` and transforms
3. Places results into a shared memory buffer

The main process collates batches and calls `.to(device)` to transfer to GPU. So `num_workers` is purely CPU parallelism — its job is to keep the GPU fed so it never idles waiting for data [[PyTorch Forums: Guidelines for num_workers](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813)].

When `num_workers=0`, data loading and training are sequential — the GPU stalls every time the CPU fetches a batch. With `num_workers > 0`, fetching and training overlap [[GeeksforGeeks: num_workers](https://www.geeksforgeeks.org/deep-learning/how-the-number-of-workers-parameter-in-pytorch-dataloader-actually-works/)].

**Diminishing returns exist.** Increasing workers improves throughput up to a point, after which inter-process overhead and CPU saturation cause regression. Empirical benchmarks show throughput peaking around `num_workers=14` for a given workload and declining beyond that [[Medium: PyTorch num_workers tip](https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7)].

A common starting heuristic: `num_workers = 4 × num_GPUs` on a node, then tune [[PyTorch Forums](https://discuss.pytorch.org/t/understanding-dataloader-and-how-to-speed-up-gpu-training-with-num-workers/138854)]. For a single-GPU setup, 4–8 workers is usually a good range.

### `persistent_workers=True`

When `num_workers > 0` and epochs are fast (small dataset), Python incurs overhead respawning workers at the start of every epoch. Setting `persistent_workers=True` keeps workers alive between epochs, which PyTorch Lightning explicitly recommends for this pattern [[PyTorch Lightning Docs: Speed](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)].

### `multiprocessing_context='spawn'` vs `'fork'`

- `fork`: copies parent process memory — fast but **unsafe with CUDA**. CUDA contexts do not fork cleanly [[PyTorch Docs](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)].
- `spawn`: starts a fresh Python interpreter — safe but slower to start.

Always use `spawn` (or `forkserver`) when any CUDA call has been made before DataLoader workers are created.

### DDP — One Process Per GPU

For multi-GPU training, PyTorch uses **one process per GPU**, not one process managing multiple GPUs (that is `DataParallel`, which serializes through a single process and is slower). Each DDP process:
- Has its own Python interpreter
- Owns one GPU exclusively
- Holds a full copy of the model
- Communicates gradients via NCCL all-reduce after each backward pass

Gradient synchronization via NCCL happens **device-to-device**, bypassing the CPU on NVLink systems. `torchrun` / `torch.multiprocessing.spawn` handles launching N processes, assigning each a `LOCAL_RANK` (GPU index on this node) and `GLOBAL_RANK` (across all nodes) [[PyG Docs: Multi-Node Training](https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/multi_gpu_vanilla.html)].

---

## 3. The SLURM Layer

SLURM allocates resources. Here is the mapping to what actually gets provisioned:

| SLURM Primitive | What It Allocates |
|---|---|
| `--nodes` / `-N` | Physical machines |
| `--ntasks` / `-n` | Total **processes** across all nodes |
| `--ntasks-per-node` | Processes per node — **set equal to GPUs per node for DDP** |
| `--cpus-per-task` | CPU cores given to **each process** |
| `--gres=gpu:N` | GPUs per node |
| `--mem` | RAM per node |

### The Key Identity for DDP Jobs

```
total processes = nodes × ntasks-per-node
                = nodes × gpus-per-node   ← for DDP, 1 process per GPU
```

Each task (process) gets `--cpus-per-task` cores. These cores serve:
- The main training process (1 core)
- DataLoader worker subprocesses (`num_workers` cores)

So: `--cpus-per-task = num_workers + 1` (or +2 for buffer). For example, with 4 GPUs/node and 8 workers per GPU: `--cpus-per-task=9`, `--ntasks-per-node=4` → 36 cores/node needed [[KAUST HPC: PyTorch DDP](https://docs.hpc.kaust.edu.sa/soft_env/science_platforms/data_science/dist_mldl/torch_ddp.html)].

> **Pitfall**: Using `--ntasks=8` with 2 nodes does not guarantee 4 tasks per node — SLURM may pack them unevenly. Always use `--ntasks-per-node` for DDP [[CSC Docs: Multi-GPU](https://docs.csc.fi/support/tutorials/ml-multi/)].

### Example: 2-Node, 4-GPU-per-Node DDP Job

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # 1 process per GPU
#SBATCH --cpus-per-task=9        # 8 workers + 1 main thread
#SBATCH --gres=gpu:4             # 4 GPUs per node
#SBATCH --mem=256G

torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py
```

SLURM sets `SLURM_PROCID`, `SLURM_LOCALID`, etc.; `torchrun` maps these to `RANK`, `LOCAL_RANK`, `WORLD_SIZE` for PyTorch [[PyG Multi-Node Docs](https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/multi_gpu_vanilla.html)].

### Mental Model

```
SLURM node
│
├── Task 0 (process, RANK=0, LOCAL_RANK=0)
│   ├── GPU 0  ← owns this exclusively
│   ├── CPU cores [0–8]
│   │   ├── Main thread: forward/backward, optimizer step
│   │   └── Worker threads [1–8]: DataLoader prefetch → CPU RAM
│   └── Pinned RAM buffer → DMA → GPU VRAM
│
├── Task 1 (process, RANK=1, LOCAL_RANK=1)
│   ├── GPU 1
│   └── CPU cores [9–17] ...
│
...
│
└── NCCL all-reduce: GPU0 ↔ GPU1 ↔ GPU2 ↔ GPU3 (NVLink/PCIe)
         ↕ across nodes via InfiniBand
```

---

## 4. GNN Batching: Why It's Different

With image/text models, memory cost is predictable:

```
memory ∝ batch_size × C × H × W
```

With GNNs:

```
memory ∝ f(Σ nodes, Σ edges, model_depth, hidden_dim)
```

PyG's batching strategy represents a mini-batch as a **single large disconnected graph** — individual graph adjacency matrices are combined into a block-diagonal structure [[PyG Docs: Advanced Mini-Batching](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)]:

```
Adjacency matrices are stacked in a diagonal fashion (creating a giant 
graph that holds multiple isolated subgraphs), and node and target 
features are simply concatenated in the node dimension.
```

Since message passing only propagates along edges, nodes in disconnected subgraphs cannot influence each other — so no padding is needed and there is no adjacency matrix overhead beyond the edge list itself [[PyG Docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)].

The consequence is that memory cost is **highly variable** even at fixed `batch_size`. A batch of 32 small graphs may be fine; 32 slightly larger ones OOM. The model compounds this because message passing at each layer materializes intermediate node embeddings for every node in the batch.

---

## 5. Budget-Based Batching

Instead of a fixed `batch_size`, budget by **nodes or edges per batch**. PyG's `DynamicBatchSampler` does exactly this:

> Dynamically adds samples to a mini-batch up to a maximum size (either based on number of nodes or number of edges). When data samples have a wide range in sizes, specifying a mini-batch size in terms of number of samples is not ideal and can cause CUDA OOM errors.
> — [[PyG Docs: DynamicBatchSampler](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dynamic_batch_sampler.html)]

```python
from torch_geometric.data import DataLoader
from torch_geometric.loader import DynamicBatchSampler

sampler = DynamicBatchSampler(
    dataset,
    max_num=4096,         # max nodes per batch
    mode="node",          # budget by nodes, not graphs
    shuffle=True,
    skip_too_big=True     # drop graphs that alone exceed budget
)

loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
```

This gives batches bounded in memory cost regardless of graph size variance. OOM becomes a function of one knob (`max_num`) instead of the distribution of graph sizes.

### Custom Node-Budget Sampler

If you need more control or are not using PyG's DataLoader:

```python
from torch.utils.data import Sampler
import torch

class NodeBudgetSampler(Sampler):
    def __init__(self, dataset, max_nodes, shuffle=True):
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        # Precompute node counts once — not per epoch
        self.node_counts = [data.num_nodes for data in dataset]

    def __iter__(self):
        indices = torch.randperm(len(self.dataset)).tolist() \
                  if self.shuffle else list(range(len(self.dataset)))
        batch, budget = [], 0
        for idx in indices:
            n = self.node_counts[idx]
            if budget + n > self.max_nodes and batch:
                yield batch
                batch, budget = [], 0
            if n <= self.max_nodes:
                batch.append(idx)
                budget += n
        if batch:
            yield batch
```

---

## 6. Model-Aware Memory Estimation

GNN memory cost per batch depends on model depth and hidden dimension. A rough estimate for peak VRAM during a forward + backward pass:

```
peak ≈ num_layers × Σ(nodes_in_batch) × hidden_dim × bytes_per_element
     + edge_index memory  (2 × Σ edges × 8 bytes for int64)
     + optimizer state    (2× weights for Adam)
```

To set `max_num` empirically:

```python
gpu_mem_bytes  = 24 * 1024**3          # e.g. 24GB GPU
target_util    = 0.70                   # leave 30% headroom
bytes_per_node = num_layers * hidden_dim * 4  # float32
                                              # tune this empirically
max_nodes = int((gpu_mem_bytes * target_util) / bytes_per_node)
```

Or simply binary-search: start `max_num` high, find the OOM threshold, and set to 75–80% of that value.

After the first batch, inspect actual usage:

```python
print(torch.cuda.memory_summary(device, abbreviated=True))
# Target: Active memory ~60–75% of total VRAM
```

---

## 7. Fixing the Collate / Worker Bottleneck

Low GPU utilization (5–25%) with GNN training on small graphs is almost always a **CPU bottleneck**, not a GPU one. Two indicators:

- `nvidia-smi dmon` shows spiky, low GPU utilization
- High CPU usage while GPU is idle

Identifying this pattern is described in PyTorch's performance docs: "High CPU and low GPU usage often points to a data pipeline bottleneck." [[Building Efficient Data Pipelines in PyTorch](https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/efficient-data-pipelines-pytorch)]

### Move Preprocessing into the Dataset

PyG's default collate builds a block-diagonal batch graph — this involves concatenating `edge_index` tensors and offsetting node indices. This is non-trivial at high batch rates. Move any feature engineering into `__getitem__`, not collate:

```python
class PreprocessedGraphDataset(Dataset):
    def __init__(self, data_list):
        # Pre-sort by size — reduces variance within batches
        # and makes DynamicBatchSampler more efficient
        self.data = sorted(data_list, key=lambda d: d.num_nodes)

    def __getitem__(self, idx):
        d = self.data[idx]
        # Feature engineering HERE, not in collate
        return d
```

### DataLoader Settings

```python
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=6,            # increase from 2 — your graphs are small
    persistent_workers=True,  # avoid worker respawn overhead per epoch
    pin_memory=True,          # page-locked memory for faster H2D transfer
    prefetch_factor=4,        # each worker pre-queues 4 batches ahead
)
```

`pin_memory=True` enables `pin_memory_thread`, a dedicated background thread in the DataLoader that pins batches asynchronously so the main training loop doesn't block on `cudaHostAlloc` [[Abhik Sarkar: Pinned Memory](https://www.abhik.ai/concepts/pytorch/pin-memory)].

### Profiling the Pipeline

```python
import time

for i, batch in enumerate(loader):
    t0 = time.time()
    batch = batch.to(device)
    out = model(batch)
    loss = criterion(out, batch.y)
    loss.backward()
    optimizer.step()
    data_time = time.time() - t0
    print(f"step time: {data_time:.3f}s")
    # If this is dominated by .to(device) or batch construction → CPU bottleneck
```

For more precision, use `torch.profiler` to distinguish DataLoader time from forward/backward time [[PyTorch Profiler Docs](https://pytorch.org/docs/stable/profiler.html)].

---

## 8. Bucketing for Collate Efficiency

Even with dynamic batching, high size variance *within* a batch creates irregular memory patterns and slower tensor concatenation in collate. Sorting your dataset by `num_nodes` and bucketing packs similarly-sized graphs together — the same trick used in NLP sequence bucketing:

```python
# Sort indices by node count before budget-based batching
# This packs similarly-sized graphs → uniform tensor shapes
# → faster collate + more predictable GPU kernel shapes

indices = sorted(range(len(dataset)), key=lambda i: node_counts[i])

# Then run budget-based batching on sorted indices.
# Add intra-bucket shuffle if stochasticity is needed for training.
```

This is the same strategy documented for GNN small-graph packing on fixed-size hardware accelerators [[Graphcore Tutorials: Small Graph Batching with Packing](https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch_geometric/4_small_graph_batching_with_packing/README.html)].

---

## 9. Tuning Workflow

Attack this systematically in order:

```
1. Switch to DynamicBatchSampler with max_num budget
   → eliminates OOM / oscillation between OOM and under-utilization

2. Increase num_workers to 4–6 + persistent_workers=True
   → fixes CPU bottleneck stalling GPU

3. Sort by size + bucket
   → reduces collate variance, improves tensor shape regularity

4. Profile:
   - GPU util still low → increase max_num (more nodes per batch)
   - OOM → decrease max_num or check for memory leaks
   - CPU still bottleneck → increase num_workers further

5. Use torch.cuda.memory_summary() after first batch
   → verify Active memory is 60–75% of VRAM
```

Target: **flat, high GPU utilization trace** — meaning the DataLoader is always 1–2 batches ahead and the GPU never starves.

---

## 10. Common Pitfalls

**CPU bottleneck (most common):** `num_workers` too low → GPU sits idle waiting for batches. Profile with `nvidia-smi dmon` — if GPU utilization is spiky or low, it's a data pipeline bottleneck, not a model issue [[GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/how-the-number-of-workers-parameter-in-pytorch-dataloader-actually-works/)].

**Over-allocating workers:** `num_workers > cpus-per-task` → processes compete for cores. Performance degrades beyond the CPU core ceiling [[Medium: Modexa, 8 DataLoader Tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)].

**CUDA init before fork:** Any CUDA call before spawning DataLoader workers with `fork` produces cryptic errors or hangs. Always use `spawn` or explicitly set `multiprocessing_context='spawn'` [[PyTorch Docs](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)].

**Fixed batch_size with variable graphs:** The direct cause of OOM/under-utilization oscillation. Replace with node/edge budget sampling via `DynamicBatchSampler` [[PyG Docs](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dynamic_batch_sampler.html)].

**`--ntasks` without `--ntasks-per-node`:** SLURM may unevenly distribute tasks across nodes. For DDP, always specify `--ntasks-per-node` explicitly [[CSC HPC Docs](https://docs.csc.fi/support/tutorials/ml-multi/)].

**Not accounting for model depth in batch budget:** Memory cost per node scales with `num_layers × hidden_dim`. A batch budget that works for a shallow GNN will OOM with a deeper one at the same `max_num`. Recalibrate `max_num` after changing model architecture.

---

## References

| Topic | Source |
|---|---|
| PyG batching mechanics | [PyG Docs: Advanced Mini-Batching](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html) |
| DynamicBatchSampler API | [PyG Source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dynamic_batch_sampler.html) |
| Small graph packing (GNN) | [Graphcore Tutorials](https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch_geometric/4_small_graph_batching_with_packing/README.html) |
| PyTorch `pin_memory` / DMA | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) |
| Pinned memory bandwidth benchmark | [ThatSzucs](https://thatszucs.github.io/pinned-std-vector/) |
| `num_workers` guidelines | [PyTorch Forums](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813) |
| `num_workers` empirical benchmark | [Medium: Talha Anwar](https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7) |
| `persistent_workers` | [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/advanced/speed.html) |
| DataLoader pipeline bottleneck | [Building Efficient Data Pipelines](https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/efficient-data-pipelines-pytorch) |
| 8 DataLoader optimization tactics | [Medium: Modexa](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) |
| SLURM + DDP multi-node | [PyG Multi-Node Docs](https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/multi_gpu_vanilla.html) |
| SLURM + DDP (KAUST HPC) | [KAUST Docs](https://docs.hpc.kaust.edu.sa/soft_env/science_platforms/data_science/dist_mldl/torch_ddp.html) |
| SLURM multi-GPU patterns | [CSC HPC Docs](https://docs.csc.fi/support/tutorials/ml-multi/) |
| SLURM example gist | [TengdaHan/GitHub](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904) |
