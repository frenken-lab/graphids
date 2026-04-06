Here's the clean responsibility map:

**Jsonnet** — structure and composition only. Inheritance between configs, environment overlays (dev vs slurm vs cluster), shared blocks that get reused across experiments. It produces a raw dict. No validation, no types, just the shape of the config file.

**Jsonargparse** — CLI binding and override injection. Takes the resolved jsonnet dict and lets you do `--model.heads 8` on the command line without rewriting the config file. Also handles lazy vs eager: some values (like dataset size) can't be known until runtime, jsonargparse lets you defer those. It produces a namespace or dict with CLI overrides merged in.

**Pydantic** — validation at the Python boundary only. Takes the resolved+overridden dict and turns it into a typed Python object with guarantees. This is where you catch "heads must be > 0" or "vgae_ckpt_path must exist on disk". Fails fast with a clear error before any training starts. Lives inline next to whatever it's validating — `GATConfig` next to the GAT model, `CANBusConfig` next to the dataset.

**Dagster** — orchestration and run tracking only. Knows about dependencies between pipeline stages (VGAE must train before GAT, GAT must train before fusion). Owns the run record/sidecar. Does not know about model internals.

**Slurm** — resource allocation only. CPUs, GPUs, memory, wall time. Dagster tells slurm what to run, slurm decides where and when.

The pipeline is strictly one-directional:

```
jsonnet resolves
    ↓
jsonargparse merges CLI overrides
    ↓
Pydantic validates → typed Python objects
    ↓
your code runs — dagster tracks it, slurm executes it
```

Nothing flows backwards. Dagster doesn't touch configs. Pydantic doesn't know about CLI. Jsonnet doesn't know about types. Each layer hands off to the next and gets out of the way.

The thing currently in `contracts/` that does NFS/GPS writes and I/O specs — that's not config and it's not types. That's infrastructure that belongs in a `core/io.py` or alongside the dagster assets that actually trigger those writes.
