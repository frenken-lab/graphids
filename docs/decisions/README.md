# Architecture Decision Log

Consolidated 2026-04-09. Full rationale for each decision is in git history
(search by ADR number). Decisions are permanent — if reversed, add a note here.

## Decisions

**0001 — Reject Hydra/OmegaConf.**
Jsonnet replaced YAML chains for config composition; Hydra's defaults lists offer no advantage over jsonnet's native imports, and OmegaConf's DictConfig impedance creates a dual-merge anti-pattern.

**0002 — Forced callbacks via explicit construction.**
Stage configs that set `trainer.callbacks` silently dropped ModelCheckpoint/EarlyStopping (jsonnet list replacement). Critical callbacks now live in top-level namespaces (`checkpoint.*`, `early_stopping.*`) and are constructed by `instantiate._build_callbacks()`, immune to stage overrides.

**0003 — Consolidate train/test/analyze into one SLURM job.** *(superseded 2026-04-15: pipeline deleted entirely — each ablation preset trains+evals in one job; analysis is a separate `graphids analyze` invocation.)*
Original context: separate SLURM jobs per phase caused analysis to run on CPU dagster workers and introduced process-boundary failures.

**0004 — Keep custom VRAM probe, reject Lightning profilers.**
The VRAM probe must run *before* DataLoader construction to size `NodeBudgetBatchSampler`. All Lightning profilers/callbacks run *after* the DataLoader is built — lifecycle mismatch makes them unusable for batch sizing.

**0005 — Wandb removed as direct dependency; OTel replaces it.**
WandbLogger/CSVLogger replaced by OpenTelemetry (`OTelTrainingCallback` + `OTelTrainingLogger`). Wandb Weave receives traces optionally via OTLP when `WANDB_API_KEY` is set. Model Registry and Data Artifacts were rejected (quota limits, no `file://` support).

**0006 — Dagster removed; pipeline driver deleted.** *(updated 2026-04-15: the Python pipeline driver was itself removed — multi-stage chains became a bash loop over `scripts/run <preset.jsonnet>` with `SBATCH_DEP=afterok:<jid>`. Updated 2026-04-24: `scripts/run` was collapsed into the Typer CLI — multi-stage chains are now an in-memory DAG in `graphids.slurm.dag.OFAT_DAG` calling `graphids.slurm.submit.submit()` directly with `dep_jids` afterok chaining.)*
Dagster's multi-job model caused queue-wait overhead between stages; the Python in-process driver that replaced it duplicated declarations the jsonnet presets already made (`run_dir`, identity hash, stage DAG, upstream family mapping). Dropping the driver collapsed the two routes into one.

**0007 — Config system: independent axes + typed contract.** *(simplified 2026-04-15: `TrainingRunConfig` / `StageConfig` / `ResolvedConfig.resolve` deleted with the pipeline. Each ablation preset is now a self-contained jsonnet function; validation is Pydantic `ValidatedConfig` on the rendered dict only.)*
Original context: config combinatorial explosion (scale x model in one file) and parallel topology declarations caused silent drift. Fix: independent config axes in jsonnet, Pydantic `extra="forbid"` validator on the rendered dict. Don't adopt Hydra, don't mirror every `__init__` signature.

**0008 — No custom collation; prebatched path supersedes both.**
Custom `_FastCollate` was 1.6x slower than warm `Batch.from_data_list()` over full training (warm cache via `persistent_workers=True`). Both paths are now moot — prebatching collates all batches once at setup with `num_workers=0`.

**0009 — Collapse override handoff chain.** *(superseded 2026-04-15: the remaining two-step handoff (`ResolvedConfig.resolve → instantiate`) collapsed to one: `ResolvedConfig.from_rendered → build_run`.)*
Original context: a 9-step handoff stringified override dicts across process boundaries, with validation only inside the SLURM job. Collapsed iteratively; the final path is `render → apply_overrides → ResolvedConfig.from_rendered → build_run`.

**0010 — Use the `jsonnet` PyPI package (libjsonnet C bindings), declared in `pyproject.toml`.** *(reversed 2026-04-24: original ADR mandated the go-jsonnet binary at `~/.local/bin/jsonnet` via `subprocess.run`. In practice `graphids/config/jsonnet.py` has been calling `import _jsonnet` for months; the binding survived in the venv as an undeclared install until commit `8e14e06`'s `uv sync` pruned it, crashing 4 fusion fits on Cardinal seed 42 with `ModuleNotFoundError`.)*
The hand-installed binary at `~/.local/bin/jsonnet` is filesystem state outside the lockfile — fragile across machines, fresh checkouts, and home-dir cleanup. The PyPI package ships manylinux2014 cp312 wheels (`jsonnet==0.22.0`, no compile on install), is pinned in `uv.lock`, and survives `uv sync`. Render is ~5 ms per call; the 10–100x perf claim of go-jsonnet is irrelevant at our render volume.

## Library Evaluations (don't re-investigate)

**tach** (module boundary enforcement) — Strong fit for enforcing `config/` never imports torch and `orchestrate` never imports `core` at definition time. Not yet adopted; revisit when adding CI. Full report in git history.

**icontract** (Design by Contract) — Marginal benefit. Overlap with Pydantic for config validation is near-total. Only use case: `SLOW`-gated tensor shape/NaN contracts during development. Not adopted.

**PySlurm** (Cython SLURM bindings) — Technically viable on OSC (`libslurmfull.so` at `/usr/lib64/slurm/`, PySlurm 25.5.0 matches SLURM 25.05). Not adopted: tight version coupling (every SLURM upgrade requires rebuild), GPL-2.0 license, replaces only ~330 lines of sacct subprocess calls.

**simple_slurm** (subprocess sbatch wrapper) — Installs trivially, no version coupling, clean script-generation API. But solves a problem we don't have (sbatch generation — `submit.sh` exists), lacks sacct parsing, AGPL-3.0 license, squeue broken on array jobs (issue #44), hijacks root logger at import time (issue #42). Net code savings: ~0. Not adopted.

**pyslurmutils** (SLURM REST executor) — Blocked on OSC: requires `slurmrestd` daemon which is not available. Not adopted.

**slurm-pipeline** (shell-script pipeline DAGs) — Pipeline model doesn't fit (expects `TASK:` stdout protocol). Heavy deps (pandas+plotly). Stale (no updates since Oct 2024). Not adopted.

**Parsl** (parallel workflow library) — Strong SLURM support, auto-scaling pilot jobs, active maintenance (UChicago/Argonne, weekly releases). Not needed for current fixed 3-stage pipeline (the in-process loop is sufficient). **Worth revisiting if large hyperparameter sweeps outgrow the explicit `configs/ablations/` tree.**

**Globus Compute / funcX** (federated FaaS) — Cloud-routed task dispatch. Adds complexity without benefit for single-cluster use. Not adopted.

**Garden AI, Cascade, ProxyStore, Colmena, GlassBox** — Domain-specific or wrong phase. Not relevant.

**python-fire** (CLI generator) — Zero-boilerplate CLI from introspection. No benefit over Typer: loses type validation, repeatable `--tla` flags, and structured help. Not adopted.

**pydantic-settings** — Adopted (session 39). `GraphIDSSettings(BaseSettings)` with `env_prefix="GRAPHIDS_"` replaced 19 scattered `os.environ.get()` calls.

**OpenTelemetry** — Adopted (session 39). Replaced wandb + RunRecordCallback + ResourceProfileCallback + CSVLogger + custom logging with unified OTel stack. See `docs/reference/observability.md`.
