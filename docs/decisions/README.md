# Architecture Decision Log

Consolidated 2026-04-09. Full rationale for each decision is in git history
(search by ADR number). Decisions are permanent — if reversed, add a note here.

## Decisions

**0001 — Reject Hydra/OmegaConf.**
Jsonnet replaced YAML chains for config composition; Hydra's defaults lists offer no advantage over jsonnet's native imports, and OmegaConf's DictConfig impedance creates a dual-merge anti-pattern.

**0002 — Forced callbacks via explicit construction.**
Stage configs that set `trainer.callbacks` silently dropped ModelCheckpoint/EarlyStopping (jsonnet list replacement). Critical callbacks now live in top-level namespaces (`checkpoint.*`, `early_stopping.*`) and are constructed by `instantiate._build_callbacks()`, immune to stage overrides.

**0003 — Consolidate train/test/analyze into one SLURM job.**
Separate SLURM jobs for each phase caused analysis to run on CPU dagster workers and introduced process-boundary failures. A single job runs all three phases sequentially with per-phase markers (`.train_complete`, `.test_complete`, `.analyze_complete`).

**0004 — Keep custom VRAM probe, reject Lightning profilers.**
The VRAM probe must run *before* DataLoader construction to size `NodeBudgetBatchSampler`. All Lightning profilers/callbacks run *after* the DataLoader is built — lifecycle mismatch makes them unusable for batch sizing.

**0005 — Wandb removed as direct dependency; OTel replaces it.**
WandbLogger/CSVLogger replaced by OpenTelemetry (`OTelTrainingCallback` + `OTelTrainingLogger`). Wandb Weave receives traces optionally via OTLP when `WANDB_API_KEY` is set. Model Registry and Data Artifacts were rejected (quota limits, no `file://` support).

**0006 — Dagster removed; Monarch actors replaced it.**
Dagster's multi-job model (one SLURM job per asset) caused queue-wait overhead between pipeline stages. Monarch actors run the 3-stage chain in a single SLURM allocation with in-process `ResolvedConfig.resolve()` — no serialization boundary, no inter-job wait.

**0007 — Config system: independent axes + typed contract.**
Config combinatorial explosion (scale x model in one file) and parallel topology declarations caused silent drift. Fix: independent config axes in jsonnet, `TrainingRunConfig` (Pydantic, `extra="forbid"`) for boundary parameters, `ConfigResolver` for cross-field validation. Don't adopt Hydra, don't mirror every `__init__` signature.

**0008 — No custom collation; prebatched path supersedes both.**
Custom `_FastCollate` was 1.6x slower than warm `Batch.from_data_list()` over full training (warm cache via `persistent_workers=True`). Both paths are now moot — prebatching collates all batches once at setup with `num_workers=0`.

**0009 — Collapse override handoff chain.**
The original 9-step handoff stringified override dicts across process boundaries, with validation only inside the SLURM job. Collapsed to 3 steps: `enumerate_assets` -> `ResolvedConfig.resolve` (render + validate) -> `instantiate`. Override typos now fail before job submission.

**0010 — Use go-jsonnet binary, not the jsonnet PyPI package.**
go-jsonnet is 10-100x faster than libjsonnet, requires no C++ compile step on OSC, and installs as a single static binary to `~/.local/bin/jsonnet`. Python access via `subprocess.run` in `graphids/config/jsonnet.py` (~5ms per render, not a hot path).

## Library Evaluations (don't re-investigate)

**tach** (module boundary enforcement) — Strong fit for enforcing `config/` never imports torch and `orchestrate` never imports `core` at definition time. Not yet adopted; revisit when adding CI. Full report in git history.

**icontract** (Design by Contract) — Marginal benefit. Overlap with Pydantic for config validation is near-total. Only use case: `SLOW`-gated tensor shape/NaN contracts during development. Not adopted.

**pydantic-settings** — Adopted (session 39). `GraphIDSSettings(BaseSettings)` with `env_prefix="KD_GAT_"` replaced 19 scattered `os.environ.get()` calls.

**OpenTelemetry** — Adopted (session 39). Replaced wandb + RunRecordCallback + ResourceProfileCallback + CSVLogger + custom logging with unified OTel stack. See `docs/reference/observability.md`.
