# Observability Data Layers

> Status: Layer 1 implemented · Layers 2 + 3 designed, not yet built (2026-04-10)

GraphIDS observability splits into **three storage layers** with distinct
write semantics and consumers. Layer 1 is the single source of truth —
immutable per-run files written by OTel + the training loop. Layers 2
and 3 are **materialized views** over Layer 1 with different update
models: Layer 2 (workflow SQLite) is eagerly pushed on stage enter/exit
for orchestration needs; Layer 3 (DuckDB analytics catalog) is lazily
rebuilt on demand for experiment analytics. They never merge.

## At a glance

| Layer | Store | Grain | Write model | Consumer | Rebuildable? |
|---|---|---|---|---|---|
| **1. Source of truth** | `{run_dir}/traces.jsonl`, `metrics.jsonl`, `checkpoints/*.ckpt`, `.complete` marker | One directory per training run | Append-only, fsync'd via OTel `SimpleSpanProcessor` + `torch.save` | Layers 2 + 3, debuggers, manual inspection | N/A — authoritative |
| **2. Workflow state** | `{lake_root}/workflow.db` (SQLite + WAL) | One row per stage × retry attempt | Synchronous INSERT/UPDATE from `_run_one_stage` | Driver (resume, retry), debugger, SLO dashboards | **No** — primary store |
| **3. Analytics catalog** | `{lake_root}/catalog/graphids.duckdb` (DuckDB) | One row per `training.fit` span | Stateless `CREATE OR REPLACE` rebuild via `rebuild-catalog` CLI | Researcher (leaderboards, ablation plots, sweep analysis) | **Yes** — always from Layer 1 |

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 1 — Source of truth (implemented)                          │
│   {run_dir}/traces.jsonl     OTel spans, fsync'd per span end    │
│   {run_dir}/metrics.jsonl    OTel periodic metrics (10s push)    │
│   {run_dir}/checkpoints/     torch.save via ModelCheckpoint      │
│   {run_dir}/.complete, .train_complete, .test_complete, …        │
│   {run_dir}/artifacts/       Analyzer outputs                    │
│   Immutable, append-only, authoritative for ONE training run     │
└──────────────────────────────────────────────────────────────────┘
       │                                             │
       │ eager push                                  │ lazy pull
       │ (on stage enter/exit)                       │ (on rebuild-catalog)
       ▼                                             ▼
┌────────────────────────────┐   ┌──────────────────────────────────┐
│ Layer 2 — workflow.db      │   │ Layer 3 — graphids.duckdb        │
│ SQLite (stdlib, WAL mode)  │   │ DuckDB (already a declared dep)  │
│                            │   │                                  │
│ pipeline_runs              │   │ runs                             │
│ stage_attempts             │   │ epoch_events  (companion)        │
│                            │   │ hyperparams   (companion)        │
│                            │   │ metrics_timeseries (VIEW)        │
│                            │   │ leaderboard        (VIEW)        │
│                            │   │                                  │
│ Grain: attempts + retries  │   │ Grain: terminal training.fit     │
│ States: running/completed/ │   │ States: OK / ERROR / UNSET       │
│         failed/skipped     │   │                                  │
│ Consumer: driver, debugger │   │ Consumer: researcher, notebook   │
│ Corruption: lose resume    │   │ Corruption: rm + rebuild-catalog │
└────────────────────────────┘   └──────────────────────────────────┘
         Join key: run_dir  (content-addressed identity path)
```

## Why three layers, not one or two

The historical DuckDB catalog (`orchestrate/ops/catalog.py`, deleted
2026-04-10) tried to be both a workflow tracker and an analytics
catalog. As a result:

- It **couldn't track retries** — only saw the terminal span per run. A
  stage that succeeded on attempt 3 showed up as one row indistinguishable
  from a first-attempt success.
- It **couldn't track skips** — a `.complete` marker hit skips training
  entirely, so no span is written, so no row exists. Resume-skip activity
  was invisible.
- It **couldn't track mid-flight** — spans only close in `on_fit_end` or
  `on_exception`. Driver crashes before fit starts (config errors, OOM in
  datamodule setup) left no row anywhere.
- It **also** didn't capture rich experiment metadata — only what
  `OTelTrainingCallback` chose to set as span attributes. Pivoting
  hyperparameters, joining epoch events, building leaderboards were all
  either impossible or required SQL gymnastics over raw JSON.

The right carve-up: **workflow DB owns orchestration state, DuckDB
catalog owns experiment outcomes, both join on `run_dir`**. Each becomes
smaller and clearer when it isn't trying to be the other.

The grains are genuinely different:

| Question | Layer 2 answers | Layer 3 answers |
|---|---|---|
| Which stages are running right now? | ✓ | ✗ (terminal-only) |
| What was mid-flight when the driver crashed? | ✓ | ✗ |
| What's the retry rate on fusion? | ✓ | ✗ (retries collapsed) |
| What's the skip rate from marker hits? | ✓ | ✗ (skips invisible) |
| How long does each stage take on average? | ✓ (wrapper wall time) | partial (span time only covers `trainer.fit`) |
| What's the best val_loss on hcrl_sa for vgae/large? | ✗ | ✓ |
| Which hyperparameters correlate with convergence? | ✗ | ✓ |
| Plot loss curves for the last 20 runs | ✗ | ✓ (epoch events) |
| KD lineage: which autoencoder span feeds which GAT run? | partial (upstream_ckpts column) | ✓ (OTel span links) |

---

## Layer 1 — Source of truth (already exists)

See [`observability.md`](observability.md) for full details. Summary of
what Layer 2 and Layer 3 consume:

**`{run_dir}/traces.jsonl`** — one JSON object per OTel span, written
by `ConsoleSpanExporter` with `SimpleSpanProcessor` (fsync per span).
The key span is `training.fit`, emitted by
`OTelTrainingCallback` (`graphids/core/monitoring.py:70`). Carries:

| Field | Populated by |
|---|---|
| `name: "training.fit"` | span creation |
| `status.status_code`: `OK` / `ERROR` / `UNSET` | `on_fit_end` / `on_exception` |
| `start_time`, `end_time` | span lifecycle |
| `resource.attributes.slurm.*` | `SlurmResourceDetector` |
| `attributes.ml.run_dir`, `ml.model_class`, `ml.max_epochs`, `ml.stage`, `ml.dataset`, `ml.scale`, `ml.seed`, `ml.model_type` | `on_fit_start` |
| `attributes.ml.epochs_run`, `ml.checkpoint.best_path` | `on_fit_end` |
| `attributes.ml.metric.*` | `on_fit_end` from `trainer.callback_metrics` (val_loss, val_acc, val_f1, train_loss, anything `model.log()` emits) |
| `attributes.hparam.*` | `OTelTrainingLogger.log_hyperparams` (flattened) |
| `events[].name = "epoch.end"` with `epoch`, `train_loss`, `val_loss`, `lr`, `early_stopping.wait_count`, `early_stopping.best_score` | `on_train_epoch_end` |
| `links[]` | `_discover_upstream_links` — OTel span links for KD lineage (VGAE/GAT upstream spans) |

**`{run_dir}/metrics.jsonl`** — per-batch telemetry pushed on a 10s
`PeriodicExportingMetricReader` interval. Carries `ml.batch.duration_s`,
`ml.train.loss`, `ml.cuda.allocated_mb`, `ml.cuda.reserved_mb`,
`ml.gpu.utilization_pct`, `ml.gpu.temperature_c`, `ml.gpu.power_w`.

**`{run_dir}/.complete`, `.train_complete`, `.test_complete`,
`.analyze_complete`** — phase markers, `touch_marker` at
`orchestrate/_setup.py:33` (fsync of file + parent dir for NFS safety).

Layer 1 is the ground truth. Layers 2 and 3 can be lost and rebuilt
(Layer 3 automatically, Layer 2 with effort) as long as Layer 1
survives. Backup strategy: rsync the lake root.

---

## Layer 2 — Workflow SQLite (proposed)

### Purpose

Track **every stage execution attempt** — including retries, skips, and
mid-flight rows — so the driver can: (a) resume from crashes knowing
what was in progress, (b) compute retry/skip analytics, (c) produce
failure-rate dashboards per stage.

### Grain

One row per `(pipeline_run_id, asset_name, attempt)` tuple. A single
stage that succeeds on attempt 3 produces three rows (failed, failed,
completed). A stage skipped by the marker check produces one row with
`status='skipped'`. A stage currently in progress produces one row with
`status='running'` that gets UPDATEd to a terminal state on exit.

### Schema

```sql
-- graphids/workflow/schema.sql

PRAGMA journal_mode = WAL;         -- concurrent readers, one writer
PRAGMA synchronous = NORMAL;       -- fsync on commit, not every write
PRAGMA foreign_keys = ON;

CREATE TABLE pipeline_runs (
    pipeline_run_id    TEXT PRIMARY KEY,        -- uuid4 hex
    dataset            TEXT NOT NULL,
    seed               INTEGER NOT NULL,
    scale              TEXT NOT NULL,
    stages             TEXT NOT NULL,           -- comma-sep: "autoencoder,supervised,fusion"
    cli_args_json      TEXT,                    -- full PipelineConfig.model_dump_json()
    user               TEXT NOT NULL,
    host               TEXT NOT NULL,
    slurm_job_id       TEXT,
    started_at         TIMESTAMP NOT NULL,
    finished_at        TIMESTAMP,
    status             TEXT NOT NULL
                       CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    final_error        TEXT
);

CREATE TABLE stage_attempts (
    attempt_id          TEXT PRIMARY KEY,                              -- uuid4 hex
    pipeline_run_id     TEXT NOT NULL REFERENCES pipeline_runs ON DELETE CASCADE,
    asset_name          TEXT NOT NULL,                                 -- "autoencoder_ff9f9014"
    stage               TEXT NOT NULL,                                 -- "autoencoder"
    model_type          TEXT NOT NULL,                                 -- "vgae"
    attempt             INTEGER NOT NULL,                              -- 1-based retry counter
    started_at          TIMESTAMP NOT NULL,
    finished_at         TIMESTAMP,
    status              TEXT NOT NULL
                        CHECK (status IN ('running', 'completed', 'failed', 'skipped')),
    skipped_reason      TEXT,                                          -- "complete_marker_exists"
    run_dir             TEXT,
    ckpt_path           TEXT,
    upstream_ckpts_json TEXT,                                          -- {asset_name: ckpt_path}
    error_type          TEXT,                                          -- "torch.cuda.OutOfMemoryError"
    error_msg           TEXT,                                          -- first 500 chars
    wall_time_s         REAL                                           -- finished_at - started_at
);

CREATE INDEX idx_attempts_run     ON stage_attempts(pipeline_run_id);
CREATE INDEX idx_attempts_asset   ON stage_attempts(asset_name);
CREATE INDEX idx_attempts_status  ON stage_attempts(status);
CREATE INDEX idx_attempts_started ON stage_attempts(started_at);
```

### Write model — synchronous, transactional, push

`graphids/workflow/db.py`:

```python
from __future__ import annotations
import json, os, socket, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SCHEMA_VERSION = 1


class WorkflowDB:
    """SQLite-backed workflow state store. WAL mode, one writer at a time.

    Safe for one pipeline driver + multiple read-only query clients.
    Not safe for concurrent drivers writing the same db from different
    SLURM jobs — use per-pipeline_run_id isolation if you do that.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._bootstrap()

    def _bootstrap(self) -> None:
        self._conn.executescript(_SCHEMA_PATH.read_text())

    # ---------- pipeline lifecycle ----------

    def start_pipeline(self, config, slurm_job_id: str | None) -> str:
        run_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, config.dataset, config.seed, config.scale,
             ",".join(config.stages), config.model_dump_json(),
             os.environ.get("USER", "unknown"), socket.gethostname(),
             slurm_job_id, _now(), None, "running", None),
        )
        return run_id

    def finish_pipeline(self, run_id: str, status: str, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE pipeline_runs SET finished_at=?, status=?, final_error=? "
            "WHERE pipeline_run_id=?",
            (_now(), status, error, run_id),
        )

    # ---------- stage lifecycle ----------

    def start_attempt(
        self, *, pipeline_run_id: str, asset_name: str, stage: str,
        model_type: str, attempt: int, upstream_ckpts: dict[str, str],
    ) -> str:
        attempt_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO stage_attempts(attempt_id, pipeline_run_id, asset_name, "
            "stage, model_type, attempt, started_at, status, upstream_ckpts_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (attempt_id, pipeline_run_id, asset_name, stage, model_type, attempt,
             _now(), "running", json.dumps(upstream_ckpts)),
        )
        return attempt_id

    def finish_attempt(
        self, attempt_id: str, *, status: str, run_dir: str | None = None,
        ckpt_path: str | None = None, error_type: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE stage_attempts SET finished_at=?, status=?, run_dir=?, "
            "ckpt_path=?, error_type=?, error_msg=?, "
            "wall_time_s = (julianday(?) - julianday(started_at)) * 86400.0 "
            "WHERE attempt_id=?",
            (_now(), status, run_dir, ckpt_path, error_type,
             (error_msg or "")[:500], _now(), attempt_id),
        )

    def record_skip(
        self, *, pipeline_run_id: str, asset_name: str, stage: str,
        model_type: str, reason: str, run_dir: str, ckpt_path: str,
    ) -> None:
        """Record a stage skipped by the .complete marker check.

        Skips get a synthetic attempt row so retry analytics can
        distinguish 'skipped (resumed)' from 'never attempted'.
        """
        self._conn.execute(
            "INSERT INTO stage_attempts(attempt_id, pipeline_run_id, asset_name, "
            "stage, model_type, attempt, started_at, finished_at, status, "
            "skipped_reason, run_dir, ckpt_path, wall_time_s) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0.0)",
            (uuid.uuid4().hex, pipeline_run_id, asset_name, stage, model_type,
             0, _now(), _now(), "skipped", reason, run_dir, ckpt_path),
        )

    # ---------- queries used by the driver itself ----------

    def mid_flight_attempts(self, pipeline_run_id: str) -> list[dict]:
        """For crash-recovery: attempts left in 'running' state from a prior run."""
        cur = self._conn.execute(
            "SELECT * FROM stage_attempts WHERE pipeline_run_id=? AND status='running'",
            (pipeline_run_id,),
        )
        return [dict(zip([c[0] for c in cur.description], row)) for row in cur]

    def close(self) -> None:
        self._conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
```

### Hook points in `_run_one_stage`

The workflow DB writer attaches four hooks to `run.py::_run_one_stage`.
Pseudo-code showing the insertion points:

```python
def _run_one_stage(
    cfg, *, dataset, seed, lake_root, user, upstream_ckpts,
    db: WorkflowDB, pipeline_run_id: str, attempt: int,
) -> tuple[str, bool]:
    # HOOK 1: Resolve identity before any writes
    resolved = ResolvedConfig.resolve(cfg, ...)
    run_dir = Path(str(resolved.paths.run_dir))
    ckpt_file = Path(str(resolved.paths.ckpt_file))

    # HOOK 2: Marker-hit skip path
    if ckpt_file.exists() and resolved.paths.complete_marker.exists():
        db.record_skip(
            pipeline_run_id=pipeline_run_id, asset_name=cfg.asset_name,
            stage=cfg.stage, model_type=cfg.model_type,
            reason="complete_marker_exists",
            run_dir=str(run_dir), ckpt_path=str(ckpt_file),
        )
        return str(ckpt_file), (run_dir / PHASE_MARKERS["analyze"]).exists()

    # HOOK 3: Actual attempt — insert on entry
    attempt_id = db.start_attempt(
        pipeline_run_id=pipeline_run_id, asset_name=cfg.asset_name,
        stage=cfg.stage, model_type=cfg.model_type,
        attempt=attempt, upstream_ckpts=upstream_ckpts,
    )

    try:
        artifacts = build(resolved.rendered, resolved.validated)
        train(artifacts, run_dir=run_dir, ckpt_file=ckpt_file, stage=cfg.stage)
        evaluate(artifacts, run_dir=run_dir, ckpt=ckpt_file, stage=cfg.stage)
        analyzed = _maybe_analyze(cfg, ckpt_file, dataset, seed)

        # HOOK 4a: Success — update to completed
        db.finish_attempt(
            attempt_id, status="completed",
            run_dir=str(run_dir), ckpt_path=str(ckpt_file),
        )
        return str(ckpt_file), analyzed

    except Exception as exc:
        # HOOK 4b: Failure — update to failed with error type
        db.finish_attempt(
            attempt_id, status="failed",
            run_dir=str(run_dir), ckpt_path=str(ckpt_file),
            error_type=type(exc).__qualname__, error_msg=str(exc),
        )
        raise
```

`run_pipeline` opens the DB once at the top of the function,
`start_pipeline`s, loops with retry, `finish_pipeline`s in a `finally`
block.

### Query examples (Layer 2 only)

```sql
-- What was running when the driver crashed?
SELECT attempt_id, asset_name, stage, started_at, run_dir
FROM stage_attempts
WHERE status='running' AND pipeline_run_id = ?;

-- Retry overhead: extra wall time per eventually-successful asset
SELECT asset_name,
       MAX(attempt) AS max_attempt,
       SUM(wall_time_s) AS total_wall_s,
       SUM(wall_time_s) FILTER (WHERE status='completed') AS successful_wall_s
FROM stage_attempts
GROUP BY asset_name HAVING MAX(attempt) > 1;

-- Failure rate by stage (tells you what's fragile)
SELECT stage,
       COUNT(*) FILTER (WHERE status='failed') * 1.0 / COUNT(*) AS failure_rate,
       COUNT(*) FILTER (WHERE status='failed' AND error_type LIKE '%OutOfMemory%')
         * 1.0 / COUNT(*) AS oom_rate
FROM stage_attempts GROUP BY stage;

-- Skip rate per asset (how much work resume is saving)
SELECT asset_name,
       COUNT(*) FILTER (WHERE status='skipped') * 1.0 / COUNT(*) AS skip_rate,
       COUNT(*) AS total_invocations
FROM stage_attempts GROUP BY asset_name;

-- Average pipeline wall time over the last 10 completed runs
SELECT AVG(julianday(finished_at) - julianday(started_at)) * 86400.0 AS avg_wall_s
FROM pipeline_runs
WHERE status='completed'
ORDER BY finished_at DESC LIMIT 10;
```

### Why SQLite, not DuckDB, for Layer 2

- **Transactional inserts + updates are SQLite's core competency**; DuckDB
  is optimized for analytical `CREATE OR REPLACE TABLE AS SELECT`
  workloads and its write path is slower for point updates.
- **WAL mode enables concurrent readers** while the driver writes.
- **Stdlib — zero new deps.**
- **Corruption is recoverable** via `sqlite3 workflow.db ".recover"` in
  the worst case. DuckDB corruption stories are worse.
- **Single-writer constraint is fine** — the pipeline driver is the only
  writer in our architecture. Multi-driver scenarios should use one DB
  per pipeline_run_id, not one shared DB.

---

## Layer 3 — DuckDB analytics catalog (redesign)

### Purpose

Experiment analytics: leaderboards, ablation comparisons, hyperparameter
sweeps, loss-curve plots, KD lineage tracing. Optimized for
**read-heavy SQL over completed training runs**, not orchestration
state.

### Grain

One row in `runs` per `training.fit` span in any `traces.jsonl` under
`{lake_root}/**`. Companion tables hold per-epoch events and pivoted
hyperparameters.

### Schema

```sql
-- graphids/catalog/schema.sql

-- Core: one row per completed or failed training run
CREATE OR REPLACE TABLE runs AS
SELECT
    -- Identity (extracted from span resource + attributes)
    json_extract_string(resource, '$.attributes."service.name"')    AS service,
    json_extract_string(resource, '$.attributes."slurm.job_id"')    AS slurm_job_id,
    json_extract_string(resource, '$.attributes."slurm.partition"') AS slurm_partition,
    json_extract_string(resource, '$.attributes."slurm.nodelist"')  AS slurm_nodelist,
    json_extract_string(attributes, '$."ml.run_dir"')               AS run_dir,
    json_extract_string(attributes, '$."ml.stage"')                 AS stage,
    json_extract_string(attributes, '$."ml.dataset"')               AS dataset,
    json_extract_string(attributes, '$."ml.scale"')                 AS scale,
    CAST(json_extract(attributes, '$."ml.seed"')       AS INTEGER)  AS seed,
    json_extract_string(attributes, '$."ml.model_type"')            AS model_type,
    json_extract_string(attributes, '$."ml.model_class"')           AS model_class,
    -- Lifecycle
    json_extract_string(status, '$.status_code')                    AS status_code,
    start_time, end_time,
    EPOCH_MS(end_time) - EPOCH_MS(start_time)                        AS fit_duration_ms,
    -- Outcomes
    CAST(json_extract(attributes, '$."ml.max_epochs"') AS INTEGER)  AS max_epochs,
    CAST(json_extract(attributes, '$."ml.epochs_run"') AS INTEGER)  AS epochs_run,
    json_extract_string(attributes, '$."ml.checkpoint.best_path"')  AS best_ckpt_path,
    -- Metrics — pulled out explicitly so queries can reference them as columns.
    -- Extend this list when new metrics are added to model.log() calls.
    CAST(json_extract(attributes, '$."ml.metric.val_loss"')  AS DOUBLE) AS val_loss,
    CAST(json_extract(attributes, '$."ml.metric.val_acc"')   AS DOUBLE) AS val_acc,
    CAST(json_extract(attributes, '$."ml.metric.val_f1"')    AS DOUBLE) AS val_f1,
    CAST(json_extract(attributes, '$."ml.metric.val_auroc"') AS DOUBLE) AS val_auroc,
    CAST(json_extract(attributes, '$."ml.metric.train_loss"') AS DOUBLE) AS train_loss,
    -- Raw attribute blob — catches anything we forgot to pivot
    attributes AS _raw_attributes,
    -- Upstream lineage — OTel span links
    links AS upstream_links,
    current_timestamp AS catalog_updated_at
FROM read_json_auto(
    '{lake_root}/**/traces.jsonl',
    format='newline_delimited',
    union_by_name=true,
    maximum_object_size=1048576
)
WHERE name = 'training.fit';

CREATE INDEX idx_runs_dataset_scale ON runs(dataset, scale);
CREATE INDEX idx_runs_status        ON runs(status_code);

-- Companion: per-epoch events (loss curves)
CREATE OR REPLACE TABLE epoch_events AS
SELECT
    json_extract_string(attributes, '$."ml.run_dir"') AS run_dir,
    json_extract_string(attributes, '$."ml.stage"')   AS stage,
    event.name                                         AS event_name,
    event.timestamp                                    AS event_time,
    CAST(json_extract(event.attributes, '$.epoch')      AS INTEGER) AS epoch,
    CAST(json_extract(event.attributes, '$.train_loss') AS DOUBLE)  AS train_loss,
    CAST(json_extract(event.attributes, '$.val_loss')   AS DOUBLE)  AS val_loss,
    CAST(json_extract(event.attributes, '$.lr')         AS DOUBLE)  AS lr,
    CAST(json_extract(event.attributes, '$."early_stopping.wait_count"') AS INTEGER)
                                                        AS early_stopping_wait_count,
    CAST(json_extract(event.attributes, '$."early_stopping.best_score"') AS DOUBLE)
                                                        AS early_stopping_best_score
FROM (
    SELECT attributes, UNNEST(events) AS event
    FROM read_json_auto('{lake_root}/**/traces.jsonl', ...)
    WHERE name = 'training.fit'
)
WHERE event.name = 'epoch.end';

-- Companion: pivoted hyperparameters (one row per run × one column per hparam is infeasible;
-- instead one row per (run, param_name, param_value) tuple)
CREATE OR REPLACE TABLE hyperparams AS
SELECT
    json_extract_string(attributes, '$."ml.run_dir"') AS run_dir,
    regexp_extract(param_key, '^hparam\.(.*)', 1)    AS param_name,
    param_value
FROM (
    SELECT attributes,
           UNNEST(json_keys(attributes)) AS param_key,
           json_extract(attributes, '$.' || param_key) AS param_value
    FROM read_json_auto('{lake_root}/**/traces.jsonl', ...)
    WHERE name = 'training.fit'
)
WHERE param_key LIKE 'hparam.%';

-- View over raw metrics.jsonl — per-batch GPU telemetry. NOT materialized —
-- queried ad-hoc so catalog rebuilds don't scan hundreds of MB of metrics.
CREATE OR REPLACE VIEW metrics_timeseries AS
SELECT *
FROM read_json_auto(
    '{lake_root}/**/metrics.jsonl',
    format='newline_delimited',
    union_by_name=true
);

-- View: top result per (dataset, scale, model_type)
CREATE OR REPLACE VIEW leaderboard AS
SELECT dataset, scale, model_type,
       MIN(val_loss) AS best_val_loss,
       MAX(val_acc)  AS best_val_acc,
       MAX(val_f1)   AS best_val_f1,
       ARG_MIN(run_dir, val_loss) AS best_val_loss_run,
       ARG_MAX(run_dir, val_acc)  AS best_val_acc_run,
       COUNT(*) FILTER (WHERE status_code = 'OK') AS n_successful,
       COUNT(*) FILTER (WHERE status_code = 'ERROR') AS n_failed
FROM runs
GROUP BY dataset, scale, model_type;
```

### Write model — stateless rebuild

`graphids/catalog/build.py`:

```python
from __future__ import annotations
from pathlib import Path
import duckdb

from graphids._otel import get_logger
from graphids.config.constants import CATALOG_SUBPATH
from graphids.config.settings import require_lake_write

log = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def rebuild_catalog(*, lake_root: str, dry_run: bool = False) -> None:
    """Stateless CREATE OR REPLACE rebuild of all catalog tables from
    traces.jsonl files under {lake_root}/**. Idempotent."""
    cat_path = Path(lake_root) / CATALOG_SUBPATH
    traces_glob = str(Path(lake_root) / "**" / "traces.jsonl")

    traces_files = list(Path(lake_root).glob("**/traces.jsonl"))
    if not traces_files:
        log.info("no_traces_found", lake_root=lake_root)
        return

    if dry_run:
        log.info("rebuild_catalog_dry_run", n_traces=len(traces_files),
                 target=str(cat_path))
        return

    require_lake_write()
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(cat_path))
    try:
        schema_sql = _SCHEMA_PATH.read_text().replace("{lake_root}", lake_root)
        db.execute(schema_sql)

        n_runs = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_epochs = db.execute("SELECT COUNT(*) FROM epoch_events").fetchone()[0]
        n_hparams = db.execute("SELECT COUNT(*) FROM hyperparams").fetchone()[0]
        by_status = db.execute(
            "SELECT status_code, COUNT(*) FROM runs GROUP BY status_code"
        ).fetchall()

        log.info("catalog_rebuilt",
                 runs=n_runs, epoch_events=n_epochs, hyperparams=n_hparams,
                 by_status=dict(by_status), catalog_path=str(cat_path))
    finally:
        db.close()
```

### CLI wiring

Add to `graphids/cli/_data.py` or a new `graphids/cli/_catalog.py`:

```python
@app.command("rebuild-catalog", rich_help_panel="Analytics")
def rebuild_catalog_cmd(
    lake_root: Annotated[str, typer.Option(help="Lake root override")] = "",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Rebuild the DuckDB analytics catalog from traces.jsonl files."""
    from graphids.catalog.build import rebuild_catalog
    from graphids.config.constants import LAKE_ROOT
    rebuild_catalog(lake_root=lake_root or LAKE_ROOT, dry_run=dry_run)
```

And a submit profile so it runs inside SLURM for large lake scans:

```json
"rebuild-catalog": {
    "partition": "cpu",
    "cpus": 2,
    "mem": "8G",
    "time": "0:30:00",
    "signal": "",
    "mode": "cpu",
    "command": "python -m graphids rebuild-catalog"
}
```

### Query examples (Layer 3 only)

```sql
-- Leaderboard: best val_loss per (dataset, scale, model_type)
SELECT * FROM leaderboard WHERE dataset='hcrl_sa' ORDER BY best_val_loss;

-- Convergence: median epochs_run by scale
SELECT scale, median(epochs_run), median(max_epochs)
FROM runs WHERE status_code='OK' GROUP BY scale;

-- Loss curves for a specific run (for plotting)
SELECT epoch, train_loss, val_loss, lr
FROM epoch_events
WHERE run_dir = ? ORDER BY epoch;

-- Hyperparameter sweep: val_loss vs learning rate
SELECT CAST(h.param_value AS DOUBLE) AS lr, r.val_loss
FROM runs r JOIN hyperparams h ON h.run_dir = r.run_dir
WHERE h.param_name = 'model.init_args.lr'
  AND r.dataset = 'hcrl_sa' AND r.status_code = 'OK';

-- KD lineage: which autoencoder runs feed which GAT runs?
SELECT gat.run_dir      AS gat_run,
       upstream.run_dir AS upstream_run,
       upstream.stage,
       upstream.val_loss
FROM runs gat,
     UNNEST(gat.upstream_links) AS link,
     runs upstream
WHERE gat.model_type = 'gat'
  AND upstream.span_id = link.context.span_id;

-- Per-batch GPU telemetry for a run (using the view over metrics.jsonl)
SELECT *
FROM metrics_timeseries
WHERE json_extract_string(attributes, '$."ml.run_dir"') = ?
  AND name = 'ml.gpu.utilization_pct'
ORDER BY timestamp;
```

### Why DuckDB, not SQLite, for Layer 3

- **`read_json_auto` over a glob** is the entire ingestion pipeline —
  ~10 lines of SQL vs. a Python parser loop.
- **Columnar storage + vectorized execution** make leaderboard /
  group-by / aggregate queries fast over thousands of runs.
- **Zero-copy views over JSONL files** (`metrics_timeseries`) avoid
  materializing per-batch telemetry — DuckDB re-scans the JSONL at query
  time, which is fine because you query a single run's timeseries at a
  time.
- **DuckDB is already a declared dependency** in `pyproject.toml` — no
  new cost.
- **Corruption is cheap to recover from**: `rm catalog/graphids.duckdb &&
  python -m graphids rebuild-catalog`.

---

## Joined query examples (when you need both layers)

```sql
-- ATTACH both databases in one DuckDB session
ATTACH '{lake_root}/catalog/graphids.duckdb' AS cat (TYPE DUCKDB);
ATTACH '{lake_root}/workflow.db'          AS wf  (TYPE SQLITE);

-- Does retry-recovered training produce worse val_loss?
SELECT c.dataset, c.scale,
       w.attempt,
       AVG(c.val_loss) AS mean_val_loss,
       COUNT(*) AS n
FROM cat.runs c
JOIN wf.stage_attempts w ON w.run_dir = c.run_dir AND w.status = 'completed'
GROUP BY c.dataset, c.scale, w.attempt
ORDER BY c.dataset, c.scale, w.attempt;

-- Correlation: wall time vs final val_loss (are slower runs better?)
SELECT c.run_dir, w.wall_time_s, c.fit_duration_ms / 1000.0 AS fit_time_s,
       c.val_loss, c.epochs_run
FROM cat.runs c
JOIN wf.stage_attempts w
  ON w.run_dir = c.run_dir AND w.status = 'completed';

-- What fraction of pipeline wall time was retries?
SELECT p.pipeline_run_id,
       (julianday(p.finished_at) - julianday(p.started_at)) * 86400.0 AS total_wall_s,
       SUM(w.wall_time_s) FILTER (WHERE w.status = 'failed') AS wasted_wall_s
FROM wf.pipeline_runs p
JOIN wf.stage_attempts w USING (pipeline_run_id)
GROUP BY p.pipeline_run_id, p.finished_at, p.started_at;
```

The join key is always **`run_dir`** — the content-addressed
`{lake_root}/…/{model}_{scale}_{stage}_{identity_hash}/seed_{N}` path
from `config/paths.py::compute_identity_hash`. It's stable across both
layers because both derive it from `ResolvedConfig.resolve`.

---

## Implementation plan

### Priority order

1. **Layer 2 (workflow SQLite) first.** Gives immediate orchestration
   value: crash-recovery via mid-flight rows, retry analytics, failure
   rate dashboards. Unblocks nothing but pays for itself the first time
   a pipeline crashes and you need to know what was running.
2. **Layer 3 (DuckDB catalog) second.** Restore a cleaner version of the
   deleted `orchestrate/ops/catalog.py` with `epoch_events` + `hparams`
   companions + `leaderboard` view. Blocked by nothing technically, but
   the analytics value grows with the number of completed runs — low
   urgency until ≥20 runs accumulate.

### Estimated cost

| Task | Lines | Touches |
|---|---|---|
| `graphids/workflow/schema.sql` + `db.py` | ~180 | new package |
| Hook points in `orchestrate/run.py::_run_one_stage` | ~20 | existing |
| Open `WorkflowDB` in `run_pipeline`; thread `pipeline_run_id` + `attempt` through `_run_one_stage` | ~15 | existing |
| `tests/workflow/test_db.py` — differential test using an in-memory DB | ~80 | new |
| `graphids/catalog/schema.sql` + `build.py` | ~200 | new package |
| `rebuild-catalog` CLI command | ~15 | existing `cli/_data.py` |
| `configs/resources/submit_profiles.json` — add `rebuild-catalog` entry | ~9 | existing |
| `tests/catalog/test_rebuild.py` — fixture with synthetic traces.jsonl | ~100 | new |
| Update `docs/reference/observability.md` to point here | ~5 | existing |
| **Total** | **~625** | **4 new files, 4 edits** |

### Open questions

1. **Per-lake vs per-pipeline workflow DB?** Current design: one
   `workflow.db` per `lake_root`. Alternative: one per pipeline_run_id
   (under the run directory). Per-lake is simpler for cross-run
   analytics; per-run sidesteps concurrent-writer races. **Recommendation:
   per-lake with WAL**, since the driver is single-writer and analytics
   queries are read-only.
2. **Concurrent drivers?** If two SLURM jobs run `pipeline-run`
   simultaneously against the same lake, they'll both try to write to
   `workflow.db`. WAL mode handles concurrent readers but only one writer
   at a time — the second writer will block on SQLite's file lock up to
   `timeout=30s`. For sweep workloads this is a bottleneck. **Mitigation**:
   either use per-pipeline DBs, or introduce a lightweight coordinator
   (small server writing to the DB on behalf of workers — overkill for now).
3. **Catalog staleness**: should `rebuild-catalog` run automatically on
   every pipeline finish, or only on demand? **Recommendation**: on-demand
   only. Auto-rebuild after every run is O(N²) work and adds tail latency
   to short pipelines. A nightly cron or manual `rebuild-catalog` call
   suffices for analytics.
4. **`metrics_timeseries` as view vs table**: materializing per-batch
   metrics from all runs into one table blows up catalog size. Leaving it
   as a view over `read_json_auto('**/metrics.jsonl')` costs a full scan
   per query but queries are typically filtered to one `run_dir`, so the
   scan is cheap if DuckDB's predicate pushdown works through
   `read_json_auto`. **Recommendation**: view first, promote to a
   materialized table only if query performance demands it.

---

## What NOT to build

- **Don't merge Layers 2 and 3 into one store.** The write patterns and
  failure modes are fundamentally different. Merging re-creates the
  historical catalog's grain confusion.
- **Don't build a `metrics_timeseries` ingestion table.** Use the DuckDB
  view over raw `metrics.jsonl` files. Materialized timeseries tables
  grow unbounded and rebuild time becomes dominated by per-batch data.
- **Don't add auth, RBAC, or multi-tenancy.** Both stores are single-user
  local files inside the lake root. If you need cross-user analytics,
  point DuckDB at a shared lake path — the catalog is stateless and can
  be rebuilt by any reader.
- **Don't replace `.complete` marker files with workflow DB rows as the
  primary resume signal.** The marker is on the filesystem next to the
  checkpoint; the DB is elsewhere. If someone manually deletes a
  checkpoint, the marker disappears too but the DB row doesn't. The
  marker stays as the authoritative "this stage is done on disk" signal;
  the DB complements it with richer state.
- **Don't write to Layer 2 inside `stage.build` / `train` / `evaluate`.**
  Keep the workflow DB writes in `run.py::_run_one_stage` so the
  primitives stay DB-unaware and stay shared with the `fit`/`test` CLI
  commands. CLI fit doesn't want workflow DB side effects.
- **Don't try to make Layer 3 incremental.** `CREATE OR REPLACE` is
  cheap — DuckDB scans JSONL with predicate pushdown, and the whole
  catalog rebuilds in seconds even for hundreds of runs. Incremental
  ingest adds state, state adds bugs.

---

## Cross-references

- [`observability.md`](observability.md) — OTel architecture, what
  `training.fit` spans contain, `OTelTrainingCallback` lifecycle
- [`write-paths.md`](write-paths.md) — filesystem layout under `lake_root`,
  marker files, phase markers, `run_dir` content-addressing
- [`orchestration.md`](orchestration.md) — `run_pipeline` execution flow,
  `_run_one_stage` structure, retry loop (where Layer 2 hooks in)
- [`config-architecture.md`](config-architecture.md) — `ResolvedConfig.resolve`,
  `compute_identity_hash` (how `run_dir` is derived — the join key)
- Historical reference: `git show a2929d5:graphids/orchestrate/ops/catalog.py`
  for the prior DuckDB catalog implementation
