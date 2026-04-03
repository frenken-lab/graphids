# Pipeline observability on headless OSC

> **Status: RESOLVED** — 2026-04-03 (session 15)

## Resolved

- `python -m graphids pipeline-status` — dagster instance batch API + phase markers +
  rich table + JSON output mode (`commands/pipeline_status.py`)
- **Orchestrator event log** — structlog JSONL per run at
  `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`. Events: `orchestrator_init`,
  `asset_start`, `asset_skip`, `resource_scaled`, `asset_complete`, `asset_failed`,
  `submitted`, `slurm_poll`, `config_resolved`. Configured in `definitions.py`.
- **`pipeline-status --log`** — CLI reader for orchestrator JSONL with filters
  (`failures`, `retries`, `completions`, `submissions`, `polls`) and `--follow` mode
- **`PYTHONUNBUFFERED=1`** in `_preamble.sh` — fixes turm real-time log tailing
- **Spec file preservation** — SLURM spec JSONs kept in `{SLURM_LOG_DIR}/specs/`
- nvitop, reportseff, SlurmTUI installed as complementary monitoring tools
- DeviceStatsMonitor logs CUDA memory stats per step
- `_epilog.sh` prints sacct summary
- sacct reconciliation in pipeline-status (session 13)

## Remaining

| Item | Priority | Notes |
|------|----------|-------|
| `--watch` mode for `pipeline-status` | Low | Auto-refresh with DAG context |
| Failure alerting | Low | `--mail-type=END,FAIL` or dagster-slack sensor |
