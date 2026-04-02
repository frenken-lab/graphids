# Pipeline observability on headless OSC

> **Status: PARTIALLY RESOLVED** — 2026-04-01 (session 7b)

## Resolved

- `python -m graphids pipeline-status` — dagster instance batch API + phase markers +
  rich table + JSON output mode (`commands/pipeline_status.py`)
- nvitop, reportseff, SlurmTUI installed as complementary monitoring tools
- DeviceStatsMonitor logs CUDA memory stats per step
- `_epilog.sh` prints sacct summary

## Remaining

| Item | Priority | Notes |
|------|----------|-------|
| `--watch` mode for `pipeline-status` | Low | Auto-refresh with DAG context |
| Structured JSON logs | Low | `JSONRenderer` when `SLURM_JOB_ID` set, enables `jq` |
| Failure alerting | Low | `--mail-type=END,FAIL` or dagster-slack sensor |
