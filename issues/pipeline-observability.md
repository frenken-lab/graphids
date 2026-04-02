# No pipeline observability on headless OSC

> **Status: PARTIALLY RESOLVED** — 2026-04-01 (session 7b). `pipeline-status` CLI
> command implemented (`graphids/commands/pipeline_status.py`). Uses DagsterInstance
> batch API + phase markers + rich table. Installed nvitop, reportseff, SlurmTUI as
> complementary tools. Remaining: failure alerting (--mail-type or dagster-slack),
> `--watch` mode, structured JSON logs.

## Problem

Zero visibility into pipeline state while jobs are running:

1. **Dagster UI unusable** — OSC is headless, no browser. Dagster webserver requires port forwarding which is unreliable on OSC login nodes and forbidden on compute nodes.
2. **No structured job status** — only way to check is `squeue -u $USER` (no asset names, no DAG position) and manually tailing `slurm_logs/`.
3. **No aggregated progress** — can't tell at a glance which stages completed, which failed, which are pending. Have to cross-reference squeue job names with dagster asset names.
4. **No failure alerting** — if a stage fails at 2am, nothing notifies. Next morning requires manual sacct forensics.

## What's needed

### Minimum (CLI-native, no UI)

- `python -m graphids.orchestrate status` — reads dagster event log + sacct, prints DAG with per-asset state (pending/running/completed/failed), wall time, job ID
- Structured log output from orchestrator (JSON lines) that can be tailed and filtered
- On failure: write a summary to a known path (`{lake_root}/pipeline_status.json`) so a cron or follow-up check can report

### Nice to have

- Slack/email webhook on stage failure (dagster sensors or a simple sacct poller)
- `--watch` mode that refreshes every N seconds (like `watch squeue` but with DAG context)
- SSH tunnel recipe in lab-setup-guide for dagster UI when needed

## Current workarounds

```bash
squeue -u $USER                          # what's running
sacct -u $USER --starttime=today -o JobID,JobName,State,Elapsed,ExitCode  # what finished
tail -f slurm_logs/ablation_*.out        # orchestrator log
```
