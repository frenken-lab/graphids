# Project hooks

## `route-to-cli.sh` — PreToolUse on Bash

Blocks bash calls that bypass canonical graphids tooling. Three trigger categories:

| Category | What's blocked | Canonical tool |
|---|---|---|
| `mlflow_query` | inline `python -c` / heredoc python with `mlflow` or `MlflowClient`; direct `mlflow runs/experiments` CLI; `sqlite3` against `mlflow.db` | `python scripts/results.py --view <name>` (profiles in `configs/result_views.yml` — add YAML, not Python, when extending) |
| `slurm_submit` | `sbatch` invoked directly | `gx submit --plan F --row-name N --cluster X` (single row) or `gx plans submit --plan F --cluster X` (bulk). Direct sbatch breaks chassis-invariants (drift resistance, SIGUSR2, MLflow lifecycle). |
| `dead_verbs` | `python -m graphids {fit,test,train}` | `gx exec --row '<json>'` or `gx submit --plan F --row-name N`. Those verbs were removed; every job is a row (chassis-invariants §1). |

### Bypass

When the canonical tool is genuinely insufficient (one-shot debug query, ops job outside the chassis, etc.), prefix the command with both env vars:

```bash
BYPASS_ROUTE_TO_CLI=1 BYPASS_JUSTIFICATION="reading metric key not in any view" \
    python -c "from mlflow.tracking import MlflowClient; ..."
```

The hook appends one JSONL line to `.claude/bypasses.jsonl`:
```json
{"ts": "...", "category": "mlflow_query", "command": "...", "justification": "..."}
```

The hook also prints to stderr `You MUST acknowledge this bypass…`. **Claude is required to mention the bypass and reason in its assistant response so you see it in-channel** in addition to the audit log.

To audit:
```bash
tail .claude/bypasses.jsonl
jq . .claude/bypasses.jsonl   # pretty
```

### Disabling per-session

Comment out the hook block in `.claude/settings.json` or move it to `.claude/settings.local.json` for a personal override.
