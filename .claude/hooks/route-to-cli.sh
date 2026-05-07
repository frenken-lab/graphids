#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash). Blocks bash commands that bypass the
# canonical graphids CLIs for MLflow queries / SLURM submission. To
# override, the caller must set both BYPASS_ROUTE_TO_CLI=1 and a non-
# empty BYPASS_JUSTIFICATION; the bypass is appended to
# .claude/bypasses.jsonl. See .claude/hooks/README.md for the policy
# and what each canonical tool is.
set -euo pipefail

input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command // ""')
[[ -z "$cmd" ]] && exit 0

# Match → category + canonical hint
category=""
canonical=""
if echo "$cmd" | grep -qiE 'python.*-c.*\b(mlflow|MlflowClient)\b|python\s*<<.*\b(mlflow|MlflowClient)\b|\bmlflow\s+(runs|experiments)\s+|sqlite3[^|]*mlflow\.db'; then
    category="mlflow_query"
    canonical='python scripts/results.py --view <name>   (profiles in configs/result_views.yml; add YAML, not Python)'
elif echo "$cmd" | grep -qE '(^|;|&&|\|\|)\s*sbatch\b'; then
    category="slurm_submit"
    canonical='gx submit --plan FILE --row-name NAME --cluster X      |   gx plans submit --plan FILE --cluster X'
elif echo "$cmd" | grep -qE 'python\s+-m\s+graphids\s+(fit|test|train)\b'; then
    category="dead_verbs"
    canonical='gx exec --row "<json>" [--ckpt-path X]   |   gx submit --plan F --row-name N   (every job is a row)'
else
    exit 0
fi

# Bypass path
if [[ "${BYPASS_ROUTE_TO_CLI:-}" == "1" ]]; then
    reason="${BYPASS_JUSTIFICATION:-}"
    if [[ -z "$reason" ]]; then
        cat >&2 <<EOF
[route-to-cli] BLOCKED [$category]: BYPASS_ROUTE_TO_CLI=1 set, but BYPASS_JUSTIFICATION is empty.
Set both to override: BYPASS_ROUTE_TO_CLI=1 BYPASS_JUSTIFICATION="<one-line reason canonical insufficient>"
EOF
        exit 1
    fi
    log="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/bypasses.jsonl"
    mkdir -p "$(dirname "$log")"
    ts=$(date -Iseconds)
    cmd_short=$(printf '%s' "$cmd" | head -c 400 | tr '\n' ' ')
    printf '{"ts":"%s","category":"%s","command":%s,"justification":%s}\n' \
        "$ts" "$category" "$(jq -Rs <<<"$cmd_short")" "$(jq -Rs <<<"$reason")" >>"$log"
    cat >&2 <<EOF
[route-to-cli] BYPASS LOGGED [$category] → $log
You MUST acknowledge this bypass and the justification in your response to the user.
EOF
    exit 0
fi

# Block path
cat >&2 <<EOF
[route-to-cli] BLOCKED [$category]. Use the canonical tool:
    $canonical

If canonical is genuinely insufficient, prefix the command with both:
    BYPASS_ROUTE_TO_CLI=1 BYPASS_JUSTIFICATION="<one-line reason>" <your-command>
The bypass + reason is appended to .claude/bypasses.jsonl. See .claude/hooks/README.md.
EOF
exit 1
