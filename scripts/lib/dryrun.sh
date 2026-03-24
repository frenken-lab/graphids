#!/usr/bin/env bash
# scripts/lib/dryrun.sh — Dry-run flag and guarded execution.
#
# Provides:
#   KD_DRY_RUN         — boolean flag (default: false)
#   kd_parse_dry_run   — extract --dry-run from args, set flag
#   kd_exec            — execute or echo command based on KD_DRY_RUN
#   kd_mkdir           — mkdir -p with dry-run awareness

[[ -n "${_KD_DRYRUN_LOADED:-}" ]] && return 0
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
_KD_DRYRUN_LOADED=1

KD_DRY_RUN="${KD_DRY_RUN:-false}"

kd_parse_dry_run() {
    # Scan args for --dry-run. Sets KD_DRY_RUN=true if found.
    for arg in "$@"; do
        if [[ "$arg" == "--dry-run" ]]; then
            KD_DRY_RUN=true
            kd_log INFO "Dry-run mode enabled"
            return 0
        fi
    done
}

kd_exec() {
    # Execute a command, or log it if dry-run mode is active.
    # Usage: kd_exec rm -rf /some/path
    if [[ "$KD_DRY_RUN" == "true" ]]; then
        kd_log DRY "$*"
        return 0
    else
        "$@"
    fi
}

kd_mkdir() {
    # mkdir -p with dry-run awareness.
    kd_exec mkdir -p "$@"
}
