#!/usr/bin/env bash
# scripts/lib/validation.sh — Checkpoint and run directory validation.
#
# Provides:
#   kd_run_dir            — build canonical run directory path
#   kd_check_checkpoint   — verify a file exists, log result (returns 0/1)
#   kd_require_checkpoint — check + die on failure
#   kd_run_complete       — check if run dir has best_model.pt or metrics.json

[[ -n "${_KD_VALIDATION_LOADED:-}" ]] && return 0
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
_KD_VALIDATION_LOADED=1

kd_run_dir() {
    # Build canonical run directory path.
    # Usage: kd_run_dir <lake_root> <dataset> <model_type> <scale> <stage> [seed] [suffix]
    # Output: {lake_root}/{dataset}/{model}_{scale}_{stage}[_{suffix}]/seed_{N}
    # suffix: identity hash (e.g., "a3f2b1c9") or legacy aux label
    local lake_root="$1" dataset="$2" model="$3" scale="$4" stage="$5"
    local seed="${6:-42}" suffix="${7:-}"
    local run_name="${model}_${scale}_${stage}"
    [[ -n "$suffix" ]] && run_name="${run_name}_${suffix}"
    echo "${lake_root}/${dataset}/${run_name}/seed_${seed}"
}

kd_check_checkpoint() {
    # Check if a file exists. Returns 0 if found, 1 if missing. Logs either way.
    # Usage: kd_check_checkpoint <path> [description]
    local ckpt="$1"
    local desc="${2:-checkpoint}"
    if [[ -f "$ckpt" ]]; then
        kd_log INFO "Verified: ${desc}" path="$ckpt"
        return 0
    else
        kd_log WARN "Missing: ${desc}" path="$ckpt"
        return 1
    fi
}

kd_require_checkpoint() {
    # Like kd_check_checkpoint but dies on failure.
    kd_check_checkpoint "$@" || kd_die "Required ${2:-checkpoint} not found: $1"
}

kd_run_complete() {
    # Check if a run directory has completion artifacts.
    # Returns 0 if complete, 1 if orphaned/incomplete.
    local run_dir="$1"
    [[ -f "${run_dir}/best_model.pt" ]] || [[ -f "${run_dir}/metrics.json" ]]
}
