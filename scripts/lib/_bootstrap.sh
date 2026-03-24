#!/usr/bin/env bash
# scripts/lib/_bootstrap.sh — Foundation for all KD-GAT shell scripts.
# Source this first. Everything else in lib/ sources it via guard.
#
# Provides:
#   KD_PROJECT_ROOT  — absolute path to repo root
#   kd_log           — structured log to stderr
#   kd_die           — log + exit 1
#   kd_load_env      — idempotent .env sourcer
#
# Safe to source multiple times (source guard).

[[ -n "${_KD_BOOTSTRAP_LOADED:-}" ]] && return 0
_KD_BOOTSTRAP_LOADED=1

# --- PROJECT_ROOT: reuse _preamble.sh's value or derive from .git/ ---
if [[ -n "${PROJECT_ROOT:-}" ]]; then
    KD_PROJECT_ROOT="$PROJECT_ROOT"
else
    _kd_find_root() {
        local dir
        dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
        while [[ "$dir" != "/" ]]; do
            [[ -d "$dir/.git" ]] && { echo "$dir"; return 0; }
            dir="$(dirname "$dir")"
        done
        echo "/users/PAS2022/rf15/KD-GAT"  # fallback
    }
    KD_PROJECT_ROOT="$(_kd_find_root)"
    unset -f _kd_find_root
fi
export KD_PROJECT_ROOT

# --- Structured logging (to stderr, never stdout) ---
kd_log() {
    # Usage: kd_log INFO "message" [key=value ...]
    local level="$1"; shift
    local msg="$1"; shift
    local ts
    ts="$(date +%H:%M:%S)"
    local extras=""
    if [[ $# -gt 0 ]]; then
        extras=" $(printf '%s ' "$@")"
    fi
    printf '[%s] %-5s %s%s\n' "$ts" "$level" "$msg" "$extras" >&2
}

kd_die() {
    kd_log ERROR "$1"
    exit 1
}

# --- Idempotent .env loading ---
_KD_ENV_LOADED=0
kd_load_env() {
    [[ "$_KD_ENV_LOADED" -eq 1 ]] && return 0
    local env_file="${KD_PROJECT_ROOT}/.env"
    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$env_file"
        set +a
        _KD_ENV_LOADED=1
    else
        kd_log WARN ".env not found" path="$env_file"
    fi
}
