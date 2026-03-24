#!/usr/bin/env bash
# scripts/lib/datasets.sh — Canonical dataset list and iteration helpers.
#
# Provides:
#   KD_ALL_DATASETS    — array of all dataset names (from datasets.yaml)
#   kd_parse_datasets  — parse CLI args into dataset list, default=all
#   kd_each_dataset    — run a callback for each dataset

[[ -n "${_KD_DATASETS_LOADED:-}" ]] && return 0
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
_KD_DATASETS_LOADED=1

# --- Canonical dataset list (read from datasets.yaml, fallback hardcoded) ---
_kd_read_datasets() {
    local yaml_file="${KD_PROJECT_ROOT}/graphids/config/datasets.yaml"
    if [[ -f "$yaml_file" ]] && command -v python3 &>/dev/null; then
        python3 -c "
import yaml, pathlib
ds = yaml.safe_load(pathlib.Path('${yaml_file}').read_text())
print(' '.join(ds.keys()))
" 2>/dev/null && return 0
    fi
    # Fallback (matches datasets.yaml as of 2026-03-24)
    echo "hcrl_ch hcrl_sa set_01 set_02 set_03 set_04"
}

# shellcheck disable=SC2207
KD_ALL_DATASETS=($(_kd_read_datasets))
unset -f _kd_read_datasets
export KD_ALL_DATASETS

kd_parse_datasets() {
    # Extract dataset names from args (skips --flags). Defaults to all if none given.
    # Usage: read -ra ds <<< "$(kd_parse_datasets "$@")"
    local result=()
    for arg in "$@"; do
        [[ "$arg" == --* ]] && continue
        result+=("$arg")
    done
    if [[ ${#result[@]} -eq 0 ]]; then
        echo "${KD_ALL_DATASETS[*]}"
    else
        echo "${result[*]}"
    fi
}

kd_each_dataset() {
    # Run a callback for each dataset.
    # Usage: kd_each_dataset my_func [dataset1 dataset2 ...]
    #        If no datasets given, iterates all.
    local callback="$1"; shift
    local datasets
    if [[ $# -gt 0 ]]; then
        datasets=("$@")
    else
        datasets=("${KD_ALL_DATASETS[@]}")
    fi
    for ds in "${datasets[@]}"; do
        "$callback" "$ds"
    done
}
