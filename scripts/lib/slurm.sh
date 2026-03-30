#!/usr/bin/env bash
# scripts/lib/slurm.sh — SLURM helpers for job submission and resource defaults.
#
# Provides:
#   kd_slurm_account     — resolve SLURM account from .env
#   kd_require_slurm_env — die if not running inside a SLURM job
#   kd_sbatch_gpu_args   — standard GPU sbatch args string
#   kd_sbatch_cpu_args   — standard CPU sbatch args string
#   kd_submit            — submit a command via sbatch with standard args

[[ -n "${_KD_SLURM_LOADED:-}" ]] && return 0
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
_KD_SLURM_LOADED=1

kd_slurm_account() {
    # Resolve and print SLURM account. Loads .env if needed. Dies if unset.
    kd_load_env
    local acct="${KD_GAT_SLURM_ACCOUNT:-}"
    [[ -z "$acct" ]] && kd_die "KD_GAT_SLURM_ACCOUNT not set (check .env)"
    echo "$acct"
}

kd_require_slurm_env() {
    # Die if not running inside a SLURM job.
    [[ -n "${SLURM_JOB_ID:-}" ]] || kd_die "Must run under SLURM (submit via sbatch)"
}

kd_sbatch_gpu_args() {
    # Print standard GPU sbatch args. Caller can append/override.
    # Usage: sbatch $(kd_sbatch_gpu_args) --job-name=X --wrap="..."
    #        sbatch $(kd_sbatch_gpu_args "04:00:00" "32G") ...
    local acct
    acct="$(kd_slurm_account)"
    local time="${1:-08:00:00}"
    local mem="${2:-48G}"
    echo "--account=${acct} --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${mem} --time=${time} --signal=B:USR1@300"
}

kd_sbatch_cpu_args() {
    # Print standard CPU sbatch args.
    # Usage: sbatch $(kd_sbatch_cpu_args) --job-name=X --wrap="..."
    #        sbatch $(kd_sbatch_cpu_args "01:00:00" "32G") ...
    local acct
    acct="$(kd_slurm_account)"
    local time="${1:-00:30:00}"
    local mem="${2:-16G}"
    echo "--account=${acct} --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${mem} --time=${time}"
}

kd_submit() {
    # Submit a command via sbatch with standard output routing.
    # Usage: kd_submit gpu "my-job" "python -m graphids stage=autoencoder ..."
    #        kd_submit cpu "my-test" "python -m pytest tests/ -v" "--time=01:00:00"
    local mode="$1"
    local job_name="$2"
    local command="$3"
    local extra_args="${4:-}"

    local base_args
    if [[ "$mode" == "gpu" ]]; then
        base_args="$(kd_sbatch_gpu_args)"
    else
        base_args="$(kd_sbatch_cpu_args)"
    fi

    local log_dir="${KD_GAT_SLURM_LOG_DIR:-${KD_PROJECT_ROOT}/slurm_logs}"
    mkdir -p "$log_dir"

    # shellcheck disable=SC2086
    kd_exec sbatch ${base_args} \
        --job-name="kd-gat-${job_name}" \
        --output="${log_dir}/%j-${job_name}.out" \
        --error="${log_dir}/%j-${job_name}.err" \
        ${extra_args} \
        --wrap="$command"
}
