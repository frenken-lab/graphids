"""Spike 1: Can submitit submit a Python callable to SLURM and get a result back?

Run from login node: .venv/bin/python tests/spikes/spike_submitit_basic.py
"""
import submitit

def square(x):
    import os
    return {
        "result": x ** 2,
        "hostname": os.environ.get("SLURMD_NODENAME", "unknown"),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    }

executor = submitit.SlurmExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(
    time=5, partition="cpu", account="PAS1266",
    mem="4G", cpus_per_task=1,
    setup=["source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"],
)

print("Submitting spike_submitit_basic...")
job = executor.submit(square, 42)
print(f"Job ID: {job.job_id}")
print(f"State: {job.state}")
print("Waiting for result (this blocks)...")

try:
    result = job.result()
    print(f"Spike 1 PASSED: {result}")
except Exception as e:
    print(f"Spike 1 FAILED: {e}")
    print(f"Stderr: {job.stderr()}")
    print(f"Stdout: {job.stdout()}")
