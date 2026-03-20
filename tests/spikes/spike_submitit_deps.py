"""Spike 4: Do SLURM dependency chains work through submitit's native dependency param?

Run from login node: .venv/bin/python tests/spikes/spike_submitit_deps.py
"""
import submitit

def step(name):
    import os, time
    time.sleep(5)
    return {"step": name, "node": os.environ.get("SLURMD_NODENAME", "unknown")}

executor = submitit.SlurmExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(
    time=5, partition="cpu", account="PAS1266",
    mem="4G", cpus_per_task=1,
)

print("Submitting job 1...")
job1 = executor.submit(step, "first")
print(f"Job 1 ID: {job1.job_id}")

# Submit job 2 with afterok dependency on job 1
executor.update_parameters(dependency=f"afterok:{job1.job_id}")
print(f"Submitting job 2 (depends on {job1.job_id})...")
job2 = executor.submit(step, "second")
print(f"Job 2 ID: {job2.job_id}")

print("Waiting for results...")
try:
    r1 = job1.result()
    print(f"Job 1 result: {r1}")
    r2 = job2.result()
    print(f"Job 2 result: {r2}")
    print("Spike 4 PASSED: dependency chain worked")
except Exception as e:
    print(f"Spike 4 FAILED: {e}")
    for j, label in [(job1, "Job 1"), (job2, "Job 2")]:
        try:
            print(f"{label} stderr: {j.stderr()}")
        except Exception:
            pass
