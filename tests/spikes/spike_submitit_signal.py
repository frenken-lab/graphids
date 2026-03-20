"""Spike 3: Does SIGUSR1 reach our Python handler with submitit (Option A: no submitit requeue)?

Run from login node: .venv/bin/python tests/spikes/spike_submitit_signal.py
Submits a 2-min job. signal_delay_s=60 sends USR1 60s before timeout.
Job sleeps 90s, so USR1 should arrive at ~60s.
"""
import submitit

def signal_test():
    import os, signal, time

    received = []

    def handler(signum, frame):
        received.append({"signal": "USR1", "time": time.time()})
        print(f"[SIGNAL] Received SIGUSR1 at {time.time()}")

    # Check what submitit registered before us
    original = signal.getsignal(signal.SIGUSR1)
    print(f"[INFO] Original USR1 handler: {original}")

    signal.signal(signal.SIGUSR1, handler)
    print(f"[INFO] PID={os.getpid()}, waiting for SIGUSR1...")

    start = time.time()
    while time.time() - start < 90:
        time.sleep(1)

    return {
        "received": received,
        "pid": os.getpid(),
        "node": os.environ.get("SLURMD_NODENAME", "unknown"),
        "elapsed": time.time() - start,
    }

executor = submitit.SlurmExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(
    time=2, partition="cpu", account="PAS1266",
    mem="4G", cpus_per_task=1,
    signal_delay_s=60,  # send USR1 60s before the 2-min timeout
)

print("Submitting signal test (2-min timeout, USR1 at ~60s)...")
job = executor.submit(signal_test)
print(f"Job ID: {job.job_id}")
print("Waiting for result (expect ~90s)...")

try:
    result = job.result()
    if result["received"]:
        print(f"Spike 3 PASSED: SIGUSR1 received. {result}")
    else:
        print(f"Spike 3 INCONCLUSIVE: no signal received. {result}")
except Exception as e:
    print(f"Spike 3 result (may be expected if job timed out): {e}")
    try:
        print(f"Stdout:\n{job.stdout()}")
        print(f"Stderr:\n{job.stderr()}")
    except Exception:
        pass
