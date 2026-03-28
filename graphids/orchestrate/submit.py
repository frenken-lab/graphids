"""Pipeline orchestrator: DAG-ordered SLURM submission with adaptive retry.

Each stage is a LightningCLI config YAML with a `meta:` block declaring
model_type, scale, and stage. Lightning handles everything inside the job.
This module handles everything outside: ordering, submission, polling, retry.

Stage YAML convention:
    meta:
      model_type: vgae
      scale: medium
      stage: autoencoder

    trainer:
      default_root_dir: experimentruns/dev/rf15/set_01/vgae_medium_autoencoder
      ...
"""

from __future__ import annotations

import subprocess
import time as _time
from pathlib import Path

import structlog
import yaml

from graphids.config.constants import PROJECT_ROOT, SLURM_ACCOUNT, STAGE_DEPENDENCIES, PIPELINE_YAML
from .resources import get_failure_reactions, get_resources, scale_resources

log = structlog.get_logger()

_STAGE_ORDER = list(PIPELINE_YAML["stages"].keys())
_TERMINAL = frozenset({"COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "CANCELLED", "PREEMPTED"})


class Pipeline:
    """DAG orchestrator for LightningCLI stages on SLURM.

    Each stage config YAML must have a `meta:` block (model_type, scale, stage)
    and a `trainer.default_root_dir`. Lightning handles training, checkpointing,
    and resume via --ckpt_path. This class handles ordering, submission, and retry.
    """

    def __init__(self, config_dir: Path, *, datasets: list[str] | None = None,
                 seeds: list[int] | None = None, max_retries: int = 2,
                 poll_interval: int = 300, dry_run: bool = False):
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.stages: list[dict] = []
        self._jobs: dict[str, int] = {}  # name → slurm job id

        for f in sorted(config_dir.glob("*.yaml")):
            raw = yaml.safe_load(f.read_text())
            root_dir = raw.get("trainer", {}).get("default_root_dir", "")
            dataset = raw.get("data", {}).get("init_args", {}).get("dataset", "")
            seed = raw.get("seed_everything", 42)

            # Parse model_type/scale/stage from filename convention:
            # {model_type}_{scale}_{stage}_{dataset}_s{seed}.yaml
            parts = f.stem.split("_")
            if len(parts) < 3:
                log.warning("skip_unparseable", file=f.name,
                            hint="expected: {model}_{scale}_{stage}_*.yaml")
                continue
            model_type, scale, stage = parts[0], parts[1], parts[2]

            s = {
                "name": f.stem,
                "path": f,
                "model_type": model_type,
                "scale": scale,
                "stage": stage,
                "dataset": dataset,
                "seed": seed,
                "root_dir": root_dir,
                "status": "pending",
                "retries": 0,
            }
            if datasets and s["dataset"] not in datasets:
                continue
            if seeds and s["seed"] not in seeds:
                continue
            self.stages.append(s)

        # Dedup by root_dir (shared upstream stages)
        seen: dict[str, dict] = {}
        for s in self.stages:
            key = s["root_dir"] or s["name"]
            if key in seen:
                log.info("dedup", skipped=s["name"], same_as=seen[key]["name"])
            else:
                seen[key] = s
        self.stages = sorted(seen.values(), key=lambda s: _STAGE_ORDER.index(s["stage"]) if s["stage"] in _STAGE_ORDER else 99)

    def _is_done(self, s: dict) -> bool:
        d = Path(s["root_dir"])
        return d.is_dir() and (d / "best_model.ckpt").exists()

    def _ckpt_arg(self, s: dict) -> str:
        """If last.ckpt exists, return --ckpt_path flag for Lightning auto-resume."""
        last = Path(s["root_dir"]) / "last.ckpt"
        return f" --ckpt_path {last}" if last.exists() else ""

    def _submit(self, s: dict) -> int:
        res = get_resources(s["model_type"], s["scale"], s["stage"])
        args = [
            "sbatch",
            f"--partition={res.partition}", f"--time={res.time}",
            f"--mem={res.mem}", f"--cpus-per-task={res.cpus_per_task}",
            "--signal=B:USR1@300", f"--account={SLURM_ACCOUNT}",
            f"--output={PROJECT_ROOT}/slurm_logs/{s['name']}_%j.out",
            f"--error={PROJECT_ROOT}/slurm_logs/{s['name']}_%j.err",
        ]
        if res.gres:
            args.append(f"--gres={res.gres}")

        cmd = f"python -m graphids fit --config {s['path'].resolve()}{self._ckpt_arg(s)}"
        script = (
            "#!/bin/bash\n"
            f"source {PROJECT_ROOT}/scripts/slurm/_preamble.sh\n"
            f"{cmd}\n"
            f"source {PROJECT_ROOT}/scripts/slurm/_epilog.sh\n"
        )
        if self.dry_run:
            log.info("dry_run", cmd=cmd, mem=res.mem, time=res.time)
            return 0

        # Fix 2+3: handle sbatch failure gracefully, parse job ID safely
        r = subprocess.run(
            [*args, "--wrap", script],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            log.error("sbatch_failed", stage=s["name"], stderr=r.stderr.strip())
            return -1

        import re
        m = re.search(r"(\d+)\s*$", r.stdout.strip())
        if not m:
            log.error("sbatch_parse_failed", stage=s["name"], stdout=r.stdout.strip())
            return -1
        return int(m.group(1))

    @staticmethod
    def _check_job(job_id: int) -> str:
        r = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=JobID,State",
             "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log.warning("sacct_failed", job=job_id, stderr=r.stderr.strip())
            return "UNKNOWN"
        # Parse parent job state (skip .batch/.extern substeps)
        for line in r.stdout.strip().split("\n"):
            parts = line.strip().split("|")
            if len(parts) >= 2 and "." not in parts[0]:
                return parts[1].strip()
        return "UNKNOWN"

    def _wait(self, s: dict) -> str:
        unknown_count = 0
        while True:
            status = self._check_job(self._jobs[s["name"]])
            if status in _TERMINAL:
                s["status"] = status
                return status
            if status == "UNKNOWN":
                unknown_count += 1
                if unknown_count > 5:
                    log.error("sacct_unreachable", stage=s["name"],
                              job=self._jobs[s["name"]])
                    s["status"] = "UNKNOWN"
                    return "UNKNOWN"
            else:
                unknown_count = 0
            _time.sleep(self.poll_interval)

    def _deps_ready(self, s: dict) -> bool:
        """Wait for upstream deps. Returns False if a dep failed."""
        for dep_model, dep_stage in STAGE_DEPENDENCIES.get(s["stage"], []):
            for dep in self.stages:
                if dep["stage"] == dep_stage and dep["model_type"] == dep_model and dep["dataset"] == s["dataset"]:
                    if dep["name"] in self._jobs and dep["status"] not in _TERMINAL:
                        log.info("waiting", stage=s["name"], dep=dep["name"])
                        self._wait(dep)
                    if dep["status"] not in ("COMPLETED", "pending"):
                        if not self._is_done(dep):
                            log.error("dep_failed", stage=s["name"], dep=dep["name"], dep_status=dep["status"])
                            return False
        return True

    def run(self):
        """Execute pipeline. Blocks until all stages complete or fail."""
        log.info("start", stages=len(self.stages), dry_run=self.dry_run)

        for s in self.stages:
            if self._is_done(s):
                s["status"] = "COMPLETED"
                log.info("skip_done", stage=s["name"])
                continue

            if not self._deps_ready(s):
                s["status"] = "DEP_FAILED"
                continue

            # Submit + wait + retry
            job_id = self._submit(s)
            if job_id < 0:
                s["status"] = "SUBMIT_FAILED"
                log.error("submit_failed", stage=s["name"])
                continue
            self._jobs[s["name"]] = job_id
            log.info("submitted", stage=s["name"], job=job_id,
                     model=s["model_type"], dataset=s["dataset"])

            if self.dry_run:
                s["status"] = "COMPLETED"
                continue

            status = self._wait(s)
            reactions = get_failure_reactions()
            while status in reactions and s["retries"] < min(reactions[status].get("max_retries", 0), self.max_retries):
                s["retries"] += 1
                old_res = get_resources(s["model_type"], s["scale"], s["stage"])
                new_res = scale_resources(old_res, status)
                log.info("retry", stage=s["name"], reason=status, attempt=s["retries"],
                         mem=f"{old_res.mem}→{new_res.mem}", time=f"{old_res.time}→{new_res.time}")
                # Re-submit reads ckpt_path again (last.ckpt may exist now from failed run)
                self._jobs[s["name"]] = self._submit(s)
                status = self._wait(s)

            if status != "COMPLETED":
                log.error("failed", stage=s["name"], status=status, retries=s["retries"])

        done = [s for s in self.stages if s["status"] == "COMPLETED"]
        log.info("done", completed=len(done), total=len(self.stages))

    def summary(self) -> list[dict]:
        return [{"name": s["name"], "status": s["status"], "retries": s["retries"],
                 "job": self._jobs.get(s["name"])} for s in self.stages]


def run_pipeline(**kwargs):
    Pipeline(**kwargs).run()
