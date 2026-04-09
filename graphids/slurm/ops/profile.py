"""Post-hoc SLURM resource profiler: sacct aggregation.

GPU/VRAM metrics are handled by wandb (system metrics) and DeviceStatsMonitor
(CUDA allocator stats). This tool covers what those can't: SLURM accounting
(RSS, CPU efficiency, wall time, mem efficiency) across completed jobs.

Usage (via __main__.py):
    python -m graphids profile 46152810 46152812
    python -m graphids profile --since 2026-03-28
    python -m graphids profile --json 46152810
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field

from graphids.slurm.core.accounting import parse_elapsed, sacct_query


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SLURMStats:
    job_id: str = ""
    job_name: str = ""
    state: str = ""
    elapsed: str = ""
    max_rss_gib: float = 0.0
    req_mem_gib: float = 0.0
    mem_efficiency_pct: float = 0.0
    alloc_cpus: int = 0
    cpu_efficiency_pct: float = 0.0


@dataclass
class JobProfile:
    job_id: str = ""
    stage: str = ""
    dataset: str = ""
    slurm: SLURMStats = field(default_factory=SLURMStats)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_mem(s: str) -> float:
    """Parse sacct memory string like '12.55G' or '25165024K' to float GiB."""
    s = s.strip()
    if not s:
        return 0.0
    if s.endswith("G"):
        try:
            return float(s[:-1])
        except ValueError:
            return 0.0
    if s.endswith("K"):
        try:
            return float(s[:-1]) / (1024 * 1024)
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# SLURM sacct
# ---------------------------------------------------------------------------

def get_sacct_stats(job_ids: list[str], *, cluster: str | None = None) -> dict[str, SLURMStats]:
    """Query sacct for resource usage."""
    if not job_ids:
        return {}
    stdout = sacct_query(
        job_ids, "JobID,JobName,State,Elapsed,MaxRSS,ReqMem,AllocCPUS,TRESUsageInTot",
        cluster=cluster,
    )
    if not stdout:
        return {}

    stats: dict[str, SLURMStats] = {}
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 8:
            continue
        raw_id, job_name, state, elapsed, max_rss, req_mem, alloc_cpus, tres = fields
        base_id = raw_id.split(".")[0]

        if base_id not in stats:
            stats[base_id] = SLURMStats(job_id=base_id)
        s = stats[base_id]

        if raw_id == base_id:
            s.state = state.split(" ")[0]
            s.elapsed = elapsed
            s.job_name = job_name
            s.req_mem_gib = _parse_mem(req_mem)
            try:
                s.alloc_cpus = int(alloc_cpus)
            except ValueError:
                pass
        elif raw_id.endswith(".batch"):
            s.max_rss_gib = _parse_mem(max_rss)
            for kv in tres.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k == "cpu":
                        cpu_sec = parse_elapsed(v) or 0.0
                        wall_sec = (parse_elapsed(s.elapsed) or 0.0) if s.elapsed else 0.0
                        if wall_sec > 0 and s.alloc_cpus > 0:
                            s.cpu_efficiency_pct = round(
                                cpu_sec / (wall_sec * s.alloc_cpus) * 100, 1
                            )

    for s in stats.values():
        if s.req_mem_gib > 0:
            s.mem_efficiency_pct = round(s.max_rss_gib / s.req_mem_gib * 100, 1)
    return stats


def discover_jobs_since(since: str, *, cluster: str | None = None) -> list[str]:
    """Find job IDs from sacct since a given time."""
    user = subprocess.check_output(["whoami"]).decode().strip()
    cmd = [
        "sacct", "-u", user, f"--starttime={since}",
        "--parsable2", "--noheader", "--format=JobID,JobName", "-X",
    ]
    if cluster:
        cmd.extend(["-M", cluster])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [
        line.split("|")[0]
        for line in result.stdout.strip().split("\n")
        if line and "|" in line
    ]


# ---------------------------------------------------------------------------
# Job name metadata
# ---------------------------------------------------------------------------

_JOB_NAME_RE = re.compile(
    r"^(?P<stage>\w+?)_(?P<identity>[0-9a-f]{8})(?:_kd)?_(?P<dataset>\w+)_s(?P<seed>\d+)$"
)


def _parse_job_name(job_name: str) -> dict[str, str]:
    """Extract stage/dataset/seed from job name like autoencoder_bf355e79_set_01_s42."""
    m = _JOB_NAME_RE.match(job_name)
    if m:
        return m.groupdict()
    return {}


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------

def profile_jobs(job_ids: list[str], *, cluster: str | None = None) -> list[JobProfile]:
    """Build profiles for SLURM job IDs."""
    sacct = get_sacct_stats(job_ids, cluster=cluster)
    profiles = []
    for jid in job_ids:
        slurm = sacct.get(jid, SLURMStats(job_id=jid))
        meta = _parse_job_name(slurm.job_name)
        profiles.append(JobProfile(
            job_id=jid,
            stage=meta.get("stage", ""),
            dataset=meta.get("dataset", ""),
            slurm=slurm,
        ))
    return profiles


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(profiles: list[JobProfile]) -> str:
    """Compact table output with recommendations."""
    lines: list[str] = [""]
    hdr = (
        f"{'JobID':<12} {'State':<12} {'Elapsed':>8} {'RSS':>6} {'ReqMem':>6} "
        f"{'Mem%':>5} {'CPU%':>5} {'Stage'}"
    )
    lines += ["=== Job Summary ===", "", hdr, "-" * len(hdr)]
    for p in profiles:
        s = p.slurm
        label = p.stage or s.job_name
        if p.dataset:
            label = f"{p.dataset}/{label}"
        lines.append(
            f"{s.job_id:<12} {s.state:<12} {s.elapsed:>8} {s.max_rss_gib:>4.1f}G "
            f"{s.req_mem_gib:>4.0f}G {s.mem_efficiency_pct:>4.0f}% "
            f"{s.cpu_efficiency_pct:>4.0f}%  {label}"
        )
    lines.append("")

    rss_vals = [p.slurm.max_rss_gib for p in profiles if p.slurm.max_rss_gib > 0]
    req_mems = [p.slurm.req_mem_gib for p in profiles if p.slurm.req_mem_gib > 0]
    if rss_vals and req_mems:
        suggested = int(max(rss_vals) * 1.3) + 1
        current = int(max(req_mems))
        if suggested < current * 0.7:
            lines.append("=== Recommendations ===")
            lines.append(f"  --mem={suggested}G  (currently {current}G, peak {max(rss_vals):.1f}G)")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> None:
    """CLI entry point -- called from __main__.py."""
    parser = argparse.ArgumentParser(description="Profile SLURM job resources")
    parser.add_argument("job_ids", nargs="*", help="SLURM job IDs")
    parser.add_argument("--since", help="All jobs since (e.g. 2026-03-28)")
    parser.add_argument("--cluster", "-M", help="SLURM cluster (e.g. ascend)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    job_ids = list(args.job_ids)
    if args.since:
        job_ids.extend(discover_jobs_since(args.since, cluster=args.cluster))
    if not job_ids:
        parser.error("Provide job IDs or --since")

    seen: set[str] = set()
    unique = [j for j in job_ids if j not in seen and not seen.add(j)]  # type: ignore[func-returns-value]

    profiles = profile_jobs(unique, cluster=args.cluster)
    if args.json:
        print(json.dumps([asdict(p) for p in profiles], indent=2))
    else:
        print(format_table(profiles))
