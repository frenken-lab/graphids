#!/usr/bin/env python3
"""Profile SLURM job resource usage from sacct + nvidia-smi gpu_stats.csv.

Usage:
    python scripts/profile_jobs.py 45982486 45982490       # specific jobs
    python scripts/profile_jobs.py --since 2026-03-25      # all submitit jobs since
    python scripts/profile_jobs.py --timeline 45985737      # deep GPU phase analysis
    python scripts/profile_jobs.py --plot gpu.png 45985737 45985746  # timeline + PNG
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


GiB = 1 << 30
V100_VRAM = 16 * GiB


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VRAMStats:
    alloc_current_mean_gib: float = 0.0
    alloc_peak_gib: float = 0.0
    reserved_peak_gib: float = 0.0
    utilization_pct: float = 0.0
    num_ooms: int = 0
    num_samples: int = 0


@dataclass
class SLURMStats:
    job_id: str = ""
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
    run_dir: str = ""
    model: str = ""
    stage: str = ""
    dataset: str = ""
    slurm: SLURMStats = field(default_factory=SLURMStats)
    vram: VRAMStats = field(default_factory=VRAMStats)
    gpu_samples: int = 0
    gpu_util_mean: float = 0.0
    gpu_util_max: float = 0.0
    vram_used_mean_gib: float = 0.0
    vram_used_max_gib: float = 0.0
    power_mean_w: float = 0.0
    temp_max_c: float = 0.0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_elapsed(s: str) -> float:
    """Parse HH:MM:SS or D-HH:MM:SS to seconds."""
    parts = s.replace("-", ":").split(":")
    nums = [float(p) for p in parts]
    return sum(v * m for v, m in zip(reversed(nums), [1, 60, 3600, 86400]))


def _parse_mem(s: str) -> float:
    """Parse sacct memory string like '12.55G' to float GiB."""
    s = s.strip()
    if not s:
        return 0.0
    if s.endswith("G"):
        try:
            return float(s[:-1])
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_gpu_timeline(csv_path: Path) -> list[dict]:
    """Parse nvidia-smi gpu_stats.csv into list of sample dicts."""
    rows = []
    try:
        for line in csv_path.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "time": parts[0],
                    "gpu_util": int(parts[1]),
                    "mem_used_mb": int(parts[4]),
                    "mem_total_mb": int(parts[3]),
                    "power_w": float(parts[6]),
                    "temp_c": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    except OSError:
        pass
    return rows


def _gpu_summary_from_timeline(rows: list[dict]) -> dict:
    """Compute summary stats from timeline rows."""
    if not rows:
        return {}
    n = len(rows)
    gu = [r["gpu_util"] for r in rows]
    mu = [r["mem_used_mb"] for r in rows]
    pw = [r["power_w"] for r in rows]
    return {
        "num_samples": n,
        "gpu_util_mean": round(sum(gu) / n, 1),
        "gpu_util_max": max(gu),
        "vram_used_mean_gib": round(sum(mu) / n / 1024, 2),
        "vram_used_max_gib": round(max(mu) / 1024, 2),
        "power_mean_w": round(sum(pw) / n, 1),
        "temp_max_c": max(r["temp_c"] for r in rows),
    }


def _extract_job_metadata(log_dir: Path, jid: str) -> dict[str, str]:
    """Extract dataset/model/stage/scale/run_dir from structlog lines."""
    out = log_dir / f"{jid}_0_log.out"
    meta: dict[str, str] = {}
    if not out.exists():
        return meta
    for line in out.read_text().splitlines():
        if "training_complete" in line or "node_budget_computed" in line:
            for key in ("dataset", "model", "stage", "scale", "run_dir"):
                m = re.search(rf"{key}=(\S+)", line)
                if m:
                    meta[key] = m.group(1)
    return meta


# ---------------------------------------------------------------------------
# SLURM sacct
# ---------------------------------------------------------------------------

def get_sacct_stats(job_ids: list[str]) -> dict[str, SLURMStats]:
    """Query sacct for resource usage."""
    if not job_ids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(job_ids), "--parsable2", "--noheader",
        "--format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocCPUS,TRESUsageInTot",
        "--units=G",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sacct error: {result.stderr}", file=sys.stderr)
        return {}

    stats: dict[str, SLURMStats] = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 7:
            continue
        raw_id, state, elapsed, max_rss, req_mem, alloc_cpus, tres = fields
        base_id = raw_id.split(".")[0]

        if base_id not in stats:
            stats[base_id] = SLURMStats(job_id=base_id)
        s = stats[base_id]

        if raw_id == base_id:
            s.state = state.split(" ")[0]
            s.elapsed = elapsed
            s.req_mem_gib = _parse_mem(req_mem)
            try:
                s.alloc_cpus = int(alloc_cpus)
            except ValueError:
                pass
        elif raw_id.endswith(".0"):
            s.max_rss_gib = _parse_mem(max_rss)
            for kv in tres.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k == "cpu":
                        cpu_sec = _parse_elapsed(v)
                        wall_sec = _parse_elapsed(s.elapsed) if s.elapsed else 0
                        if wall_sec > 0 and s.alloc_cpus > 0:
                            s.cpu_efficiency_pct = round(cpu_sec / (wall_sec * s.alloc_cpus) * 100, 1)

    for s in stats.values():
        if s.req_mem_gib > 0:
            s.mem_efficiency_pct = round(s.max_rss_gib / s.req_mem_gib * 100, 1)
    return stats


def discover_jobs_since(since: str) -> list[str]:
    """Find submitit job IDs from sacct since a given time."""
    cmd = [
        "sacct", "-u", subprocess.check_output(["whoami"]).decode().strip(),
        f"--starttime={since}", "--parsable2", "--noheader",
        "--format=JobID,JobName", "-X",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [
        line.split("|")[0]
        for line in result.stdout.strip().split("\n")
        if line and "|" in line and "submitit" in line.split("|")[1]
    ]


# ---------------------------------------------------------------------------
# VRAM from Lightning DeviceStatsMonitor
# ---------------------------------------------------------------------------

_PREFIX = "DeviceStatsMonitor.on_train_batch_end"
_ALLOC_CURRENT = f"{_PREFIX}/allocated_bytes.all.current"
_ALLOC_PEAK = f"{_PREFIX}/allocated_bytes.all.peak"
_RESERVED_PEAK = f"{_PREFIX}/reserved_bytes.all.peak"
_NUM_OOMS = f"{_PREFIX}/num_ooms"


def parse_metrics_csv(csv_path: Path) -> VRAMStats:
    """Extract VRAM stats from Lightning CSVLogger metrics.csv."""
    import csv as csvmod
    stats = VRAMStats()
    alloc_currents, alloc_peaks, reserved_peaks, ooms = [], [], [], []
    try:
        with open(csv_path) as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                if _ALLOC_CURRENT in row and row[_ALLOC_CURRENT]:
                    alloc_currents.append(float(row[_ALLOC_CURRENT]))
                if _ALLOC_PEAK in row and row[_ALLOC_PEAK]:
                    alloc_peaks.append(float(row[_ALLOC_PEAK]))
                if _RESERVED_PEAK in row and row[_RESERVED_PEAK]:
                    reserved_peaks.append(float(row[_RESERVED_PEAK]))
                if _NUM_OOMS in row and row[_NUM_OOMS]:
                    ooms.append(int(float(row[_NUM_OOMS])))
    except OSError:
        return stats

    if alloc_currents:
        n = len(alloc_currents)
        stats.num_samples = n
        stats.alloc_current_mean_gib = round(sum(alloc_currents) / n / GiB, 2)
        stats.alloc_peak_gib = round(max(alloc_peaks) / GiB, 2) if alloc_peaks else 0
        stats.reserved_peak_gib = round(max(reserved_peaks) / GiB, 2) if reserved_peaks else 0
        stats.utilization_pct = round(stats.reserved_peak_gib / (V100_VRAM / GiB) * 100, 1)
        stats.num_ooms = max(ooms) if ooms else 0
    return stats


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------

def profile_jobs(job_ids: list[str], log_root: Path) -> list[JobProfile]:
    """Build profiles for SLURM job IDs."""
    sacct = get_sacct_stats(job_ids)
    profiles = []
    for jid in job_ids:
        meta = _extract_job_metadata(log_root / jid, jid)
        p = JobProfile(
            job_id=jid,
            model=meta.get("model", ""),
            stage=meta.get("stage", ""),
            dataset=meta.get("dataset", ""),
            run_dir=meta.get("run_dir", ""),
            slurm=sacct.get(jid, SLURMStats(job_id=jid)),
        )
        # Lightning VRAM stats
        if p.run_dir:
            csv_path = Path(p.run_dir) / "lightning_logs" / "version_0" / "metrics.csv"
            if csv_path.exists():
                p.vram = parse_metrics_csv(csv_path)
        # nvidia-smi GPU stats
        rows = _parse_gpu_timeline(log_root / jid / "gpu_stats.csv")
        gs = _gpu_summary_from_timeline(rows)
        if gs:
            p.gpu_samples = gs["num_samples"]
            p.gpu_util_mean = gs["gpu_util_mean"]
            p.gpu_util_max = gs["gpu_util_max"]
            p.vram_used_mean_gib = gs["vram_used_mean_gib"]
            p.vram_used_max_gib = gs["vram_used_max_gib"]
            p.power_mean_w = gs["power_mean_w"]
            p.temp_max_c = gs["temp_max_c"]
        profiles.append(p)
    return profiles


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(profiles: list[JobProfile]) -> str:
    """Compact table output with recommendations."""
    lines: list[str] = [""]

    # GPU + SLURM combined table
    gpu_or_slurm = [p for p in profiles if p.gpu_samples > 0 or p.slurm.job_id]
    if gpu_or_slurm:
        hdr = f"{'JobID':<12} {'State':<10} {'Elapsed':>8} {'RSS':>6} {'ReqMem':>6} {'GPU%':>5} {'VRAM':>6} {'Power':>6} {'Model/Stage'}"
        lines += ["=== Job Summary ===", "", hdr, "-" * len(hdr)]
        for p in gpu_or_slurm:
            s = p.slurm
            label = f"{p.model}/{p.stage}" if p.model else ""
            if p.dataset:
                label = f"{p.dataset}/{label}"
            lines.append(
                f"{s.job_id:<12} {s.state:<10} {s.elapsed:>8} {s.max_rss_gib:>4.1f}G {s.req_mem_gib:>4.0f}G "
                f"{p.gpu_util_mean:>4.0f}% {p.vram_used_max_gib:>4.1f}G {p.power_mean_w:>4.0f}W  {label}"
            )
        lines.append("")

    # Recommendations
    rss_vals = [p.slurm.max_rss_gib for p in profiles if p.slurm.max_rss_gib > 0]
    gpu_utils = [p.gpu_util_mean for p in profiles if p.gpu_samples > 0]
    req_mems = [p.slurm.req_mem_gib for p in profiles if p.slurm.req_mem_gib > 0]

    if rss_vals or gpu_utils:
        lines.append("=== Recommendations ===")
        if rss_vals and req_mems:
            suggested = int(max(rss_vals) * 1.3) + 1
            current = int(max(req_mems))
            if suggested < current * 0.7:
                lines.append(f"  --mem={suggested}G  (currently {current}G, peak {max(rss_vals):.1f}G)")
        if gpu_utils and sum(gpu_utils) / len(gpu_utils) < 50:
            lines.append(f"  GPU util avg {sum(gpu_utils)/len(gpu_utils):.0f}% — data loading bottleneck")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeline analysis
# ---------------------------------------------------------------------------

def timeline_analysis(job_ids: list[str], log_root: Path, *, plot_path: str | None = None) -> None:
    """Phase-level GPU analysis with optional timeline plot."""
    sacct = get_sacct_stats(job_ids)

    for jid in job_ids:
        rows = _parse_gpu_timeline(log_root / jid / "gpu_stats.csv")
        meta = _extract_job_metadata(log_root / jid, jid)
        slurm = sacct.get(jid)
        label = f"{meta.get('model', '?')}/{meta.get('scale', '?')}/{meta.get('stage', '?')} on {meta.get('dataset', '?')}"
        rss = f"{slurm.max_rss_gib:.1f}G" if slurm else "?"

        print(f"=== Job {jid} — {label} (RSS={rss}) ===")
        if not rows:
            print("  No gpu_stats.csv\n")
            continue

        n = len(rows)
        gu = [r["gpu_util"] for r in rows]
        mu = [r["mem_used_mb"] for r in rows]
        pw = [r["power_w"] for r in rows]

        print(f"  Duration:  {n * 30 / 60:.0f} min ({n} samples)")
        print(f"  GPU util:  avg={sum(gu)/n:.0f}%  peak={max(gu)}%")
        print(f"  VRAM:      avg={sum(mu)/n:.0f}MB  peak={max(mu)}MB  of {rows[0]['mem_total_mb']}MB ({100*max(mu)//rows[0]['mem_total_mb']}%)")
        print(f"  Power:     avg={sum(pw)/n:.0f}W  peak={max(pw):.0f}W")

        p10 = max(1, n // 10)
        for name, subset in [("Startup", rows[:p10]), ("Training", rows[p10:n-p10]), ("Shutdown", rows[n-p10:])]:
            sg = [r["gpu_util"] for r in subset]
            sm = [r["mem_used_mb"] for r in subset]
            print(f"  {name}: gpu={sum(sg)/len(sg):.0f}%  vram={sum(sm)/len(sm):.0f}MB")

        idle = sum(1 for g in gu if g < 5)
        print(f"  Idle (<5%): {idle}/{n} ({100 * idle // n}%)\n")

    if plot_path:
        _plot_timelines(job_ids, log_root, plot_path)


def _plot_timelines(job_ids: list[str], log_root: Path, plot_path: str) -> None:
    """Multi-panel timeline PNG: GPU util + VRAM + power."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_gpu, ax_vram, ax_power) = plt.subplots(3, 1, figsize=(14, 8))

    for jid in job_ids:
        rows = _parse_gpu_timeline(log_root / jid / "gpu_stats.csv")
        if not rows:
            continue
        meta = _extract_job_metadata(log_root / jid, jid)
        label = f"{jid} {meta.get('model', '?')}/{meta.get('dataset', '?')}"
        mins = [i * 0.5 for i in range(len(rows))]
        ax_gpu.plot(mins, [r["gpu_util"] for r in rows], label=label, alpha=0.8)
        ax_vram.plot(mins, [r["mem_used_mb"] / 1024 for r in rows], label=label, alpha=0.8)
        ax_power.plot(mins, [r["power_w"] for r in rows], label=label, alpha=0.8)

    ax_gpu.set_ylabel("GPU Util %"); ax_gpu.set_ylim(0, 105)
    ax_gpu.axhline(80, color="gray", ls="--", alpha=0.5); ax_gpu.legend(fontsize=7)
    ax_gpu.set_title("GPU Timeline")
    ax_vram.set_ylabel("VRAM (GiB)")
    ax_vram.axhline(16, color="red", ls="--", alpha=0.5, label="V100 limit"); ax_vram.legend(fontsize=7)
    ax_power.set_ylabel("Power (W)"); ax_power.set_xlabel("Time (min)")
    plt.tight_layout(); plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile SLURM job resources")
    parser.add_argument("job_ids", nargs="*", help="SLURM job IDs")
    parser.add_argument("--since", help="All submitit jobs since (e.g. 2026-03-25)")
    parser.add_argument("--log-root", default="slurm_logs")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeline", action="store_true", help="Deep GPU phase analysis")
    parser.add_argument("--plot", help="Save timeline PNG (implies --timeline)")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    job_ids = list(args.job_ids)
    if args.since:
        job_ids.extend(discover_jobs_since(args.since))
    if not job_ids:
        parser.error("Provide job IDs or --since")

    # Deduplicate
    seen: set[str] = set()
    unique = [j for j in job_ids if j not in seen and not seen.add(j)]

    if args.timeline or args.plot:
        timeline_analysis(unique, log_root, plot_path=args.plot)
    elif args.json:
        profiles = profile_jobs(unique, log_root)
        print(json.dumps([asdict(p) for p in profiles], indent=2))
    else:
        profiles = profile_jobs(unique, log_root)
        print(format_table(profiles))


if __name__ == "__main__":
    main()
