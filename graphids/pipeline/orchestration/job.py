"""Job definition models for pipeline orchestration.

Pydantic v2 frozen models for describing SLURM resource requirements.
Used by Dagster orchestration.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel


class ResourceSpec(BaseModel, frozen=True):
    """Resource requirements for a pipeline job.

    Used by Dagster SLURM orchestration.
    SLURM-specific fields (partition, exclude_nodes) default to safe values
    so non-SLURM callers can ignore them.
    """

    nodes: int = 1
    gpus: int = 0
    cpus: int = 4
    memory_gb: int = 20
    walltime: timedelta = timedelta(hours=3)
    partition: str = "cpu"
    exclude_nodes: str = ""

    @property
    def mem_slurm(self) -> str:
        """Memory as SLURM string, e.g. '20G'."""
        return f"{self.memory_gb}G"

    @property
    def walltime_slurm(self) -> str:
        """Walltime as SLURM 'H:MM:SS' string."""
        total = int(self.walltime.total_seconds())
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h}:{m:02d}:{s:02d}"

    @classmethod
    def from_yaml(cls, data: dict) -> ResourceSpec:
        """Construct from resources.yaml entry.

        Accepts either ``memory_gb`` (int) or ``mem`` (str like '20G').
        Accepts either ``walltime`` (timedelta-compatible) or string 'H:MM:SS'.
        """
        # Memory: accept 'mem' string ('20G') or 'memory_gb' int
        mem_gb = data.get("memory_gb")
        if mem_gb is None and "mem" in data:
            mem_str = data["mem"]
            # Parse '32G' -> 32, '512M' -> 0 (round down)
            if mem_str.upper().endswith("G"):
                mem_gb = int(mem_str[:-1])
            elif mem_str.upper().endswith("M"):
                mem_gb = max(1, int(mem_str[:-1]) // 1024)
            else:
                mem_gb = int(mem_str)
        elif mem_gb is None:
            mem_gb = 20

        # Walltime: accept 'H:MM:SS' string
        wt = data.get("walltime", "3:00:00")
        if isinstance(wt, str):
            parts = wt.split(":")
            if len(parts) == 3:
                wt = timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
            elif len(parts) == 2:
                wt = timedelta(minutes=int(parts[0]), seconds=int(parts[1]))

        return cls(
            nodes=data.get("nodes", 1),
            gpus=data.get("gpus", 0),
            cpus=data.get("cpus", 4),
            memory_gb=mem_gb,
            walltime=wt,
            partition=data.get("partition", "cpu"),
            exclude_nodes=data.get("exclude_nodes", ""),
        )
