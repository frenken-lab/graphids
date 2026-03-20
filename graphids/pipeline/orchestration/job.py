"""Job definition models for pipeline orchestration.

Pydantic v2 frozen models for describing resource requirements.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel


class ResourceSpec(BaseModel, frozen=True):
    """Resource requirements for a pipeline job.

    Platform-agnostic resource description. Backend adapters
    (e.g. slurm.make_slurm_executor) map these to platform-specific params.
    """

    nodes: int = 1
    gpus: int = 0
    cpus: int = 4
    memory_gb: int = 20
    walltime: timedelta = timedelta(hours=3)
    partition: str = "cpu"
    exclude_nodes: str = ""

    @classmethod
    def from_yaml(cls, data: dict) -> ResourceSpec:
        """Construct from resources.yaml entry.

        Accepts either ``memory_gb`` (int) or ``mem`` (str like '20G').
        Accepts either ``walltime`` (timedelta-compatible) or string 'H:MM:SS'.
        """
        mem_gb = data.get("memory_gb")
        if mem_gb is None and "mem" in data:
            mem_str = data["mem"]
            if mem_str.upper().endswith("G"):
                mem_gb = int(mem_str[:-1])
            elif mem_str.upper().endswith("M"):
                mem_gb = max(1, int(mem_str[:-1]) // 1024)
            else:
                mem_gb = int(mem_str)
        elif mem_gb is None:
            mem_gb = 20

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
