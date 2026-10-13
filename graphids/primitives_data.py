"""Snapshot stream batching and pooled CAN data configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.core.data.snapshots.preprocessing import CANSnapshotRepresentationCfg


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CANSnapshotDMCfg(_Cfg):
    type: Literal["can_snapshot_dm"] = "can_snapshot_dm"
    manifest_path: str
    datasets: list[str]
    representation_cfg: CANSnapshotRepresentationCfg = Field(
        default_factory=CANSnapshotRepresentationCfg
    )
    sequence_bins: int = Field(default=128, ge=1)
    warmup_bins: int = Field(default=128, ge=0)
    stream_batch_size: int = Field(default=1, ge=1)
    eval_chunk_bins: int = Field(default=256, ge=1)
    num_workers: int = Field(default=0, ge=0)
    seed: int = 42
    context_withholding: float = Field(default=0.25, ge=0, le=1)
    neighbor_cap: int = Field(default=8, ge=1)
    retention_seconds: float = Field(default=1.0, gt=0)
    anchor_capacity: int | None = Field(default=None, ge=1)
    transient_capacity: int | None = Field(default=None, ge=1)
    max_node_bin_tokens: int = Field(default=131072, ge=1)
    max_graph_messages: int = Field(default=1048576, ge=1)
    max_snapshot_bytes: int = Field(default=4 * 1024**3, ge=1)
    cache_root: str | None = None

    def build(self):
        from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule

        values = self.model_dump(exclude={"type", "representation_cfg"})
        return CANSnapshotDataModule(
            **values, representation_cfg=self.representation_cfg
        )


def can_snapshot_dm(**kwargs) -> CANSnapshotDMCfg:
    return CANSnapshotDMCfg(**kwargs)


DataCfg = Annotated[CANSnapshotDMCfg, Field(discriminator="type")]
