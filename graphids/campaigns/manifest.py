"""Campaign manifest schema + YAML loader.

One ``<name>.yaml`` = metadata + ``defaults`` + ``cells`` (sparse overrides).
Merging a cell onto defaults → ``PipelineConfig`` is the validation gate.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from graphids.config.constants import VALID_FUSION_METHODS, VALID_SCALES
from graphids.config.topology import TOPOLOGY

if TYPE_CHECKING:
    from graphids.orchestrate.config import PipelineConfig

# Literal duplicated from PipelineConfig (not imported) — any manifest-side
# import from ``graphids.orchestrate.*`` drags ``torch.nn`` via ``instantiate``.
_ConvType = Literal["gatv2", "gat", "gps"]
_LossFn = Literal["focal", "ce", "weighted_ce"]
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


def _in(valid, label):
    def _v(v):
        if v not in valid:
            raise ValueError(f"{label}={v!r} not in {sorted(valid)}")
        return v
    return _v


def _all_in(valid, label):
    def _v(v):
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
        return v
    return _v


class CampaignDefaults(BaseModel):
    """Optional ``PipelineConfig`` overrides shared by every cell.

    ``extra='forbid'`` catches YAML typos that ``PipelineConfig`` (extra=ignore)
    would silently swallow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str | None = None
    seed: int | None = None
    scale: Annotated[str, AfterValidator(_in(VALID_SCALES, "scale"))] | None = None
    lake_root: str | None = None
    fusion_method: (
        Annotated[str, AfterValidator(_in(VALID_FUSION_METHODS, "fusion_method"))] | None
    ) = None
    stages: Annotated[list[str], AfterValidator(_all_in(TOPOLOGY.stages, "stage"))] | None = None
    conv_type: _ConvType | None = None
    variational: bool | None = None
    loss_fn: _LossFn | None = None
    tla_overrides: dict[str, Any] | None = None
    max_retries: int | None = None

    def overrides(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Cell(CampaignDefaults):
    """One campaign cell — a sparse override identified by a stable id."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(..., pattern=_ID_PATTERN, min_length=1, max_length=64)

    def overrides(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"id"})


class Campaign(BaseModel):
    """A frozen-once plan of cells, loaded from ``<name>.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    name: str = Field(..., pattern=_ID_PATTERN, min_length=1, max_length=64)
    description: str = ""
    created: date
    frozen_at: date | None = None
    defaults: CampaignDefaults = Field(default_factory=CampaignDefaults)
    cells: tuple[Cell, ...] = ()

    @field_validator("cells")
    @classmethod
    def _unique_ids(cls, cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
        ids = [c.id for c in cells]
        dupes = {cid for cid in ids if ids.count(cid) > 1}
        if dupes:
            raise ValueError(f"duplicate cell ids: {sorted(dupes)}")
        return cells

    @property
    def is_frozen(self) -> bool:
        return self.frozen_at is not None

    def get_cell(self, cell_id: str) -> Cell:
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        raise KeyError(f"cell {cell_id!r} not in campaign {self.name!r}")

    def merged_config(self, cell_id: str) -> "PipelineConfig":
        return merged_pipeline_config(self.get_cell(cell_id), self.defaults)


def merged_pipeline_config(cell: Cell, defaults: CampaignDefaults) -> "PipelineConfig":
    """Merge defaults + cell into a validated ``PipelineConfig``.

    tla_overrides is dict-merged (cell wins on collision); all other
    fields are scalar-overwrite with cell winning.
    """
    # Deferred — orchestrate.config pulls torch via the package __init__.
    from graphids.orchestrate.config import PipelineConfig

    merged: dict[str, Any] = dict(defaults.overrides())
    cell_over = cell.overrides()
    d_tla = merged.pop("tla_overrides", None)
    c_tla = cell_over.pop("tla_overrides", None)
    if d_tla or c_tla:
        merged["tla_overrides"] = {**(d_tla or {}), **(c_tla or {})}
    merged.update(cell_over)
    return PipelineConfig(**merged)


def load_campaign(path: Path) -> Campaign:
    """Load + validate a campaign manifest from YAML."""
    import yaml

    return Campaign.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def cell_statuses(
    campaign: Campaign, *, manifest_path: Path, lake_root: Path
) -> dict[str, str]:
    """cell_id → {running, completed, failed}, reconstructed from traces.jsonl.

    Walks every traces.jsonl under lake_root, filters spans tagged with
    ``campaign.manifest == manifest_path`` (absolute), groups by cell_id,
    collects per-stage status. completed = all stages OK; failed = any
    ERROR; otherwise running. Cells with zero matching spans are absent
    (callers treat absent as pending).
    """
    manifest_str = str(Path(manifest_path).resolve())
    per_cell: dict[str, dict[str, str]] = {}
    for f in Path(lake_root).rglob("traces.jsonl"):
        try:
            text = f.read_text()
        except OSError:
            continue
        for raw in text.splitlines():
            if not raw.strip():
                continue
            try:
                span = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if span.get("name") != "training.fit":
                continue
            attrs = span.get("attributes") or {}
            if attrs.get("campaign.manifest") != manifest_str:
                continue
            cid, stage = attrs.get("campaign.cell_id"), attrs.get("ml.stage")
            if not cid or not stage:
                continue
            per_cell.setdefault(cid, {})[stage] = (
                (span.get("status") or {}).get("status_code") or ""
            )

    out: dict[str, str] = {}
    for cell in campaign.cells:
        seen = per_cell.get(cell.id)
        if not seen:
            continue
        stages = campaign.merged_config(cell.id).stages
        if any(seen.get(s) == "ERROR" for s in stages):
            out[cell.id] = "failed"
        elif all(seen.get(s) == "OK" for s in stages):
            out[cell.id] = "completed"
        else:
            out[cell.id] = "running"
    return out
