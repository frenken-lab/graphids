"""Per-datamodule Pydantic schemas — auto-generated from each ``__init__``.

Same pattern as ``graphids.core.models.schemas``. See that file's
docstring for rationale.
"""

from __future__ import annotations

from graphids.core._schema_gen import schema_for
from graphids.core.data.datamodule.fusion import FusionDataModule
from graphids.core.data.datamodule.graph import GraphDataModule

GraphDataConfig = schema_for(GraphDataModule)
FusionDataConfig = schema_for(FusionDataModule)

__all__ = [
    "GraphDataConfig",
    "FusionDataConfig",
]
