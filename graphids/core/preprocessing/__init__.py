"""Preprocessing module — unified API via PreprocessingPipeline.

Public API:
    from graphids.core.preprocessing import PreprocessingPipeline
    pipe = PreprocessingPipeline(cfg)
    train, val, num_ids = pipe.load_dataset()
    scenarios = pipe.load_test_scenarios()

Re-exports for convenience:
    get_batch_index, graph_attack_type     — graph utilities
    ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES   — CAN bus attack mappings
    TemporalGrouper, GraphSequence         — temporal grouping
    CollatedGraphDataset, EntityVocabulary  — data containers
    IRSchema                               — IR column schema
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._dataset import CollatedGraphDataset, GraphDataset
from ._engine import GraphEngine
from ._graph_utils import get_batch_index, graph_attack_type
from ._schema import EDGE_MANIFEST, NODE_MANIFEST, FeatureManifest, IRSchema
from ._temporal import GraphSequence, TemporalGrouper
from ._vocabulary import EntityVocabulary
from .adapters._can_bus import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES

if TYPE_CHECKING:
    from torch.utils.data import Subset

    from graphids.config import PipelineConfig

    from .adapters.base import DomainAdapter


__all__ = [
    # Primary API
    "PreprocessingPipeline",
    # Graph utilities (re-exported from _graph_utils)
    "get_batch_index",
    "graph_attack_type",
    # Attack type mappings (re-exported from adapters)
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
    # Temporal
    "TemporalGrouper",
    "GraphSequence",
    # Data containers
    "CollatedGraphDataset",
    "GraphDataset",
    "EntityVocabulary",
    "IRSchema",
    # Feature manifests
    "FeatureManifest",
    "NODE_MANIFEST",
    "EDGE_MANIFEST",
]

# Startup assertion: manifest must agree with pipeline.yaml constants
from graphids.config import EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT

assert NODE_MANIFEST.count == NODE_FEATURE_COUNT, (
    f"NODE_MANIFEST.count={NODE_MANIFEST.count} != NODE_FEATURE_COUNT={NODE_FEATURE_COUNT}"
)
assert EDGE_MANIFEST.count == EDGE_FEATURE_COUNT, (
    f"EDGE_MANIFEST.count={EDGE_MANIFEST.count} != EDGE_FEATURE_COUNT={EDGE_FEATURE_COUNT}"
)


class PreprocessingPipeline:
    """Unified API for preprocessing, caching, and dataset loading."""

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._prep = cfg.preprocessing
        self._adapter = self._resolve_adapter()

    def _resolve_adapter(self) -> DomainAdapter:
        """Resolve adapter from dataset config.

        Default: CANBusAdapter. Future: lookup by dataset_entry.adapter field.
        """
        from .adapters._can_bus import CANBusAdapter

        return CANBusAdapter(
            chunk_size=self._prep.chunk_size,
            include_attack_type=True,
        )

    def load_dataset(
        self,
        force_rebuild: bool = False,
    ) -> tuple[Subset, Subset, int]:
        """Load preprocessed dataset with caching. Returns (train, val, num_ids)."""
        from graphids.config import cache_dir, data_dir

        from ._cache import load_dataset

        return load_dataset(
            self._cfg.dataset,
            data_dir(self._cfg),
            cache_dir(self._cfg),
            force_rebuild_cache=force_rebuild,
            seed=self._cfg.seed,
            train_val_split=self._prep.train_val_split,
            adapter=self._adapter,
            window_size=self._prep.window_size,
            stride=self._prep.stride,
        )

    def load_test_scenarios(
        self,
        force_rebuild: bool = False,
    ) -> dict[str, CollatedGraphDataset]:
        """Load held-out test scenarios with per-scenario caching."""
        from graphids.config import cache_dir, data_dir

        from ._cache import load_test_scenarios

        return load_test_scenarios(
            self._cfg.dataset,
            data_dir(self._cfg),
            cache_dir(self._cfg),
            force_rebuild_cache=force_rebuild,
            adapter=self._adapter,
        )

    @staticmethod
    def get_batch_index(g, device):
        """Get batch index from graph, creating a single-graph default if absent."""
        return get_batch_index(g, device)

    @staticmethod
    def graph_attack_type(g, default=-1):
        """Get attack_type from a PyG graph, with backward-compat default."""
        return graph_attack_type(g, default)
