"""Standardized Intermediate Representation (IR) schema for preprocessing.

All domain adapters produce DataFrames conforming to this schema before
handing off to GraphEngine. The IR decouples domain-specific parsing
(CAN bus hex, network flow CSVs, etc.) from domain-agnostic graph
construction.

Column layout (order matters for numpy conversion):
    entity_id      – dense integer ID for the node (post-vocabulary encoding)
    feature_0..N   – continuous features (e.g. normalized payload bytes)
    source_id      – dense integer ID of the edge source node
    target_id      – dense integer ID of the edge target node
    label          – binary label (0=normal, 1=attack)
    attack_type    – integer-coded attack type (0=normal, 1+=attack subtypes)
                     Optional: only present when include_attack_type=True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Column name constants used throughout the preprocessing pipeline
COL_ENTITY_ID = "entity_id"
COL_SOURCE_ID = "source_id"
COL_TARGET_ID = "target_id"
COL_LABEL = "label"
COL_ATTACK_TYPE = "attack_type"


def feature_columns(n: int) -> list[str]:
    """Return feature column names for *n* features."""
    return [f"feature_{i}" for i in range(n)]


@dataclass(frozen=True)
class IRSchema:
    """Describes and validates the Intermediate Representation DataFrame.

    Parameters
    ----------
    num_features : int
        Number of continuous feature columns (``feature_0`` … ``feature_{n-1}``).
    include_attack_type : bool
        If True, include an ``attack_type`` column after ``label``.
    """

    num_features: int
    include_attack_type: bool = False
    _required_prefix: list[str] = field(init=False, repr=False)
    _required_suffix: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        prefix = [COL_ENTITY_ID] + feature_columns(self.num_features)
        suffix = [COL_SOURCE_ID, COL_TARGET_ID, COL_LABEL]
        if self.include_attack_type:
            suffix.append(COL_ATTACK_TYPE)
        object.__setattr__(self, "_required_prefix", prefix)
        object.__setattr__(self, "_required_suffix", suffix)

    @property
    def columns(self) -> list[str]:
        """Full ordered column list."""
        return self._required_prefix + self._required_suffix

    @property
    def col_source(self) -> int:
        """Numpy column index for source_id."""
        return len(self._required_prefix)

    @property
    def col_target(self) -> int:
        """Numpy column index for target_id."""
        return len(self._required_prefix) + 1

    @property
    def col_label(self) -> int:
        """Numpy column index for label."""
        return len(self._required_prefix) + 2

    @property
    def col_attack_type(self) -> int | None:
        """Numpy column index for attack_type, or None if not included."""
        if not self.include_attack_type:
            return None
        return len(self._required_prefix) + 3

    def validate(self, df, *, strict: bool = False) -> None:
        """Validate that *df* conforms to this schema.

        Always checks: column names match, DataFrame not empty.
        Strict mode adds: no NaN in required columns, numeric dtypes for features.

        Raises ``ValueError`` on any violation.
        """
        import numpy as np

        expected = self.columns
        actual = list(df.columns)
        if actual != expected:
            missing = set(expected) - set(actual)
            extra = set(actual) - set(expected)
            parts = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra:
                parts.append(f"extra={sorted(extra)}")
            if not parts:
                parts.append("wrong column order")
            raise ValueError(f"IR schema violation: {', '.join(parts)}")

        if len(df) == 0:
            raise ValueError("IR schema violation: DataFrame is empty")

        if strict:
            feat_cols = feature_columns(self.num_features)
            for col in feat_cols:
                if not np.issubdtype(df[col].dtype, np.number):
                    raise ValueError(f"IR schema violation: column '{col}' is not numeric")
            required = [COL_ENTITY_ID, COL_SOURCE_ID, COL_TARGET_ID, COL_LABEL] + feat_cols
            for col in required:
                if df[col].isna().any():
                    raise ValueError(f"IR schema violation: NaN in required column '{col}'")


# ============================================================================
# Feature Manifest — single source of truth for feature dimensions
# ============================================================================


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one feature in a node or edge feature vector."""

    name: str
    index: int
    description: str
    value_range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class FeatureManifest:
    """Ordered collection of feature specifications."""

    features: tuple[FeatureSpec, ...]

    @property
    def count(self) -> int:
        return len(self.features)

    def to_json(self) -> list[dict]:
        return [
            {
                "name": f.name,
                "index": f.index,
                "description": f.description,
                "value_range": list(f.value_range),
            }
            for f in self.features
        ]


def build_node_manifest(num_features: int = 8) -> FeatureManifest:
    """Build the node feature manifest for a given number of payload bytes.

    Layout (26-D for CAN bus with 8 payload bytes):
        [0]                  entity_id mean
        [1:1+N]              per-byte payload means
        [1+N:1+2N]           per-byte payload stds
        [1+2N]               payload entropy (Shannon)
        [1+2N+1]             payload change rate — mean
        [1+2N+2]             payload change rate — max
        [1+2N+3]             skewness
        [1+2N+4]             kurtosis
        [1+2N+5]             clustering coefficient
        [1+2N+6]             split-half ratio
        [1+2N+7]             normalized occurrence count
        [1+2N+8]             last temporal position
    """
    specs: list[FeatureSpec] = []
    idx = 0

    specs.append(FeatureSpec("entity_id_mean", idx, "Mean entity ID within window"))
    idx += 1

    for i in range(num_features):
        specs.append(FeatureSpec(f"byte_{i}_mean", idx, f"Mean of payload byte {i}"))
        idx += 1

    for i in range(num_features):
        specs.append(FeatureSpec(f"byte_{i}_std", idx, f"Std of payload byte {i}"))
        idx += 1

    extended = [
        ("payload_entropy", "Shannon entropy over byte histogram (base 2, /8)"),
        ("change_rate_mean", "Mean abs payload change between consecutive messages"),
        ("change_rate_max", "Max abs payload change between consecutive messages"),
        ("skewness", "Average skewness across payload bytes (clamped ±10, rescaled)"),
        ("kurtosis", "Average excess kurtosis across payload bytes (clamped ±10, rescaled)"),
        ("clustering_coeff", "Local clustering coefficient (networkx)"),
        ("split_half_ratio", "First-half mean / second-half mean of payload (/10)"),
        ("occurrence_count", "Min-max normalized occurrence count within window"),
        ("last_position", "Last temporal position within window (normalized)"),
    ]
    for name, desc in extended:
        specs.append(FeatureSpec(name, idx, desc))
        idx += 1

    return FeatureManifest(features=tuple(specs))


def build_edge_manifest() -> FeatureManifest:
    """Build the edge feature manifest (11-D).

    Layout:
        [0]  raw count
        [1]  frequency (count / window_length)
        [2]  mean interval between occurrences
        [3]  std interval
        [4]  regularity 1/(1+std)
        [5]  first occurrence position (normalized)
        [6]  last occurrence position (normalized)
        [7]  temporal span (last - first)
        [8]  bidirectionality flag
        [9]  degree product (src_deg * tgt_deg)
        [10] degree ratio (src_deg / tgt_deg)
    """
    specs = [
        FeatureSpec("raw_count", 0, "Number of edge occurrences in window"),
        FeatureSpec("frequency", 1, "Edge count / window length"),
        FeatureSpec("mean_interval", 2, "Mean interval between consecutive occurrences"),
        FeatureSpec("std_interval", 3, "Std of intervals between occurrences"),
        FeatureSpec("regularity", 4, "1 / (1 + std_interval)"),
        FeatureSpec("first_position", 5, "First occurrence position (normalized)"),
        FeatureSpec("last_position", 6, "Last occurrence position (normalized)"),
        FeatureSpec("temporal_span", 7, "Last - first position (normalized)"),
        FeatureSpec("bidirectionality", 8, "1 if reverse edge exists, else 0"),
        FeatureSpec("degree_product", 9, "src_degree * tgt_degree"),
        FeatureSpec("degree_ratio", 10, "src_degree / tgt_degree"),
    ]
    return FeatureManifest(features=tuple(specs))


# Module-level singletons
NODE_MANIFEST = build_node_manifest()
EDGE_MANIFEST = build_edge_manifest()
