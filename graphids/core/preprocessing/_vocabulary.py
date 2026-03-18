"""Entity vocabulary for mapping raw identifiers to dense integer indices.

Replaces the old ``build_lightweight_id_mapping`` / ``build_id_mapping_from_normal``
functions with a single class that handles building, encoding, persistence, and OOV.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Sentinel key for out-of-vocabulary entities
OOV_KEY = "OOV"


class EntityVocabulary:
    """Bidirectional mapping between raw entity IDs and dense integer indices.

    The vocabulary is built from observed IDs, sorted for reproducibility,
    with an OOV sentinel appended at the end.

    Parameters
    ----------
    id_to_index : dict
        Mapping from raw ID (int or str) to dense index. Must contain the
        ``OOV`` sentinel key.
    """

    def __init__(self, id_to_index: dict):
        if OOV_KEY not in id_to_index:
            raise ValueError(f"Vocabulary must contain '{OOV_KEY}' sentinel")
        self._id_to_index = dict(id_to_index)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_ids(cls, ids: Iterable[int]) -> EntityVocabulary:
        """Build vocabulary from an iterable of raw integer IDs."""
        unique_ids = sorted(set(ids))
        mapping = {raw_id: idx for idx, raw_id in enumerate(unique_ids)}
        mapping[OOV_KEY] = len(mapping)
        return cls(mapping)

    @classmethod
    def build_from_files(
        cls,
        csv_files: Sequence[str | Path],
        id_column: int = 1,
        hex_convert: bool = True,
        converter: Callable | None = None,
    ) -> EntityVocabulary:
        """Build vocabulary by scanning the ID column of CSV files.

        Only reads the specified column to minimize memory usage.
        Equivalent to the old ``build_lightweight_id_mapping``.

        Parameters
        ----------
        csv_files : sequence of paths
            CSV files to scan.
        id_column : int
            0-based column index containing entity IDs.
        hex_convert : bool
            If True, treat column values as hex strings and convert to int.
            Ignored when *converter* is provided.
        converter : callable, optional
            Custom value → int|None converter. When provided, *hex_convert*
            is ignored and this callable is used for every raw value.
        """
        if converter is None and hex_convert:
            from graphids.core.preprocessing.adapters._can_bus import _safe_hex_to_int

            converter = _safe_hex_to_int

        unique_ids: set[int] = set()

        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:
                log.info("Scanning file %d/%d for entity IDs...", i + 1, len(csv_files))
            try:
                df_col = pd.read_csv(csv_file, usecols=[id_column])
                raw_values = df_col.iloc[:, 0].dropna().unique()

                if converter is not None:
                    for val in raw_values:
                        converted = converter(val)
                        if converted is not None:
                            unique_ids.add(converted)
                else:
                    unique_ids.update(int(v) for v in raw_values)

                del df_col
            except Exception as e:
                log.warning("Could not scan %s: %s", csv_file, e)

        log.info("Built vocabulary with %d entities (+ OOV)", len(unique_ids))
        return cls.build_from_ids(unique_ids)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    @property
    def oov_index(self) -> int:
        """Dense index of the OOV sentinel."""
        return self._id_to_index[OOV_KEY]

    def encode(self, raw_id) -> int:
        """Map a single raw ID to its dense index (OOV if unseen)."""
        return self._id_to_index.get(raw_id, self.oov_index)

    def encode_series(self, series: pd.Series) -> pd.Series:
        """Vectorized encoding of a pandas Series."""
        return series.map(self._id_to_index).fillna(self.oov_index).astype(int)

    def __len__(self) -> int:
        """Number of entries including OOV."""
        return len(self._id_to_index)

    def __contains__(self, raw_id) -> bool:
        return raw_id in self._id_to_index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._id_to_index, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> EntityVocabulary:
        """Load vocabulary from a pickle file."""
        with open(path, "rb") as f:
            mapping = pickle.load(f)
        return cls(mapping)

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the raw id_to_index dict (for legacy compatibility)."""
        return dict(self._id_to_index)

    @classmethod
    def from_legacy_mapping(cls, id_mapping: dict) -> EntityVocabulary:
        """Wrap a legacy id_mapping dict into an EntityVocabulary."""
        if OOV_KEY not in id_mapping:
            id_mapping = dict(id_mapping)
            id_mapping[OOV_KEY] = len(id_mapping)
        return cls(id_mapping)

    def __repr__(self) -> str:
        return f"EntityVocabulary(size={len(self)}, oov_index={self.oov_index})"
