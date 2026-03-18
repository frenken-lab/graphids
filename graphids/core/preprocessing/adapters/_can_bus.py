"""CAN bus domain adapter.

Encapsulates all CAN-bus-specific knowledge:
- 4-column CSV format: [Timestamp, arbitration_id, data_field, attack]
- Hex-encoded CAN IDs and payload bytes
- Temporal adjacency: edge topology via ``CAN_ID.shift(-1)``
- 8-byte payload extraction from hex data_field
- Attack type exclusion (suppress, masquerade)

The ``shift(-1)`` temporal adjacency is the most critical implicit
assumption: message N connects to message N+1. This is CAN-specific
and does NOT belong in the GraphEngine.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from graphids.config import EXCLUDED_ATTACK_TYPES, MAX_DATA_BYTES

from .._schema import (
    COL_ATTACK_TYPE,
    IRSchema,
    feature_columns,
)
from .._vocabulary import EntityVocabulary
from .base import DomainAdapter

log = logging.getLogger(__name__)

# CAN bus IR schemas (8 data bytes → 8 features)
CAN_BUS_SCHEMA = IRSchema(num_features=8)
CAN_BUS_SCHEMA_WITH_ATTACK_TYPE = IRSchema(num_features=8, include_attack_type=True)

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _safe_hex_to_int(value) -> int | None:
    """Safely convert hex string or numeric value to integer."""
    if pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if all(c in _HEX_CHARS for c in value):
                return int(value, 16)
            return int(value)
        return int(value)
    except (ValueError, TypeError):
        return None


# Attack type string → integer code mapping
ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0,
    "dos": 1,
    "fuzzing": 2,
    "gear_spoofing": 3,
    "rpm_spoofing": 4,
    "suppress": 5,
    "masquerade": 6,
    "mixed": 7,
    "unknown": 8,
}
ATTACK_TYPE_NAMES: dict[int, str] = {v: k for k, v in ATTACK_TYPE_CODES.items()}


class CANBusAdapter(DomainAdapter):
    """Adapter for CAN bus CSV datasets.

    Parameters
    ----------
    chunk_size : int
        Number of CSV rows to process per chunk (for memory efficiency).
    excluded_attacks : sequence of str
        Attack type substrings to exclude from file discovery.
    include_attack_type : bool
        If True, include an ``attack_type`` column in the IR output.
        The attack type is inferred from the parent directory name.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        excluded_attacks: Sequence[str] = EXCLUDED_ATTACK_TYPES,
        include_attack_type: bool = False,
    ):
        from graphids.config.schema import PreprocessingConfig

        if chunk_size is None:
            chunk_size = PreprocessingConfig().chunk_size
        self._chunk_size = chunk_size
        self._excluded_attacks = list(excluded_attacks)
        self._include_attack_type = include_attack_type

    def to_init_kwargs(self) -> dict:
        return {
            "chunk_size": self._chunk_size,
            "excluded_attacks": self._excluded_attacks,
            "include_attack_type": self._include_attack_type,
        }

    @property
    def schema(self) -> IRSchema:
        if self._include_attack_type:
            return CAN_BUS_SCHEMA_WITH_ATTACK_TYPE
        return CAN_BUS_SCHEMA

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def discover_files(
        self,
        root: str | Path,
        split: str = "train_",
    ) -> list[Path]:
        """Find CAN bus CSV files for a given split.

        Walks the directory tree, matching folders whose basename
        contains *split* (case-insensitive). Excludes files matching
        any substring in ``excluded_attacks``.
        """
        csv_files: list[Path] = []
        root_str = str(root)

        for dirpath, _dirnames, filenames in os.walk(root_str):
            dir_basename = os.path.basename(dirpath).lower()
            if split.lower().rstrip("_") not in dir_basename:
                continue
            for fname in filenames:
                if not fname.endswith(".csv"):
                    continue
                if any(at in fname.lower() for at in self._excluded_attacks):
                    continue
                csv_files.append(Path(dirpath) / fname)

        return sorted(csv_files)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocabulary(
        self,
        files: Sequence[str | Path],
    ) -> EntityVocabulary:
        """Build vocabulary by scanning the arbitration_id column."""
        return EntityVocabulary.build_from_files(
            [str(f) for f in files],
            id_column=1,
            hex_convert=True,
        )

    # ------------------------------------------------------------------
    # Raw → IR conversion
    # ------------------------------------------------------------------

    @staticmethod
    def infer_attack_type(file_path: str | Path) -> str:
        """Infer attack type from directory name.

        Maps known subdirectory patterns to attack type names:
        - train_01_attack_free / train_ → "normal"
        - test_01_DoS → "dos"
        - test_02_fuzzing → "fuzzing"
        - test_03_gear_spoofing → "gear_spoofing"
        - test_04_rpm_spoofing → "rpm_spoofing"
        - test_05_suppress → "suppress"
        - test_06_masquerade → "masquerade"
        - test_*_known_vehicle_known_attack → "mixed"
        - train_02_with_attacks → "mixed"

        Falls back to "unknown" if no pattern matches.
        """
        dirname = Path(file_path).parent.name.lower()
        if "attack_free" in dirname:
            return "normal"
        if "dos" in dirname:
            return "dos"
        if "fuzzing" in dirname or "fuzzy" in dirname:
            return "fuzzing"
        if "gear_spoofing" in dirname or "gear" in dirname:
            return "gear_spoofing"
        if "rpm_spoofing" in dirname or "rpm" in dirname:
            return "rpm_spoofing"
        if "suppress" in dirname:
            return "suppress"
        if "masquerade" in dirname:
            return "masquerade"
        if "with_attacks" in dirname:
            return "mixed"
        if "known" in dirname or "unknown" in dirname:
            return "mixed"
        if "train" in dirname:
            return "normal"
        return "unknown"

    def read_and_convert(
        self,
        file_path: str | Path,
        vocab: EntityVocabulary,
    ) -> pd.DataFrame:
        """Read a CAN bus CSV and convert to IR format.

        Processing steps:
        1. Read CSV in chunks (streaming)
        2. Parse hex data_field into byte columns
        3. Build temporal adjacency (shift(-1))
        4. Convert hex to int, apply vocabulary encoding
        5. Normalize payload bytes to [0, 1]
        6. Rename to IR column layout
        7. (Optional) Add attack_type column from directory name
        """
        chunks = self._read_chunks(str(file_path))
        if not chunks:
            return pd.DataFrame(columns=self.schema.columns)

        attack_type_code = None
        if self._include_attack_type:
            attack_type_name = self.infer_attack_type(file_path)
            attack_type_code = ATTACK_TYPE_CODES.get(attack_type_name, 0)

        ir_chunks = []
        for chunk in chunks:
            ir = self._chunk_to_ir(chunk, vocab)
            if not ir.empty:
                if self._include_attack_type and attack_type_code is not None:
                    ir[COL_ATTACK_TYPE] = attack_type_code
                ir_chunks.append(ir)

        if not ir_chunks:
            return pd.DataFrame(columns=self.schema.columns)

        result = pd.concat(ir_chunks, ignore_index=True)
        result = result[self.schema.columns]
        self.schema.validate(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_chunks(self, csv_path: str) -> list[pd.DataFrame]:
        """Read a CSV file in chunks. Returns list of raw chunks."""
        chunks = []
        try:
            reader = pd.read_csv(
                csv_path,
                chunksize=self._chunk_size,
                engine="python",
                dtype=str,
            )
            for chunk in reader:
                chunk.columns = ["Timestamp", "arbitration_id", "data_field", "attack"]
                chunk.rename(columns={"arbitration_id": "CAN ID"}, inplace=True)
                chunks.append(chunk)
        except Exception as e:
            log.debug("Chunked reading failed for %s: %s. Trying full read.", csv_path, e)
            try:
                full = pd.read_csv(csv_path, engine="python", dtype=str)
                full.columns = ["Timestamp", "arbitration_id", "data_field", "attack"]
                full.rename(columns={"arbitration_id": "CAN ID"}, inplace=True)
                chunks.append(full)
            except Exception as e2:
                log.warning("Failed to read %s: %s", csv_path, e2)
        return chunks

    def _chunk_to_ir(self, chunk: pd.DataFrame, vocab: EntityVocabulary) -> pd.DataFrame:
        """Convert a raw CAN bus chunk to IR format.

        Returns the base IR columns (without attack_type). The attack_type
        column is added by ``read_and_convert`` after all chunks are processed.
        """
        # Parse hex data_field into byte columns
        chunk["data_field"] = chunk["data_field"].astype(str).fillna("").str.strip()
        chunk["DLC"] = chunk["data_field"].str.len() // 2

        data_field = chunk["data_field"].values
        byte_cols = []
        for i in range(MAX_DATA_BYTES):
            start = i * 2
            end = start + 2
            col_name = f"Data{i + 1}"
            chunk[col_name] = [s[start:end] if len(s) >= end else "00" for s in data_field]
            byte_cols.append(col_name)

        # Pad short payloads
        mask = chunk["DLC"] < MAX_DATA_BYTES
        for i in range(MAX_DATA_BYTES):
            chunk.loc[mask & (chunk["DLC"] <= i), byte_cols[i]] = "00"
        chunk.fillna("00", inplace=True)

        # --- CRITICAL: Temporal adjacency (CAN-bus specific) ---
        # Message N's target is message N+1's CAN ID
        chunk["Source"] = chunk["CAN ID"]
        chunk["Target"] = chunk["CAN ID"].shift(-1)

        # Convert hex to int
        hex_columns = ["CAN ID", "Source", "Target"] + byte_cols
        for col in hex_columns:
            chunk[col] = chunk[col].apply(_safe_hex_to_int)

        # Apply vocabulary encoding to ID columns
        oov = vocab.oov_index
        for col in ["CAN ID", "Source", "Target"]:
            chunk[col] = chunk[col].map(vocab.to_dict()).fillna(oov).astype(int)

        # Drop last row (no valid target after shift)
        chunk = chunk.iloc[:-1].copy()
        chunk["label"] = chunk["attack"].astype(int)

        # Normalize payload bytes to [0, 1]
        for col in byte_cols:
            chunk[col] = chunk[col] / 255.0

        # Rename to IR column layout
        feat_names = feature_columns(MAX_DATA_BYTES)
        rename_map = {byte_cols[i]: feat_names[i] for i in range(MAX_DATA_BYTES)}
        rename_map["CAN ID"] = "entity_id"
        rename_map["Source"] = "source_id"
        rename_map["Target"] = "target_id"
        chunk.rename(columns=rename_map, inplace=True)

        # Use base schema columns (without attack_type) — attack_type is
        # appended by read_and_convert() after chunk processing.
        return chunk[CAN_BUS_SCHEMA.columns]
