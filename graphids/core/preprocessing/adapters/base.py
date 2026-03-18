"""Abstract base class for domain adapters.

Each adapter knows how to:
1. Discover raw data files for a given split
2. Read a single raw file into a DataFrame
3. Build an EntityVocabulary from the raw data
4. Convert raw data into the Intermediate Representation (IR) DataFrame

The adapter encapsulates all domain-specific knowledge (CAN bus hex parsing,
network flow IP extraction, etc.) so that the GraphEngine can remain
domain-agnostic.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .._vocabulary import EntityVocabulary


class DomainAdapter(abc.ABC):
    """Abstract base class for domain-specific data adapters."""

    @abc.abstractmethod
    def discover_files(
        self,
        root: str | Path,
        split: str = "train_",
    ) -> list[Path]:
        """Find raw data files for the given split.

        Parameters
        ----------
        root : path
            Root directory containing the dataset.
        split : str
            Split identifier (e.g. ``"train_"``, ``"test_01_DoS"``).

        Returns
        -------
        list[Path]
            Sorted list of data file paths.
        """

    @abc.abstractmethod
    def build_vocabulary(
        self,
        files: Sequence[str | Path],
    ) -> EntityVocabulary:
        """Scan files and build an entity vocabulary.

        Parameters
        ----------
        files : sequence of paths
            Files to scan for entity IDs.

        Returns
        -------
        EntityVocabulary
            Vocabulary mapping raw IDs to dense indices.
        """

    @abc.abstractmethod
    def read_and_convert(
        self,
        file_path: str | Path,
        vocab: EntityVocabulary,
    ) -> pd.DataFrame:
        """Read a raw file and convert it to the IR DataFrame format.

        The returned DataFrame must conform to the adapter's ``IRSchema``.

        Parameters
        ----------
        file_path : path
            Path to a single raw data file.
        vocab : EntityVocabulary
            Vocabulary for encoding entity IDs.

        Returns
        -------
        pd.DataFrame
            IR-conformant DataFrame (may be empty if file is invalid).
        """

    @abc.abstractmethod
    def to_init_kwargs(self) -> dict:
        """Return kwargs that can reconstruct this adapter via ``cls(**kwargs)``.

        Used by the parallel driver to serialize the adapter for Ray remote
        tasks. Each concrete adapter must return its own init parameters.
        """

    @property
    @abc.abstractmethod
    def schema(self):
        """Return the ``IRSchema`` for this adapter's output."""
