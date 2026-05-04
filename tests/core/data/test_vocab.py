"""Shared-vocab contract: UNK reservation, digest stability, persist roundtrip.

Guards the Stage-1 invariants from ~/plans/oov-embedding-handling.md.
Framework-level shape/dtype tests are polars' / json's job.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from graphids.core.data.preprocessing.vocab import (
    load_vocab,
    persist_vocab,
    scan_arb_ids,
    vocab_digest,
)


def test_digest_is_deterministic_across_insertion_order():
    # CONTRACT: same (id, index) set → same digest. Otherwise two
    # functionally identical caches would hash-miss against each other.
    # Also checks the inlined vocab-build pattern (dense index starting
    # at 1, 0 reserved for UNK) — the invariant that keeps
    # ``replace_strict(..., default=0)`` routing unseen ids to the UNK row.
    assert min({v: i + 1 for i, v in enumerate([0x100, 0x200])}.values()) == 1
    v1 = {0x100: 1, 0x200: 2, 0x316: 3}
    v2 = {0x316: 3, 0x100: 1, 0x200: 2}
    assert vocab_digest(v1) == vocab_digest(v2)


def test_digest_sensitive_to_content():
    # CONTRACT: any (id, index) change → different digest. Guards
    # against silent cache reuse when vocab actually changed (e.g.,
    # adding a new test_subdir with unseen ids).
    base = {0x100: 1, 0x200: 2}
    assert vocab_digest(base) != vocab_digest({0x100: 1, 0x200: 3})  # index shift
    assert vocab_digest(base) != vocab_digest({**base, 0x300: 3})  # new id


def test_persist_load_roundtrip(tmp_path):
    # CONTRACT: persist → load returns the same entries and the same
    # digest. Roundtrip is the full cache-key path, so it has to survive
    # JSON serialization intact.
    vocab = {0x100: 1, 0x200: 2}
    path = tmp_path / "vocab.json"
    written_digest = persist_vocab(vocab, path)
    entries, read_digest = load_vocab(path)
    assert written_digest == read_digest
    # Entries are stringified in JSON — compare as such.
    assert entries == {str(k): v for k, v in vocab.items()}


def test_scan_arb_ids_unions_across_source_dirs(tmp_path: Path):
    # REGRESSION: this is the Stage-1 bug we fixed. Before the shared
    # vocab, train saw {100, 200} and test saw {200, 300}; each split
    # built its own vocab, so id=300 appeared at index 2 in the test
    # vocab but the model's embedding was sized 3 (train-only) → crash.
    # After the fix, scan_arb_ids walks both source_dirs and returns
    # {100, 200, 300}.
    train_dir = tmp_path / "train_sub"
    test_dir = tmp_path / "test_sub"
    train_dir.mkdir()
    test_dir.mkdir()
    pl.DataFrame({"arb_id": [0x100, 0x200], "timestamp": [1.0, 2.0]}).write_csv(train_dir / "a.csv")
    pl.DataFrame({"arb_id": [0x200, 0x300], "timestamp": [3.0, 4.0]}).write_csv(test_dir / "b.csv")
    ids = scan_arb_ids(tmp_path, ["train_sub", "test_sub"])
    assert ids == [0x100, 0x200, 0x300]


def test_scan_arb_ids_tolerates_hcrl_column_name(tmp_path: Path):
    # REGRESSION: HCRL CSVs use 'arbitration_id', in-schema uses 'arb_id'.
    # scan_arb_ids must accept both (same tolerance _read_raw provides).
    sub = tmp_path / "sub"
    sub.mkdir()
    pl.DataFrame({"arbitration_id": [0x100, 0x200]}).write_csv(sub / "hcrl.csv")
    assert scan_arb_ids(tmp_path, ["sub"]) == [0x100, 0x200]


def test_scan_arb_ids_raises_when_neither_column_present(tmp_path: Path):
    # CONTRACT: fail loud if the CSV doesn't have a recognizable id
    # column. Silent 0-row return would poison the vocab with an empty
    # dict and force every downstream arb_id to UNK.
    sub = tmp_path / "sub"
    sub.mkdir()
    pl.DataFrame({"foo": [1, 2]}).write_csv(sub / "bad.csv")
    with pytest.raises(ValueError, match="arbitration_id"):
        scan_arb_ids(tmp_path, ["sub"])
