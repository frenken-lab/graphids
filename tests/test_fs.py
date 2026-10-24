from __future__ import annotations

import types

import pytest


def test_atomic_save_uses_unique_tempfile(monkeypatch, tmp_path):
    import sys

    from graphids._fs import atomic_save

    saved_paths: list[str] = []

    def fake_save(_obj, path):
        saved_paths.append(path)
        with open(path, "wb") as f:
            f.write(b"payload")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(save=fake_save))

    path = tmp_path / "payload.pt"
    atomic_save({"x": 1}, path)
    atomic_save({"x": 2}, path)

    assert path.read_bytes() == b"payload"
    assert path.with_suffix(".tmp").exists() is False
    assert len(saved_paths) == 2
    assert saved_paths[0] != saved_paths[1]
    assert all(".payload.pt." in saved for saved in saved_paths)


def test_atomic_json_failure_preserves_published_file_and_cleans_up(
    monkeypatch,
    tmp_path,
):
    """Failed journal publication must leave the last complete JSON readable."""
    import json
    from pathlib import Path

    from graphids._fs import atomic_write_json

    path = tmp_path / "journal" / "manifest.json"
    atomic_write_json(path, {"status": "created"})
    atomic_write_json(path, {"status": "finished", "name": "café"})
    assert json.loads(path.read_text()) == {"status": "finished", "name": "café"}
    published = path.read_bytes()

    with pytest.raises(TypeError):
        atomic_write_json(path, {"status": "running", "details": object()})
    assert path.read_bytes() == published
    assert list(path.parent.iterdir()) == [path]

    def cannot_replace(self, target):
        raise OSError("publication interrupted")

    monkeypatch.setattr(Path, "replace", cannot_replace)

    with pytest.raises(OSError, match="publication interrupted"):
        atomic_write_json(path, {"status": "running"})
    assert path.read_bytes() == published
    assert list(path.parent.iterdir()) == [path]
