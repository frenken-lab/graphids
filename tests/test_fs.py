from __future__ import annotations

import types


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
