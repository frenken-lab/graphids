from __future__ import annotations


def test_lake_root_falls_back_to_project_dotenv(monkeypatch, tmp_path):
    from graphids import paths

    monkeypatch.delenv("GRAPHIDS_LAKE_ROOT", raising=False)
    monkeypatch.delenv("GRAPHIDS_RUN_ROOT", raising=False)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    paths._dotenv_values.cache_clear()
    (tmp_path / ".env").write_text(
        """
export GRAPHIDS_LAKE_ROOT="/lake/root"
export GRAPHIDS_RUN_ROOT="/run/root"
""".lstrip()
    )

    assert paths.lake_root() == "/lake/root"
    assert paths.trial_dir().as_posix() == "/run/root"
