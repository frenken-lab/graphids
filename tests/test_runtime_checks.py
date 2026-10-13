import pytest


def test_backend_errors_identify_missing_extensions(monkeypatch):
    from graphids import runtime_checks

    def fail(_name):
        raise ImportError("missing extension")

    monkeypatch.setattr(runtime_checks, "import_module", fail)
    with pytest.raises(RuntimeError, match="pinned CUDA selective scan"):
        runtime_checks.assert_mamba_backend()
