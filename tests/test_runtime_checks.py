from __future__ import annotations


def test_module_version_reports_binary_import_errors(monkeypatch):
    from graphids import runtime_checks

    def fail_import(_name: str):
        raise OSError("bad extension")

    monkeypatch.setattr(runtime_checks, "import_module", fail_import)

    version, error = runtime_checks._module_version("torch_scatter")

    assert version is None
    assert error == "torch_scatter: OSError: bad extension"
