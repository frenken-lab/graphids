"""Runtime environment checks for compute jobs."""

from __future__ import annotations

from importlib import import_module


def _cuda_suffix(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    parts = cuda_version.split(".")
    if len(parts) < 2:
        return None
    return f"cu{parts[0]}{parts[1]}"


def _module_version(name: str) -> tuple[str | None, str | None]:
    try:
        mod = import_module(name)
    except ModuleNotFoundError:
        return None, None
    except Exception as exc:  # noqa: BLE001 - binary extension imports fail in several ways
        return None, f"{name}: {type(exc).__name__}: {exc}"
    return str(getattr(mod, "__version__", "")), None


def assert_pyg_cuda_extensions_match() -> None:
    """Fail fast when PyG native wheels do not match PyTorch's CUDA build."""

    import torch

    expected = _cuda_suffix(torch.version.cuda)
    if expected is None:
        return

    mismatches: list[str] = []
    missing: list[str] = []
    import_errors: list[str] = []
    for module_name in ("torch_scatter", "torch_sparse", "torch_cluster"):
        version, import_error = _module_version(module_name)
        if import_error is not None:
            import_errors.append(import_error)
            continue
        if version is None:
            missing.append(module_name)
            continue
        if expected not in version:
            mismatches.append(f"{module_name}=={version}")

    pyg_lib_version, pyg_lib_import_error = _module_version("pyg_lib")
    if pyg_lib_import_error is not None:
        import_errors.append(pyg_lib_import_error)
    elif pyg_lib_version is None:
        missing.append("pyg_lib")
    elif expected not in pyg_lib_version:
        mismatches.append(f"pyg_lib=={pyg_lib_version}")

    if mismatches or missing or import_errors:
        details = []
        if import_errors:
            details.append("import errors: " + ", ".join(import_errors))
        if mismatches:
            details.append("mismatched: " + ", ".join(mismatches))
        if missing:
            details.append("missing: " + ", ".join(missing))
        raise RuntimeError(
            "PyG CUDA extension preflight failed: "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"expected extension suffix {expected}; "
            + "; ".join(details)
        )
