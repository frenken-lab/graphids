"""Orchestration: manifest-driven SLURM DAG submission."""

# Lazy imports to avoid RuntimeWarning when running `python -m ...manifest`
def __getattr__(name: str):  # noqa: N807
    if name in ("build_dag", "plan_summary", "submit_manifest"):
        from . import manifest
        return getattr(manifest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
