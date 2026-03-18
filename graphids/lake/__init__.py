"""Data lake module: shared storage on ESS with DuckDB catalog.

Public API:
    from graphids.lake import write_manifest, cache_lock, rebuild_catalog

Layer placement: alongside config/ (Layer 1). Imports from graphids.config only.
"""

from graphids.lake.locking import cache_lock
from graphids.lake.manifest import read_manifest, verify_manifest, write_manifest

# Lazy import for catalog (requires duckdb)
_lazy = {"rebuild_catalog", "catalog_status"}


def __getattr__(name):
    if name in _lazy:
        from graphids.lake import catalog

        return getattr(catalog, name)
    raise AttributeError(f"module 'graphids.lake' has no attribute {name!r}")
