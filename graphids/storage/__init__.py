"""Storage layer: NFS-safe I/O gateway + domain-aware artifact mapper.

Infrastructure layer below all others. No imports from config/, pipeline/, or core/.

Usage:
    from graphids.storage import StorageGateway, ArtifactMapper, open_gateway
"""

from .gateway import StorageGateway
from .mapper import ArtifactMapper, open_gateway
from .contracts import (
    EvaluationArtifact,
    PreprocessingArtifact,
    StageArtifact,
    TrainingArtifact,
)
from .paths import (
    lake_cache_dir,
    lake_catalog_path,
    lake_exports_dir,
    lake_raw_dir,
    lake_root_from_env,
    lake_run_dir,
    lake_sweep_dir,
)

# Lazy imports for manifest and catalog (heavy deps)
def __getattr__(name: str):
    if name in ("write_manifest", "read_manifest", "verify_manifest", "Manifest", "ManifestEntry"):
        from . import manifest as _m
        return getattr(_m, name)
    if name in ("rebuild_catalog", "catalog_status"):
        from . import catalog as _c
        return getattr(_c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
