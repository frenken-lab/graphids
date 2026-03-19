"""Storage layer: NFS-safe I/O gateway + domain-aware artifact mapper.

Infrastructure layer below all others. No imports from config/, pipeline/, or core/.

Usage:
    from graphids.storage import StorageGateway, ArtifactMapper, open_gateway
"""

from .gateway import StorageGateway
from .mapper import ArtifactMapper, open_gateway
from .paths import (
    lake_cache_dir,
    lake_catalog_path,
    lake_exports_dir,
    lake_raw_dir,
    lake_root_from_env,
    lake_run_dir,
    lake_sweep_dir,
)
