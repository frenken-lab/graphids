"""Cross-run analysis from MLflow — no torch, safe on login nodes.

Distinct from :mod:`graphids.core.analysis`, which owns per-checkpoint torch
artifacts (CKA / UMAP / loss landscape). This package only reads the MLflow
SQLite catalog and uses scipy; it has no torch dependency.
"""
