"""Cache-build artifacts: pipeline, metadata, vocab, scaler, curriculum scoring.

Everything in this subpackage runs once per cache build (or once at
DataModule setup, for curriculum scoring) and writes durable artifacts
read later by datasets/datamodule. No DataLoader / batching code here —
that lives in ``graphids.core.data.datamodule``.
"""
