"""Core config paths shared across config modules."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
