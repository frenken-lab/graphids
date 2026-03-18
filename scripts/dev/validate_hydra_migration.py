#!/usr/bin/env python3
"""Validate Hydra config migration produces correct PipelineConfig for all combinations.

Checks:
1. All 12 model×scale combinations resolve without error
2. Architecture overrides applied correctly (spot-checked per YAML)
3. KD auxiliary composes correctly
4. Nested dict overrides work (E2E_OVERRIDES pattern)
5. Dataset and seed overrides work
"""

import sys

from graphids.config import resolve


def main() -> int:
    errors: list[str] = []

    # --- 1. All 12 base combos ---
    combos = [
        ("vgae", "large"),
        ("vgae", "small"),
        ("gat", "large"),
        ("gat", "small"),
        ("dqn", "large"),
        ("dqn", "small"),
    ]
    for mt, sc in combos:
        try:
            cfg = resolve(mt, sc)
            assert cfg.model_type == mt, f"model_type mismatch: {cfg.model_type}"
            assert cfg.scale == sc, f"scale mismatch: {cfg.scale}"
        except Exception as e:
            errors.append(f"resolve({mt!r}, {sc!r}): {e}")

    # --- 2. Architecture override spot checks ---
    checks = [
        # (model_type, scale, attr_path, expected)
        ("vgae", "large", "vgae.conv_type", "gatv2"),
        ("vgae", "large", "vgae.proj_dim", 48),
        ("vgae", "large", "training.lr", 0.002),
        ("vgae", "large", "training.safety_factor", 0.80),
        ("vgae", "small", "vgae.hidden_dims", (80, 40, 16)),
        ("vgae", "small", "vgae.latent_dim", 16),
        ("vgae", "small", "vgae.heads", 1),
        ("gat", "large", "gat.fc_layers", 1),
        ("gat", "large", "gat.conv_type", "gatv2"),
        ("gat", "large", "training.batch_size", 8192),
        ("gat", "small", "gat.hidden", 24),
        ("gat", "small", "gat.layers", 2),
        ("gat", "small", "training.lr", 0.001),
        ("gat", "small", "training.patience", 50),
        ("dqn", "large", "training.safety_factor", 0.45),
        ("dqn", "small", "dqn.hidden", 160),
        ("dqn", "small", "dqn.layers", 2),
        ("dqn", "small", "dqn.buffer_size", 50000),
    ]
    for mt, sc, attr_path, expected in checks:
        cfg = resolve(mt, sc)
        parts = attr_path.split(".")
        val = cfg
        for p in parts:
            val = getattr(val, p)
        if val != expected:
            errors.append(f"{mt}/{sc} {attr_path}: expected {expected!r}, got {val!r}")

    # --- 3. KD auxiliary ---
    for mt, sc in combos:
        cfg = resolve(mt, sc, auxiliaries="kd_standard")
        if not cfg.has_kd:
            errors.append(f"{mt}/{sc} kd_standard: has_kd is False")
        if cfg.kd is None:
            errors.append(f"{mt}/{sc} kd_standard: kd is None")
        elif cfg.kd.temperature != 4.0:
            errors.append(f"{mt}/{sc} kd_standard: temperature={cfg.kd.temperature}")

    # --- 4. Nested dict overrides (E2E_OVERRIDES pattern) ---
    e2e = {
        "training": {"max_epochs": 2, "batch_size": 32},
        "vgae": {"hidden_dims": [16, 8, 4], "latent_dim": 4},
    }
    cfg = resolve("vgae", "large", **e2e)
    if cfg.training.max_epochs != 2:
        errors.append(f"E2E max_epochs: {cfg.training.max_epochs}")
    if cfg.training.batch_size != 32:
        errors.append(f"E2E batch_size: {cfg.training.batch_size}")
    if cfg.vgae.hidden_dims != (16, 8, 4):
        errors.append(f"E2E hidden_dims: {cfg.vgae.hidden_dims}")
    if cfg.vgae.latent_dim != 4:
        errors.append(f"E2E latent_dim: {cfg.vgae.latent_dim}")

    # --- 5. Dataset and seed overrides ---
    cfg = resolve("vgae", "large", dataset="hcrl_ch", seed=123)
    if cfg.dataset != "hcrl_ch":
        errors.append(f"dataset override: {cfg.dataset}")
    if cfg.seed != 123:
        errors.append(f"seed override: {cfg.seed}")

    # --- 6. Top-level overrides ---
    cfg = resolve("vgae", "large", lake_root="/tmp/test")
    if cfg.lake_root != "/tmp/test":
        errors.append(f"lake_root override: {cfg.lake_root}")

    # --- 7. Defaults preserved (fields not in YAML use Pydantic defaults) ---
    cfg = resolve("dqn", "large")
    if cfg.dqn.hidden != 576:  # Pydantic default, not overridden
        errors.append(f"dqn large default hidden: {cfg.dqn.hidden}")
    if cfg.training.lr != 0.003:  # Pydantic default, not overridden for dqn_large
        errors.append(f"dqn large default lr: {cfg.training.lr}")

    # --- Report ---
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print(f"ALL PASSED — {len(combos)} base combos, {len(checks)} spot checks,")
    print(f"  {len(combos)} KD combos, nested overrides, dataset/seed/top-level overrides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
