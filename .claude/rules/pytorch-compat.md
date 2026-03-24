# uv + PyTorch + PyG on OSC — Version Compatibility

This stack has a **three-way version coupling** that must stay in sync. Getting any axis wrong causes segfaults (not import errors — silent C++ ABI mismatches).

## The Constraint Triangle

```
PyTorch version ←→ PyG extension wheels (torch-scatter, torch-sparse, torch-cluster)
      ↕
  CUDA version
```

1. **PyTorch from PyPI** bundles NVIDIA libs automatically. Do NOT use the `download.pytorch.org/whl/cu*` index.
2. **PyG extensions** compiled against specific `torch+cu` combo. Wheels at `https://data.pyg.org/whl/torch-{VERSION}+cu{CUDA}.html`. Only C++ extensions need the flat index.
3. **PyTorch version on PyPI ≠ PyG's torch version tag.** PyPI may ship torch 2.10.0 but PyG only has wheels up to 2.8.0. Mismatched versions **segfault at runtime**.

## Current Pinned Versions (2026-03-04)

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12.4 | OSC `module load python/3.12` |
| PyTorch | 2.8.0 (bundled cu128) | PyPI (default index) |
| torch-scatter | 2.1.2+pt28cu126 | `data.pyg.org/whl/torch-2.8.0+cu126.html` |
| torch-sparse | 0.6.18+pt28cu126 | same flat index |
| torch-cluster | 1.6.3+pt28cu126 | same flat index |
| torch-geometric | 2.7.0 | PyPI |
| RAPIDS | Not integrated — removed. Single uv env. |

## Traps to Avoid

- **`torch>=2.6`** without upper bound: resolves to latest with no PyG wheels. Always pin `<next_major`.
- **`requires-python = ">=3.10"`** without upper bound: uv resolves for 3.13+ where PyG wheels may not exist. Use `<3.13`.
- **uv-managed Python downloads**: Standalone Python builds from uv can segfault on OSC's RHEL 9. Always use OSC's system Python via `uv venv --python /apps/python/3.12/bin/python3`.
- **`[tool.uv] find-links`** for PyG: Use `[[tool.uv.index]]` with `format = "flat"` and `explicit = true` instead.
- **OSC CUDA modules** (cuda/12.6, cudnn/8.x): Not needed when torch comes from PyPI. Only load for RAPIDS or custom CUDA code.
