# GraphIDS Config System

Jsonnet stages live under the repo-root `configs/` tree and are rendered via
`graphids.config.jsonnet.render(...)`. Pydantic validation runs in
`graphids.config.schemas` before any Lightning instantiation.

## Sources of truth

- `configs/stages/*.jsonnet` — stage composition (autoencoder/supervised/fusion)
- `configs/models/*.libsonnet` + `configs/fusion/**` — model/fusion building blocks
- `configs/resources/submit_profiles.json` — SLURM resource profiles (static, scaling, or composed-via-stages)
<!-- configs/recipes/ + sweep/cartesian expansion deleted 2026-04-12.
     Replacement (campaign manifest + append-only status log) designed in
     ~/plans/graphids-campaign-manifest.md. -->

See `docs/reference/config-architecture.md` for the full config flow.
