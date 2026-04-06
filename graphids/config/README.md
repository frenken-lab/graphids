# GraphIDS Config System

Jsonnet stages live under the repo-root `configs/` tree and are rendered via
`graphids.config.jsonnet.render_config(...)`. Pydantic validation runs in
`graphids.config.schemas` before any Lightning instantiation.

## Sources of truth

- `configs/stages/*.jsonnet` — stage composition (autoencoder/normal/curriculum/fusion)
- `configs/models/*.libsonnet` + `configs/fusion/**` — model/fusion building blocks
- `configs/resources/job_profiles.json` — static resource profiles per family/scale/stage
- `configs/resources/clusters.json` — cluster → partition/gres mapping
- `configs/recipes/*.jsonnet` — pipeline recipes (sweep dimensions)

See `docs/reference/config-architecture.md` for the full config flow.
