# ADR 0010 — go-jsonnet binary for config composition

## Status

Accepted (2026-04-04). Superseded the YAML 3-chain + custom resolver introduced by ADR 0009.

## Context

Phase 1 of the config stack migration (`docs/migration_plan.md`) replaces
`merge_yaml_chain` — a custom deep-merge + dotted-override plumb over a
3-stage YAML chain — with a single `render_config(jsonnet_path, tla)` call.
The composition primitive is jsonnet. Two implementations exist:

- **C++ `jsonnet`** (libjsonnet, PyPI wrapper `jsonnet`) — C++ extension,
  ~1 MB libjsonnet.so, compiled from source by pip on OSC, slow.
- **`go-jsonnet`** — single static Go binary, 10–100× faster on non-trivial
  files, no runtime dep, no compile step.

## Decision

Use the `go-jsonnet` binary, installed to `~/.local/bin/jsonnet` on every
machine that runs the pipeline (OSC login/compute, WSL desktops). Python
access is via `subprocess.run` through `graphids/config/jsonnet.py::render_config`.

Do **not** depend on the `jsonnet` PyPI package. No `[tool.uv]` entry for it.

## Version pins

| Component   | Version  | Source |
|-------------|----------|--------|
| go-jsonnet  | v0.20.0  | [github release tarball](https://github.com/google/go-jsonnet/releases/tag/v0.20.0) |

Install script (login node, no root):

```bash
cd /tmp && curl -sL \
  https://github.com/google/go-jsonnet/releases/download/v0.20.0/go-jsonnet_0.20.0_Linux_x86_64.tar.gz \
  | tar -xz
install -m 755 jsonnet ~/.local/bin/jsonnet
install -m 755 jsonnetfmt ~/.local/bin/jsonnetfmt
rm -f /tmp/jsonnet /tmp/jsonnetfmt
jsonnet --version  # Jsonnet commandline interpreter (Go implementation) v0.20.0
```

`scripts/slurm/submit.sh` hard-fails if the binary is missing — SLURM jobs die at
preamble time with an actionable message rather than much later inside
`render_config`.

## Rationale

1. **Speed.** go-jsonnet renders the full `ablation.yaml` recipe tree (~100
   chains) in under 500 ms total. libjsonnet takes seconds.
2. **No Python dep.** `uv sync` stays identical across machines; jsonnet is
   a pure operational dependency like `sbatch` or `duckdb`.
3. **Same shared $HOME on Pitzer/Ascend/Cardinal.** `~/.local/bin/jsonnet`
   is visible from every compute node in OSC without per-cluster install.
4. **No JVM, no C++ toolchain.** Installs to a user home directory in under
   a second; no OSC module load required.

## Consequences

- `graphids/config/jsonnet.py` is the only site that shells out to the
  binary. Cached path lookup via `functools.lru_cache` avoids re-running
  `shutil.which` on every render (~100× per recipe validation).
- `subprocess` overhead is ~5 ms per render. Not a hot path — CLI chain
  validation runs once per asset at planning time, pipeline runs once per
  training job launch.
- If `subprocess` ever becomes a bottleneck, `_gojsonnet` Python bindings
  exist — same `render_config` signature, no call-site changes.
- Editor integration: `jsonnetfmt --test configs/` runs in `scripts/lint.sh`.
  `.editorconfig` sets 2-space indent + LF for `*.jsonnet,*.libsonnet`.

## Rejected alternatives

- **`jsonnet` PyPI package.** Slower, adds a C++ compile step to every
  `uv sync`, and the `_jsonnet` module has a history of flaky builds on
  RHEL 9 (OSC's base image).
- **CUE / Dhall.** More powerful but larger surface area, steeper learning
  curve, and CUE's schema system overlaps with Phase 2's Pydantic layer.
- **Nix expressions.** Overkill, requires nix-daemon, no OSC policy support.
- **Pure Python composition.** That's what we're replacing — the custom
  deep-merge + dotted-override plumbing was 50+ LOC of footguns.
