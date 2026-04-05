# Phase 1 — Jsonnet Port (Detailed Implementation Plan)

> Expands §"Phase 1 — Jsonnet" of `docs/migration_plan.md`.
> Target audience: whoever actually does the port (likely next session).
> Prerequisite read: `docs/reference/3-chain.md` (current 3-handoff config flow).

---

## 1. Goal

Replace the YAML chain + `merge_yaml_chain` + override plumbing with a single
`render_config(jsonnet_path, tla) → dict` call. **Full migration, single PR.**
The old code and the YAML files are deleted in the same commit that introduces
jsonnet. `merge_yaml_chain`, `deep_merge`, `apply_dotted_overrides`,
`TrainingContract.to_override_dict`, `TrainingContract.resolve_config_files`,
`StageConfig.config_files`, `TrainingSpec.config_files`,
`TrainingSpec.runtime_overrides`, and every YAML under
`graphids/config/{stages,models,fusion,defaults/trainer.yaml}/` are gone.

Git history is the rollback. No shadow path, no dual-write, no feature flag.

What stays (Phases 3–4 strip these, not Phase 1):

- `LightningCLI`, `GraphIDSCLI`, `build_cli`, `_lightning.py`, `jsonargparse`
- `cli.LINK_TARGETS`, forced callbacks, `patch_config_paths`
- `expand_recipe_configs`, `enumerate_assets`, `TrainingRunConfig`
- `ConfigResolver` (interior rewritten, outer contract identical)
- `topology.py`, identity hashing, planning, resource profiles
- `config/datasets/*.yaml`, `config/resources/**`, `config/matrix/axes.yaml`,
  `config/defaults/{global,io}.yaml`, `config/recipes/*.yaml` — all survive
  Phase 1 (separate consumers, not in the CLI chain)
- `config/stages/analyze_*.yaml` — consumed by `Analyzer`, not LightningCLI.
  Port opportunistically in §7 step 6; low priority.

### Hard exit criteria

1. `configs/` tree at repo root contains jsonnet sources covering every YAML
   file that currently feeds `merge_yaml_chain` (the 20 files enumerated in §4.1).
2. `graphids/config/jsonnet.py::render_config(jsonnet_path, tla)` is the sole
   source of the merged dict handed to `jsonargparse.parse_object`. Both the
   pipeline path (`train_entrypoint._instantiate_from_spec`) and the dev path
   (`python -m graphids fit --config ...`) go through it.
3. `grep -r merge_yaml_chain graphids/ tests/` returns nothing.
   `grep -r "runtime_overrides\|config_files\b" graphids/core/contracts/ graphids/orchestrate/` returns nothing.
   `ls graphids/config/stages/ graphids/config/models/ graphids/config/fusion/` returns `No such file or directory`.
4. End-to-end smoke: `dg launch --assets '*' --partition 'hcrl_ch|42'` under
   `smoke_test.yaml` runs train → test → analyze for at least one asset per
   stage (autoencoder, normal, curriculum, fusion) to COMPLETED. Checkpoints
   land in the lake, `run_record.json` sidecars are written, `pipeline-status`
   shows green.
5. `python -m graphids fit --config configs/stages/autoencoder.jsonnet
   --data.init_args.dataset hcrl_ch` runs a 1-epoch fit on a gpu SLURM job
   without crashing. Dev path override via `--model.init_args.lr=0.01` still
   reaches the model.
6. `python -m graphids.orchestrate validate` passes against `ablation.yaml`,
   `smoke_test.yaml`, `final_eval.yaml` (validates every unique chain — same
   gate as today, now rendering through jsonnet).
7. A one-shot parity test (§8) runs BEFORE the delete commit, proving the
   jsonnet output matches `merge_yaml_chain` output for every chain produced
   by every recipe. The test is **deleted in the same delete commit** — it
   cannot exist after `merge_yaml_chain` is gone.

---

## 2. Pre-work

### 2.1 Install `go-jsonnet`

- OSC has no jsonnet module (`module avail jsonnet` empty — confirmed).
- Do NOT use the `jsonnet` PyPI package (C++ bindings, libjsonnet.so, slow).
- Install `go-jsonnet` binary to `~/.local/bin/jsonnet`:
  ```bash
  curl -L https://github.com/google/go-jsonnet/releases/download/v0.20.0/go-jsonnet_0.20.0_Linux_x86_64.tar.gz \
    | tar -xz -C ~/.local/bin jsonnet jsonnetfmt
  chmod +x ~/.local/bin/jsonnet ~/.local/bin/jsonnetfmt
  jsonnet --version  # should print v0.20.0
  ```
- Add `jsonnet` and `jsonnetfmt` version pins to a new `docs/decisions/0010-jsonnet-binary.md` ADR (installed-from-github, not a packaged dep).
- WSL desktop/laptop: same binary works. Put the install step into
  `~/dotfiles/run_once_install_jsonnet.sh` so chezmoi keeps all three machines
  in sync.
- CI concern: tests gate on jsonnet availability. `tests/config/test_jsonnet_parity.py`
  should `pytest.skip` with an explicit reason (not a silent skip) if
  `shutil.which("jsonnet")` is None. The submit.sh `tests` job preamble should
  `command -v jsonnet || exit 1` so SLURM tests fail loudly when the binary is
  missing.

### 2.2 Editor / formatter config

- `jsonnetfmt --test configs/` in `scripts/lint.sh` alongside `ruff check`.
- `.editorconfig`: `*.jsonnet,*.libsonnet` → 2-space indent, LF.
- No new pyproject dep.

---

## 3. High-level architecture (post-migration)

```
┌──────────────────────────────────────────────────────┐
│ expand_recipe_configs (unchanged)                    │
│   └─ enumerate_assets → list[StageConfig]            │
│        StageConfig.jsonnet_path: str  (REPLACES      │
│        StageConfig.jsonnet_tla:  dict  config_files) │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ ConfigResolver.resolve  (internals rewritten)        │
│   ├─ build TLA dict from trainer/stage/kd overrides  │
│   ├─ get_resources + apply_resource_overrides        │
│   └─ _build_spec(jsonnet_path=..., jsonnet_tla=...)  │
│                                                      │
│ ConfigResolver.validate_cli_chain                    │
│   ├─ render_config(jsonnet_path, tla)                │
│   ├─ _validate_cross_fields(rendered)                │
│   └─ schema_parser().parse_object(rendered)          │
└──────────────────────┬───────────────────────────────┘
                       │
        ── JSON envelope (TrainingSpec) ──
                       │
┌──────────────────────▼───────────────────────────────┐
│ SLURM side: from-spec --phase train                  │
│   train_entrypoint._instantiate_from_spec            │
│     merged = render_config(spec.jsonnet_path,        │
│                            spec.jsonnet_tla)         │
│     write_yaml(merged, run_dir/config_snapshot.yaml) │
│     build_cli(merged)  ← jsonargparse still runs     │
│                          here; same code path        │
└──────────────────────────────────────────────────────┘
```

Single path. `render_config` is on the hot path for every pipeline run and
every dev-mode fit. `jsonargparse.parse_object` still runs at the bottom — it
receives a dict rendered by jsonnet instead of a dict merged from YAML, but
everything from `build_cli` downward is byte-identical.

---

## 4. Current state inventory (what we must reproduce)

### 4.1 YAML files that enter the CLI chain

Verified from `TrainingContract.resolve_config_files` + `CLI_KWARGS.parser_kwargs.default_config_files`:

| Chain layer | Files | Notes |
|---|---|---|
| jsonargparse implicit default | `defaults/trainer.yaml` | Applied by `CLI_KWARGS.parser_kwargs.default_config_files` for every subcommand |
| Stage | `stages/{autoencoder,normal,curriculum,fusion}.yaml` | 4 files |
| Model base | `models/{vgae,gat,dgi}/base.yaml` | 3 files |
| Model scale | `models/{vgae,gat,dgi}/scales/{small,large}.yaml` | 6 files |
| KD overlay (conditional) | `models/{vgae,gat}/kd.yaml` | 2 files, appended only when `auxiliaries` present |
| Fusion base | `fusion/base.yaml` | empty `{}` placeholder |
| Fusion method | `fusion/methods/{bandit,dqn,mlp,weighted_avg}.yaml` | 4 files |

**Not in chain** (deliberately excluded per `ops.py:94`): `fusion/scales/{small,large}.yaml`
— orchestrator metadata only, read elsewhere.

**Not in chain** (separate consumer): `datasets/*.yaml`, `resources/**`, `matrix/axes.yaml`,
`defaults/{global,io}.yaml`, `stages/analyze_*.yaml`. Phase 1 scope is the 20 files above.

### 4.2 Override sources that reach the merged dict

Verified from `TrainingContract.to_override_dict` + `ConfigResolver.resolve`:

1. **Identity/path (always present)**
   - `data.init_args.dataset` — from partition
   - `seed_everything` — from partition
   - `trainer.default_root_dir` — from `PathContext.run_dir`
2. **From `StageConfig.model_init_overrides`** (set by planning based on `identity_keys ∩ model_keys`):
   - `model.init_args.conv_type` (vgae, gat)
   - `model.init_args.variational` (vgae only)
   - `model.init_args.loss_fn` (gat only)
3. **Upstream checkpoints** (from `spec.upstream_ckpt_paths`):
   - `data.init_args.vgae_ckpt_path` (student loading vgae/dgi teacher)
   - `data.init_args.gat_ckpt_path` (fusion loading gat teacher)
4. **Recipe-level `trainer_overrides`** (flat dotted keys, e.g. `trainer.max_epochs: "50"`)
5. **Recipe-level `stage_overrides[stage]`** (same shape, stage-scoped)
6. **KD payload** (`model.init_args.auxiliaries`, emitted as a JSON string by
   `ConfigResolver` and re-parsed by jsonargparse — keep this behavior verbatim
   in Phase 1).

### 4.3 Stringification footgun

`to_override_dict` casts **every** override value to `str` (bools become
`"true"/"false"`, ints become `"50"`). jsonargparse coerces back. Jsonnet is
natively typed, so `render_config` emits real ints and bools. The parity harness
must compare with `str(naive) == str(jsonnet)` the same way
`test_merge_parity.py` already does — see §9.3.

### 4.4 Linked arguments (jsonargparse-only)

`cli.LINK_TARGETS` does `data.init_args.dataset → model.init_args.dataset`,
`seed_everything → model.init_args.seed`, etc. These fire **inside
jsonargparse**, not in the merged YAML dict. `render_config` does not need to
replicate them — the parity target is the pre-parse merged dict, not the
post-parse Namespace. (Phase 3 will need to replicate LINK_TARGETS explicitly
when we strip LightningCLI.)

---

## 5. Target file layout

```
configs/                            # NEW, at repo root (not under graphids/)
├── _lib/
│   ├── defaults.libsonnet          # trainer, checkpoint, early_stopping baselines
│   └── helpers.libsonnet           # kd_overlay(model_family, aux_list) et al.
├── stages/
│   ├── autoencoder.jsonnet         # function(...) → merged dict
│   ├── normal.jsonnet
│   ├── curriculum.jsonnet
│   └── fusion.jsonnet
├── models/
│   ├── vgae.libsonnet              # { base, scales: { small, large }, kd }
│   ├── gat.libsonnet
│   └── dgi.libsonnet
└── fusion/
    ├── base.libsonnet
    └── methods/
        ├── bandit.libsonnet
        ├── dqn.libsonnet
        ├── mlp.libsonnet
        └── weighted_avg.libsonnet
```

Lives at `configs/` (repo root), **not** under `graphids/config/`, so the old
tree stays 100% untouched. Phase 4's removal step is `rm -r graphids/config/{stages,models,fusion,defaults}/`
and no import rewrites except the resolver.

### 5.1 Stage file shape

Every `stages/*.jsonnet` is a top-level function of TLAs:

```jsonnet
// configs/stages/autoencoder.jsonnet
local defaults = import '../_lib/defaults.libsonnet';
local vgae = import '../models/vgae.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';

function(
  dataset,                  // required — from partition
  seed,                     // required — from partition
  run_dir,                  // required — from PathContext
  scale = 'small',          // from planning (identity_keys ∩ model_keys)
  conv_type = 'gatv2',
  variational = true,
  auxiliaries = [],         // list of KD entries (empty = no KD overlay)
  vgae_ckpt_path = null,
  trainer_overrides = {},   // recipe-level, flat dotted-key dict, string values
  stage_overrides = {},     // recipe-level, flat dotted-key dict, string values
)
  defaults.trainer
  + defaults.checkpoint
  + defaults.early_stopping
  + vgae.base
  + vgae.scales[scale]
  + (if std.length(auxiliaries) > 0 then vgae.kd else {})
  + {
      seed_everything: seed,
      trainer+: { default_root_dir: run_dir },
      data+: { init_args+: {
        dataset: dataset,
      } + (if vgae_ckpt_path != null then { vgae_ckpt_path: vgae_ckpt_path } else {}) },
      model+: { init_args+: {
        conv_type: conv_type,
        variational: variational,
      } + (if std.length(auxiliaries) > 0 then { auxiliaries: auxiliaries } else {}) },
    }
  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
```

**Key pattern:** `+:` deep-merges, `+` shallow-merges-with-last-wins. Every layer
that currently deep-merges via `deep_merge` in `yaml_utils.py` uses `+:` in
jsonnet. The `apply_dotted` helper emulates `apply_dotted_overrides` and must be
written as a jsonnet function — see §7.1 for the implementation sketch.

### 5.2 Model libsonnet shape

```jsonnet
// configs/models/vgae.libsonnet
{
  base: {
    model: {
      class_path: 'graphids.core.models.autoencoder.vgae.VGAEModule',
      init_args: {
        conv_type: 'gatv2',
        edge_dim: 11,
        variational: true,
        mask_ratio: 0.3,
        k_neg: 32,
        canid_weight: 0.1,
        nbr_weight: 0.05,
        kl_weight: 0.01,
        lr: 0.002,
        compile_model: true,
        gradient_checkpointing: true,
        auxiliaries: [],
      },
    },
    data: {
      class_path: 'graphids.core.preprocessing.datamodule.CANBusDataModule',
      init_args: {
        window_size: 100,
        stride: 100,
        val_fraction: 0.2,
        batch_size: 8192,
        num_workers: null,           // preserve null — see §9.1
        dynamic_batching: true,
      },
    },
  },
  scales: {
    small: { model+: { init_args+: {
      scale: 'small',
      hidden_dims: [80, 40, 16],
      latent_dim: 16,
      heads: 1,
      embedding_dim: 4,
      dropout: 0.1,
      proj_dim: 32,
    } } },
    large: { /* ... */ },
  },
  kd: { model+: { init_args+: { auxiliaries: [
    { type: 'kd', alpha: 0.7, vgae_latent_weight: 0.5, vgae_recon_weight: 0.5 },
  ] } } },
}
```

**Merge semantics:** jsonnet's `+:` is exactly the deep-merge `yaml_utils.deep_merge`
implements — recursive dict merge, last wins, lists replace. GAT base.yaml's
`data.init_args.num_workers: 4` overriding stage YAML's `num_workers: null`
becomes `gat.base + { data+: { init_args+: { num_workers: 4 } } }`, and stages
that import `vgae.base` get `null`. Identical semantics.

### 5.3 Defaults libsonnet

Port `defaults/trainer.yaml` verbatim:

```jsonnet
// configs/_lib/defaults.libsonnet
{
  trainer: { trainer: {
    accelerator: 'auto',
    devices: 'auto',
    precision: '16-mixed',
    max_epochs: 300,
    gradient_clip_val: 1.0,
    log_every_n_steps: 50,
  } },
  checkpoint: { checkpoint: {
    monitor: 'val_loss',
    mode: 'min',
    save_top_k: 1,
    save_last: true,
    filename: 'best_model',
  } },
  early_stopping: { early_stopping: {
    monitor: 'val_loss',
    mode: 'min',
    patience: 100,
  } },
}
```

---

## 6. Python shim

New module: `graphids/config/jsonnet.py`. ≤ 80 LOC. Lazy — not imported at
package import time. No torch dep.

```python
"""Jsonnet config rendering shim.

Shells out to the `jsonnet` binary (go-jsonnet) and returns the parsed JSON
as a dict. Top-level arguments are passed through --tla-code so jsonnet
receives real values (not strings). No other IO side effects.

Phase 1 only — no consumer inside graphids.* yet. See docs/phase1_jsonnet.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


class JsonnetError(RuntimeError):
    """Raised when the jsonnet binary returns non-zero."""


@lru_cache(maxsize=1)
def _jsonnet_bin() -> str:
    bin_path = shutil.which("jsonnet")
    if not bin_path:
        raise JsonnetError(
            "jsonnet binary not found on PATH. Install go-jsonnet: "
            "see docs/phase1_jsonnet.md §2.1"
        )
    return bin_path


def render_config(
    jsonnet_path: str | Path,
    tla: dict[str, Any] | None = None,
    *,
    jpath: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Render a .jsonnet file with top-level arguments and return the parsed dict.

    Every key in ``tla`` is passed as ``--tla-code <k>=<json.dumps(v)>`` so
    types round-trip (ints stay ints, bools stay bools, lists stay lists).
    Strings are also passed as --tla-code (wrapped in JSON quotes) — NOT
    --tla-str, which bypasses JSON parsing and would choke on embedded quotes.
    """
    cmd = [_jsonnet_bin()]
    for p in jpath:
        cmd += ["--jpath", p]
    for k, v in (tla or {}).items():
        cmd += ["--tla-code", f"{k}={json.dumps(v)}"]
    cmd.append(str(jsonnet_path))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise JsonnetError(
            f"jsonnet failed for {jsonnet_path}:\n{result.stderr.strip()}"
        )
    return json.loads(result.stdout)
```

### 6.1 Why subprocess, not the `jsonnet` Python package

- `jsonnet` (PyPI) wraps libjsonnet (C++), adds a compile-step dep we don't want on OSC.
- go-jsonnet is 10–100× faster on nontrivial files (pure Go, no libjsonnet overhead).
- subprocess is ~5 ms overhead. Parity harness renders ~100 configs = 500 ms total. Fine.
- If subprocess cost ever matters (it won't), swap in `_gojsonnet` bindings later — same `render_config` signature.

### 6.2 No caching beyond `_jsonnet_bin`

Each render is cheap and must reflect file edits. LRU the binary path lookup, nothing else.

---

## 7. Step-by-step migration order

Single feature branch `phase1-jsonnet`. Commits listed below in order. Each
commit should be runnable but only the last one satisfies the exit criteria.
The migration is atomic at PR level — merged as one unit or reverted as one
unit.

### Commit 1 — Tooling + empty jsonnet tree

- Install `go-jsonnet` via §2.1 (this is a dotfiles change, not a repo commit)
- `configs/` directory created with placeholder `.gitkeep`
- `graphids/config/jsonnet.py` — full `render_config` (§6)
- `docs/decisions/0010-jsonnet-binary.md` — ADR recording version pin +
  install script + why not the Python binding
- `.editorconfig` updated for `.jsonnet/.libsonnet`
- `scripts/submit.sh` tests preamble: `command -v jsonnet || { echo "install go-jsonnet — see docs/phase1_jsonnet.md §2.1"; exit 1; }`
- `pyproject.toml` — no change (jsonnet is an external binary, not a Python dep)

**Exit:** `python -c "from graphids.config.jsonnet import render_config"` works.
`render_config("/nonexistent.jsonnet", {})` raises `JsonnetError`.

### Commit 2 — Spike: one chain, bidirectional

Port exactly `autoencoder + vgae/small`. Prove render_config matches
merge_yaml_chain end-to-end on a single chain BEFORE touching any other files.

- `configs/_lib/defaults.libsonnet` (trainer/checkpoint/early_stopping)
- `configs/_lib/helpers.libsonnet` (`apply_dotted`, see §7.6 below)
- `configs/models/vgae.libsonnet` (base + scales.small only, no kd block yet)
- `configs/stages/autoencoder.jsonnet` (full TLA signature from §5.1)
- `tests/config/test_jsonnet_parity.py` — ONE parametrization:
  `autoencoder + vgae/small + hcrl_ch + seed 42`. Calls both
  `merge_yaml_chain(cfg.config_files, to_override_dict(spec))` AND
  `render_config(jsonnet_path, tla)`, diffs them via the stringification-tolerant
  comparator from §8, and also runs `schema_parser().parse_object(rendered)`.

**Exit:** `scripts/submit.sh tests -k jsonnet_parity` passes on a CPU SLURM job.
If it fails, STOP. Do not port more files — debug the spike first. Every
gotcha in §9 will bite easier on a 1-chain surface than on 100 chains.

### Commit 3 — Port remaining models + stages + fusion

- `models/vgae.libsonnet` — add `large` + `kd` block
- `models/gat.libsonnet` — base (including `data.init_args.num_workers: 4`
  override from gat/base.yaml), scales (small, large), kd block
- `models/dgi.libsonnet` — base, scales, no kd
- `fusion/base.libsonnet` — trainer overrides (cpu, precision 32, max_epochs
  1500, FusionDataModule)
- `fusion/methods/{bandit,dqn,mlp,weighted_avg}.libsonnet` — one per method
- `stages/normal.jsonnet` — GAT + CANBusDataModule
- `stages/curriculum.jsonnet` — GAT + CurriculumDataModule (note the different
  data `class_path` from `normal.jsonnet`)
- `stages/fusion.jsonnet` — dispatches on `fusion_method` TLA via `fusion.methods[method]`
- Extend `test_jsonnet_parity.py` parametrization to cover every chain from
  `smoke_test.yaml`, `ablation.yaml`, `final_eval.yaml` (dedupe by chain-key
  exactly as `validate.py:76-82` does — ~100 unique chains total)

**Note on `fusion/scales/*.yaml`:** excluded from the CLI chain today
(`ops.py:94`). Phase 1 also excludes it — jsonnet does not emit those fields.
If the orchestrator reads them separately, leave the YAML files alone for now.
Port to `fusion/scales.libsonnet` only if an orchestrator consumer is
identified (check `component.py`, `slurm.py`).

**Exit:** `scripts/submit.sh tests -k jsonnet_parity` passes all ~100
parametrizations. Zero prod code touched yet — this is still spike mode.

### Commit 4 — Rewrite contracts and planning

This is the cut-over commit. Touch the type carriers:

- `graphids/core/contracts/models.py::TrainingSpec`:
  - DELETE fields: `config_files: tuple[str, ...]`, `runtime_overrides: dict[str, Any]`
  - ADD fields: `jsonnet_path: str`, `jsonnet_tla: dict[str, Any]`
- `graphids/core/contracts/ops.py::TrainingContract`:
  - DELETE methods: `resolve_config_files`, `to_override_dict`, `_CKPT_FLAG_BY_MODEL`
  - ADD method: `build_tla_dict(stage_cfg, *, dataset, seed, run_dir, upstream_ckpts) -> dict[str, Any]` — replaces `to_override_dict`. Returns the typed TLA dict the stage jsonnet function expects. No stringification.
  - ADD method: `resolve_jsonnet_path(stage, *, fusion_method=None) -> str` — returns e.g. `"configs/stages/autoencoder.jsonnet"`. Replaces `resolve_config_files`. Takes one arg because jsonnet stages don't need chain composition.
- `graphids/orchestrate/planning.py::StageConfig`:
  - DELETE `config_files: tuple[str, ...]`
  - ADD `jsonnet_path: str`
  - `enumerate_assets` builds StageConfigs with `jsonnet_path=TrainingContract.resolve_jsonnet_path(stage, fusion_method=merged.fusion_method)`
- `graphids/orchestrate/resolve.py::ConfigResolver.resolve`:
  - Stops building `runtime_overrides` dict
  - Builds `jsonnet_tla` via `TrainingContract.build_tla_dict(cfg, ...)`, which
    packs recipe `trainer_overrides`, `stage_overrides`, `kd_overrides`,
    `model_init_overrides`, upstream ckpts, dataset, seed, run_dir into a
    single typed dict
  - `TrainingSpec(..., jsonnet_path=cfg.jsonnet_path, jsonnet_tla=tla)`
- `graphids/orchestrate/resolve.py::ConfigResolver.validate_cli_chain`:
  - Replace `merge_yaml_chain(spec.config_files, to_override_dict(spec))` with
    `render_config(spec.jsonnet_path, spec.jsonnet_tla)`
  - Downstream `_validate_cross_fields` and `parse_object` get the same dict
    shape they get today — rules are unchanged except the one that reads
    `spec.runtime_overrides` (fusion RL batch_size check): update to read from
    `spec.jsonnet_tla["trainer_overrides"]` and `["stage_overrides"]` dicts
- `graphids/orchestrate/resolve.py::_check_fusion_rl_batch_size_override`:
  - Update to read `spec.jsonnet_tla.get("trainer_overrides", {})` and
    `spec.jsonnet_tla.get("stage_overrides", {})` instead of
    `spec.runtime_overrides`
- `graphids/core/train_entrypoint.py::_instantiate_from_spec`:
  - Replace `merge_yaml_chain(spec.config_files, to_override_dict(spec))`
    with `render_config(spec.jsonnet_path, spec.jsonnet_tla)`
  - Keep `write_yaml(merged, rd / "config_snapshot.yaml")` — still useful as
    a post-hoc debugging artifact
  - Keep `build_cli(merged)` — unchanged
- `graphids/core/train_entrypoint.py::run_training_from_spec`:
  - Auto-resume branch currently sets `runtime_overrides["ckpt_path"] = ...`.
    Change to `jsonnet_tla["ckpt_path"] = ...`. Stages that accept a
    `ckpt_path` TLA propagate it to `trainer.fit(ckpt_path=...)`; the dict
    path gets threaded back out of `spec.jsonnet_tla` in the `trainer.fit`
    call below.
- `graphids/commands/profile.py`:
  - `TrainingContract.resolve_config_files` call → `resolve_jsonnet_path`
  - `runtime_overrides={"trainer.profiler": "simple"}` → `jsonnet_tla={
    ..., "trainer_overrides": {"trainer.profiler": "simple"}}`
- `graphids/_lightning.py::_BOOTSTRAP`:
  - Replace the YAML `--config` chain with `--config configs/stages/autoencoder.jsonnet`
  - OR rewrite as programmatic dict construction (simpler; `schema_parser`
    only needs a parseable object, not a specific file). Pick whichever is
    less invasive. Probably: render one chain to a temp file once, pass that.
- `graphids/_lightning.py::CLI_KWARGS.parser_kwargs.default_config_files`:
  - DELETE the `"graphids/config/defaults/trainer.yaml"` entry. The trainer
    defaults are now baked into every jsonnet stage via
    `import '_lib/defaults.libsonnet'`. Dev path no longer needs the implicit
    inject.
- `graphids/cli.py::run_lightning`:
  - Add a `.jsonnet` preprocessor — see §9.6 for the full implementation. If
    any `--config` arg ends in `.jsonnet`, render it to a temp YAML (via
    `render_config` + `write_yaml`) and substitute the path before handing
    off to `GraphIDSCLI`. Power-users keep their `--model.init_args.lr=0.01`
    override syntax.

Parity test still runs green at this point — it's still reading
`spec.config_files` internally (from the NEW `StageConfig.jsonnet_path`-equivalent
construction path). Actually no: once the field is renamed, the parity test
*can't* call `merge_yaml_chain` anymore. This commit is the point of no return.

**Revised sub-ordering for Commit 4:** keep the parity test alive for ONE extra
commit (Commit 3.5) by temporarily having `StageConfig` carry BOTH
`config_files` AND `jsonnet_path`. The parity test still works. Then Commit 4
deletes `config_files` and the parity test in one shot. This is the ONLY
dual-carry state allowed in Phase 1, and it lives for one commit, purely to
prove the rename doesn't break anything.

**Exit:** `python -m graphids.orchestrate validate --recipe
graphids/config/recipes/smoke_test.yaml` passes. No grep hits for
`merge_yaml_chain` in `graphids/orchestrate/` or `graphids/core/`.

### Commit 5 — Delete the YAML files and dead code

Now and only now, delete:

- `rm -r graphids/config/stages/`
- `rm -r graphids/config/models/`
- `rm -r graphids/config/fusion/`
- `rm graphids/config/defaults/trainer.yaml`
- From `graphids/config/yaml_utils.py`: delete `deep_merge`,
  `apply_dotted_overrides`, `merge_yaml_chain`. Keep `read_yaml`, `write_yaml`
  (still used by recipes, resources, datasets, snapshots).
- `rm tests/config/test_merge_parity.py`
- `rm tests/config/test_jsonnet_parity.py` (one-shot migration test, done)
- Trim `tests/config/test_yaml_utils.py` to the read/write tests only (delete
  any `merge_yaml_chain` / `deep_merge` / `apply_dotted` tests)
- `graphids/config/{VALIDATION_CHECKLIST.md,CONFIG_REFERENCE.md,README.md}` —
  rewrite or delete. Most of the YAML-chain guidance is obsolete. Probably
  delete wholesale and let `docs/phase1_jsonnet.md` + updated
  `docs/reference/config-architecture.md` cover it.

Add a small `tests/config/test_jsonnet_render.py` as a permanent regression
guard — NOT a parity test (there's nothing to parity against). Just asserts
`render_config` produces a dict with `trainer`, `model`, `data` top-level keys
for each stage with default TLAs. Three assertions, runs in < 1 s. Keeps a
lightweight smoke on the jsonnet side without reintroducing the parity
machinery.

**Exit:** all four exit criteria from §1 are met. `grep -r merge_yaml_chain
graphids tests` returns nothing. Directory `graphids/config/stages` does not
exist. `dg launch smoke_test` runs one asset per stage to COMPLETED.

### Commit 6 — Documentation

- Rewrite `docs/reference/config-architecture.md` to describe the jsonnet tree,
  the `render_config` shim, and the TLA surface area. Delete the YAML-chain
  sections.
- Update `docs/reference/3-chain.md`:
  - Handoff 1 step "`merge_yaml_chain + parse_object`" → "`render_config + parse_object`"
  - Handoff 3 step "`merge_yaml_chain(config_files, overrides)`" → "`render_config(jsonnet_path, jsonnet_tla)`"
  - Table "Where validation catches what": update the "Caught at" column
    entries that currently say "via merge_yaml_chain" to "via render_config"
- Update `.claude/rules/config-system.md` — replace the "LightningCLI +
  jsonargparse + plain YAML" intro with "jsonnet composition + LightningCLI +
  jsonargparse". The file layout section needs a full rewrite — just link to
  `docs/phase1_jsonnet.md §5` for the tree.
- Update `CLAUDE.md` (project root) — the "Key Commands" section mentions
  `--config graphids/config/stages/...yaml`. Change `.yaml` → `.jsonnet` and
  `graphids/config/stages/` → `configs/stages/`.
- Update `PLAN.md` with Phase 1 completion + handoff note for Phase 2.

---

### 7.1 `apply_dotted` helper (in the spike commit)

The trickiest jsonnet primitive. Must reproduce `yaml_utils.apply_dotted_overrides`:
split dotted key on `.`, nest into an object, deep-merge all of them.

```jsonnet
// configs/_lib/helpers.libsonnet
{
  // {"trainer.max_epochs": "50", "data.init_args.num_workers": "4"}
  // → {trainer+: {max_epochs: "50"}, data+: {init_args+: {num_workers: "4"}}}
  apply_dotted(overrides)::
    std.foldl(
      function(acc, key) acc + $._nest(std.split(key, '.'), overrides[key]),
      std.objectFields(overrides),
      {},
    ),

  _nest(path, value)::
    if std.length(path) == 1
    then { [path[0]]: value }
    else { [path[0]]+: $._nest(path[1:], value) },
}
```

**Strings pass through unchanged.** Current Python side flattens via
`_flatten_dict` in `recipe_expand.py:57` which stringifies everything. The
TLA dict inherits that — keys are dotted strings, values are strings. Jsonnet
echoes them verbatim. jsonargparse coerces at parse time (same as today).

### 7.2 `build_tla_dict` (Commit 4, new method on TrainingContract)

Replaces `to_override_dict`. Returns a typed dict matching the stage jsonnet
function signatures from §5.1:

```python
@classmethod
def build_tla_dict(
    cls,
    stage_cfg: "StageConfig",
    *,
    dataset: str,
    seed: int,
    run_dir: str,
    upstream_ckpts: dict[str, str],
    auxiliaries: list[dict] | None = None,
) -> dict[str, Any]:
    tla: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "run_dir": run_dir,
        "scale": stage_cfg.scale,
        "trainer_overrides": dict(stage_cfg.trainer_overrides),
        "stage_overrides": dict(stage_cfg.stage_overrides),
    }
    # model_init_overrides → flat TLAs the stage understands
    for key in ("conv_type", "variational", "loss_fn"):
        if key in stage_cfg.model_init_overrides:
            val = stage_cfg.model_init_overrides[key]
            # variational is bool-valued; others string
            if key == "variational":
                tla[key] = val in (True, "true", "True")
            else:
                tla[key] = val
    # fusion method
    if stage_cfg.stage == "fusion":
        tla["fusion_method"] = stage_cfg.resource_model or stage_cfg.model_type
    # upstream checkpoints
    for upstream_asset, ckpt_path in upstream_ckpts.items():
        model_family = stage_cfg.upstream_model_families.get(upstream_asset)
        if model_family in ("vgae", "dgi"):
            tla["vgae_ckpt_path"] = ckpt_path
        elif model_family == "gat":
            tla["gat_ckpt_path"] = ckpt_path
    # KD auxiliaries (JSON-serializable list of dicts)
    if auxiliaries:
        tla["auxiliaries"] = list(auxiliaries)
    elif stage_cfg.kd_overrides:
        tla["auxiliaries"] = [dict(stage_cfg.kd_overrides)]
    return tla
```

This is the single mapping between Python-side `StageConfig` and the jsonnet
stage function signature. Anything else in planning output that a stage needs
gets plumbed through here.

---

## 8. Parity harness (one-shot, deleted in Commit 5)

Lives only during Commits 2, 3, 3.5. Commit 4 renames the fields that feed
it; Commit 5 deletes the file alongside `merge_yaml_chain`. **This test is
not a permanent regression guard** — its entire purpose is to prove the
port is correct before flipping the contract.

One file, one test function, parametrized over every (config, dataset, seed)
combination produced by the Phase-1-in-scope recipes. Same shape as
`test_merge_parity.py` so the review surface is familiar.

```python
"""Jsonnet parity harness — proves render_config ≡ merge_yaml_chain.

Phase 1 exit gate. See docs/phase1_jsonnet.md §1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from graphids.config import CONFIG_DIR, PIPELINE_YAML, expand_recipe_configs
from graphids.config.jsonnet import render_config
from graphids.config.yaml_utils import merge_yaml_chain
from graphids.core.contracts import TrainingContract
from graphids.orchestrate.planning import enumerate_assets

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"

_JSONNET_STAGES = {
    "autoencoder": _CONFIGS / "stages/autoencoder.jsonnet",
    "normal":      _CONFIGS / "stages/normal.jsonnet",
    "curriculum":  _CONFIGS / "stages/curriculum.jsonnet",
    "fusion":      _CONFIGS / "stages/fusion.jsonnet",
}

_RECIPES = ["smoke_test.yaml", "ablation.yaml", "final_eval.yaml"]


def _chain_params():
    """Enumerate (recipe, asset_name, dataset, seed) unique up to chain-key."""
    seen: set[tuple] = set()
    out = []
    for recipe_name in _RECIPES:
        raw = yaml.safe_load((CONFIG_DIR / "recipes" / recipe_name).read_text())
        recipe = expand_recipe_configs(raw)
        datasets = raw.get("selection", {}).get("datasets") or ["hcrl_ch"]
        seeds = recipe["sweep"]["seeds"]
        for cfg in enumerate_assets(PIPELINE_YAML, recipe):
            for dataset in datasets:
                for seed in seeds:
                    key = (
                        tuple(cfg.config_files)
                        + tuple(sorted(cfg.model_init_overrides.items()))
                        + (dataset, seed)
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((recipe_name, cfg, dataset, seed))
    return out


@pytest.mark.slurm
@pytest.mark.skipif(
    shutil.which("jsonnet") is None,
    reason="go-jsonnet binary not installed — see docs/phase1_jsonnet.md §2.1",
)
@pytest.mark.parametrize("recipe,cfg,dataset,seed", _chain_params())
def test_jsonnet_parity_with_merge_yaml_chain(recipe, cfg, dataset, seed, tmp_path):
    # --- OLD path ---
    from graphids.orchestrate.resolve import ConfigResolver
    resolver = ConfigResolver(lake_root=str(tmp_path), user="parity")
    resolved = resolver.resolve(cfg, dataset=dataset, seed=seed, upstream_ckpts={})
    overrides = TrainingContract.to_override_dict(resolved.spec)
    old = merge_yaml_chain(cfg.config_files, overrides)

    # --- NEW path ---
    jsonnet_path = _JSONNET_STAGES[cfg.stage]
    tla = _build_tla(cfg, resolved.spec, dataset, seed)
    new = render_config(jsonnet_path, tla)

    # --- Compare ---
    _assert_dicts_str_equal(old, new, context=f"{recipe}/{cfg.asset_name}/{dataset}/s{seed}")


def _build_tla(cfg, spec, dataset, seed) -> dict:
    """Translate StageConfig + TrainingSpec into the TLA dict each stage.jsonnet expects."""
    return {
        "dataset": dataset,
        "seed": seed,
        "run_dir": spec.run_dir,
        "scale": cfg.scale,
        "conv_type": cfg.model_init_overrides.get("conv_type", "gatv2"),
        "variational": cfg.model_init_overrides.get("variational", "true") == "true",
        "loss_fn": cfg.model_init_overrides.get("loss_fn", "focal"),
        "fusion_method": cfg.model_init_overrides.get("fusion_method"),
        "auxiliaries": [cfg.kd_overrides] if cfg.kd_overrides else [],
        "trainer_overrides": cfg.trainer_overrides,
        "stage_overrides": cfg.stage_overrides,
        "vgae_ckpt_path": spec.upstream_ckpt_paths.get("vgae_ckpt"),
        "gat_ckpt_path": spec.upstream_ckpt_paths.get("gat_ckpt"),
    }


def _assert_dicts_str_equal(a, b, *, context: str, path: str = "") -> None:
    """Recursive deep-compare with stringification tolerance (§9.3)."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"[{context}] key mismatch at {path!r}: {set(a) ^ set(b)}"
        for k in a:
            _assert_dicts_str_equal(a[k], b[k], context=context, path=f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"[{context}] list length at {path!r}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_dicts_str_equal(x, y, context=context, path=f"{path}[{i}]")
    else:
        assert str(a) == str(b), f"[{context}] at {path!r}: old={a!r} new={b!r}"
```

**Why `@pytest.mark.slurm`:** `enumerate_assets` → `TrainingContract.resolve_config_files`
→ pulls in `graphids.config` which is lightweight, but `ConfigResolver.resolve`
does NOT need torch — the torch dep is only on `validate_cli_chain`. We skip
validate_cli_chain in the harness (we're comparing pre-parse dicts), so this
could run on a login node. BUT: step 2 of the exit criteria requires also
proving `parser.parse_object(render_config(...))` succeeds — and that DOES
import torch. Mark slurm. Run via `scripts/submit.sh tests -k jsonnet_parity`.

### 8.1 The second assertion (jsonargparse still accepts the dict)

After `_assert_dicts_str_equal(old, new)`, run:

```python
from graphids._lightning import schema_parser
schema_parser().parse_object(new)  # must not raise
```

This catches the case where the dicts deep-equal but jsonnet emits something
jsonargparse chokes on (e.g. real bool where the current YAML path has
string "true"). If this fires, the fix is usually "match the current YAML
type exactly". Do NOT try to be clever about types in jsonnet — mirror what
YAML produces.

---

## 9. Known gotchas + mitigations

### 9.1 `num_workers: null` — preserve literal null

`stages/autoencoder.yaml`, `stages/normal.yaml`, `stages/curriculum.yaml` all
have `data.init_args.num_workers: null` with a comment "auto-sized from GPU-first
sizing chain". In jsonnet: `num_workers: null` (jsonnet has a first-class
`null`). The feedback memory
[print_config null serialization](../.claude/projects/...feedback_print_config_null_serialization.md)
documents the exact trap — jsonargparse emits null, which then overrides
Python defaults at instantiation. Keep the null.

### 9.2 `auxiliaries: []` vs kd-overlay

`models/vgae/base.yaml` has `auxiliaries: []` (explicit empty list). The KD
overlay `models/vgae/kd.yaml` replaces it with a one-element list. When
runtime_overrides then set `model.init_args.auxiliaries` with a JSON blob,
jsonargparse parses the blob and that wins.

Jsonnet equivalent:
- `vgae.base` emits `auxiliaries: []`
- `vgae.kd` is applied via `+:` — but `+:` on a list with `+:` semantics in
  jsonnet is actually a gotcha: `{a: [1]} + {a: [2]}` gives `{a: [2]}` (LAST
  wins, not concat). So `vgae.base + vgae.kd` correctly replaces. Good.
- When `auxiliaries` TLA is non-empty, the stage adds
  `model+: { init_args+: { auxiliaries: auxiliaries } }` which REPLACES the
  previous list (because `+` on dict keys with non-dict values is
  last-wins). Good.

**Write a unit test** that exercises all three states: no kd, kd overlay only,
kd overlay + TLA override. Parity must hold in all three.

### 9.3 String-coercion tolerance in parity check

Current `to_override_dict` stringifies everything (see `ops.py:122`). `merge_yaml_chain`
then merges strings into a dict that already has real types from the underlying
YAML. Result: *mixed* types (e.g. `max_epochs: "50"` from override wins over
`max_epochs: 300` from defaults, dict ends up with a string). Jsonnet produces
real ints. The parity comparator normalizes via `str(a) == str(b)` at the leaf
— same trick `test_merge_parity.py:87` already uses. **Do not** switch to
`assert a == b` or "50" != 50 will fire on every override.

### 9.4 Fusion scales asymmetry

`fusion/scales/*.yaml` is NOT in the CLI chain (excluded in `ops.py:94`). Phase
1 jsonnet also excludes it. Do not port `fusion/scales/*.yaml` to jsonnet in
Phase 1 — the orchestrator reads them separately via some other path (check
`component.py` if this becomes relevant). Stays YAML.

### 9.5 Deep-merge of link target fields

`cli.LINK_TARGETS` wires `data.init_args.dataset → model.init_args.dataset`
INSIDE jsonargparse. The merged dict before parse has `data.init_args.dataset`
but NOT `model.init_args.dataset`. Current parity test treats this correctly
(it compares pre-parse dicts). Jsonnet should also NOT pre-emit the linked
fields — let jsonargparse do the linking. Phase 3 revisits this when we
strip LightningCLI.

### 9.6 `defaults/trainer.yaml` + dev-path `--config .jsonnet` preprocessor

Today `defaults/trainer.yaml` is NOT in `spec.config_files` — it's applied
silently by jsonargparse via `CLI_KWARGS.parser_kwargs.default_config_files`
before every `fit/test/validate/predict`. Current `merge_yaml_chain(
spec.config_files, ...)` therefore does NOT include it; it gets pulled in only
after `parse_object`.

**Phase 1 solution:** bake the trainer defaults into every jsonnet stage via
`import '../_lib/defaults.libsonnet'`, then DELETE `default_config_files` from
`CLI_KWARGS.parser_kwargs`. Single source, no silent injection. The
`graphids/config/defaults/trainer.yaml` file is deleted in Commit 5.

The parity harness during Commits 2–3.5 has a transient mismatch from this
gap — the jsonnet output is a SUPERSET of `merge_yaml_chain` output until
the harness prepends `defaults/trainer.yaml` to the old-path chain:

```python
# tests/config/test_jsonnet_parity.py (transient — deleted in Commit 5)
chain = (str(CONFIG_DIR / "defaults/trainer.yaml"),) + tuple(cfg.config_files)
old = merge_yaml_chain(chain, overrides)
```

**Dev-path preprocessor** for `python -m graphids fit --config X.jsonnet`:
LightningCLI's `--config` flag takes YAML paths, not jsonnet. Intercept in
`run_lightning` before handing off to `GraphIDSCLI`:

```python
# graphids/cli.py
def run_lightning(args: list[str]) -> None:
    from graphids._lightning import CLI_KWARGS, GraphIDSCLI
    GraphIDSCLI(**CLI_KWARGS, args=_preprocess_jsonnet_configs(args))


def _preprocess_jsonnet_configs(args: list[str]) -> list[str]:
    """If any `--config foo.jsonnet` appears, render it to a temp YAML and
    substitute the path. Power-user overrides (`--model.init_args.lr=0.01`)
    are untouched — they still reach jsonargparse in the normal way.
    """
    import tempfile
    from pathlib import Path
    from graphids.config.jsonnet import render_config
    from graphids.config.yaml_utils import write_yaml

    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--config" and i + 1 < len(args) and args[i + 1].endswith(".jsonnet"):
            rendered = render_config(args[i + 1], tla=None)
            tmp = Path(tempfile.mkstemp(suffix=".yaml", prefix="graphids_cfg_")[1])
            write_yaml(rendered, tmp)
            out += ["--config", str(tmp)]
            i += 2
            continue
        out.append(tok); i += 1
    return out
```

Jsonnet stage functions must therefore have sensible defaults for every TLA
(dataset="hcrl_ch", seed=42, scale="small", variational=true, conv_type="gatv2",
loss_fn="focal", auxiliaries=[], trainer_overrides={}, stage_overrides={},
vgae_ckpt_path=null, gat_ckpt_path=null, run_dir="experimentruns/dev/...") so
the dev-path invocation `python -m graphids fit --config configs/stages/autoencoder.jsonnet`
works with zero TLAs. Dev overrides still flow through LightningCLI's normal
`--model.init_args.*` mechanism *after* the jsonnet rendering.

Pipeline path does NOT use the preprocessor — `train_entrypoint` calls
`render_config(spec.jsonnet_path, spec.jsonnet_tla)` directly and passes the
dict to `build_cli` (same as today's `build_cli(merged)` call path).

### 9.7 stderr noise from SystemExit in existing validate_cli_chain

`resolve.py:360-380` catches `SystemExit` from jsonargparse. Phase 1 parity
harness does NOT call `validate_cli_chain` (§8.1 calls `parse_object` directly,
not via the resolver). If `parse_object` sys.exits on a parity mismatch, the
harness needs the same stderr-redirect dance. Copy the idiom from
`resolve.py:372-380` into a helper in the test file.

### 9.8 Jsonnet `+` vs `+:` foot-shooting

In jsonnet:
- `a: {x: 1} + {y: 2}` → `{x: 1, y: 2}` (shallow merge of TOP-LEVEL objects)
- `a+: {x: 1}` inside a parent object is DEEP-merge OF THE KEY `a`
- `{k: {x: 1}} + {k: {y: 2}}` → `{k: {y: 2}}` (REPLACES k, does not merge)
- `{k+: {x: 1}} + {k+: {y: 2}}` → `{k: {x: 1, y: 2}}` (deep-merges k)

**Always use `+:` on nested keys**. The stage files in §5.1 use `trainer+:` and
`model+: { init_args+: { ... } }` — follow that pattern religiously. A single
missing `:` silently replaces a nested dict instead of merging it, and the
parity test will catch it — but you'll waste time debugging it first. Spike
step (§7.1) exists specifically to debug this kind of surprise on a small
scope before scaling up.

### 9.9 TLA string defaults for nullable upstream ckpts

Jsonnet TLAs can take `null` via `--tla-code key=null`. The Python shim emits
`json.dumps(None)` → `"null"` → passed through `--tla-code`. Verify this round-
trips in the spike step. If it breaks, fall back to passing an empty string
sentinel and branching on `if ckpt == '' then ... else ...` — ugly but explicit.

---

## 10. Risks + open questions (flag before step 1)

1. **Recipe expansion stays Python (for now).** We are NOT porting
   `expand_recipe_configs` to jsonnet in Phase 1. Jsonnet recipes would need
   access to `VALID_SCALES`, `VALID_FUSION_METHODS`, etc., which live in
   `topology.py`. Phase 1 keeps the boundary at StageConfig. If jsonnet recipes
   look appealing later, revisit in Phase 2.
2. **`fusion/scales/*.yaml` excluded** from Phase 1 (§9.4). If a later audit
   shows they ARE in the training chain, everything in §7 step 4 needs revision.
3. **KD teacher resolution** runs inside `_resolve_kd_teachers` in
   `planning.py`. Phase 1 doesn't touch this — jsonnet just receives the
   resolved `auxiliaries` list as a TLA. Verify in the spike that the list
   round-trips through `--tla-code` correctly (especially the nested dict
   per-auxiliary).
4. **`scripts/submit.sh` preamble** must check for the jsonnet binary on SLURM
   nodes. Pitzer and Ascend share `$HOME`, so `~/.local/bin/jsonnet` is
   visible, but verify. Cardinal may differ — test on each cluster before
   merging.
5. **`jsonnetfmt --test` in CI** may flip on whitespace differences vs the
   hand-written jsonnet. Run it once over the whole tree and commit the
   formatted version so future diffs stay clean.
6. **Deep-merge of lists** — yaml_utils.deep_merge *replaces* lists
   (`out[k] = v` when `v` is non-dict). Jsonnet `+:` on list-valued keys also
   replaces when the parent is an object (§9.2). These match. If a test ever
   fails because a list got concat'd instead of replaced, the bug is in the
   jsonnet — probably stray `+:` on a list key. Correct: plain `k: [...]` at
   the inner level.

---

## 11. Verification checklist (before merging `phase1-jsonnet`)

### Tooling

- [ ] `jsonnet --version` returns 0.20.0+ on Pitzer login node
- [ ] `jsonnet --version` returns 0.20.0+ inside a gpu SLURM job on Pitzer
      (and Ascend, Cardinal if they host pipeline runs)
- [ ] `scripts/submit.sh` preamble hard-fails when jsonnet binary is missing
- [ ] `jsonnetfmt --test configs/` passes

### Jsonnet source tree exists

- [ ] `configs/stages/*.jsonnet` — 4 files (autoencoder, normal, curriculum, fusion)
- [ ] `configs/models/*.libsonnet` — 3 files (vgae, gat, dgi)
- [ ] `configs/fusion/base.libsonnet` + `fusion/methods/*.libsonnet` (4 methods)
- [ ] `configs/_lib/{defaults,helpers}.libsonnet`
- [ ] Every stage jsonnet has sensible defaults for every TLA (dev path works
      with zero TLAs)

### Old code is gone (not just new code added)

- [ ] `ls graphids/config/stages/` → no such directory
- [ ] `ls graphids/config/models/` → no such directory
- [ ] `ls graphids/config/fusion/` → no such directory
- [ ] `ls graphids/config/defaults/trainer.yaml` → no such file
- [ ] `grep -rn merge_yaml_chain graphids/ tests/` → empty
- [ ] `grep -rn deep_merge graphids/ tests/` → empty
- [ ] `grep -rn apply_dotted_overrides graphids/ tests/` → empty
- [ ] `grep -rn to_override_dict graphids/ tests/` → empty
- [ ] `grep -rn resolve_config_files graphids/ tests/` → empty
- [ ] `grep -rn "runtime_overrides\b" graphids/core/ graphids/orchestrate/` → empty
      (the field is gone; allowed to keep the name in recipe plumbing)
- [ ] `grep -rn "config_files\b" graphids/core/contracts/ graphids/orchestrate/ graphids/_lightning.py` → empty
- [ ] `tests/config/test_merge_parity.py` → deleted
- [ ] `tests/config/test_jsonnet_parity.py` → deleted (it was a one-shot gate)
- [ ] `graphids/_lightning.py::CLI_KWARGS.parser_kwargs.default_config_files` → deleted

### Runtime works end-to-end

- [ ] `python -m graphids.orchestrate validate --recipe graphids/config/recipes/ablation.yaml` passes
- [ ] `python -m graphids.orchestrate validate --recipe graphids/config/recipes/smoke_test.yaml` passes
- [ ] `python -m graphids.orchestrate validate --recipe graphids/config/recipes/final_eval.yaml` passes
- [ ] `python -m graphids fit --config configs/stages/autoencoder.jsonnet
      --data.init_args.dataset hcrl_ch --trainer.max_epochs 1` runs to
      completion on a gpudebug SLURM job (dev path smoke)
- [ ] `dg launch --assets '*' --partition 'hcrl_ch|42'` under `smoke_test.yaml`
      runs at least one asset per stage (autoencoder, normal, curriculum,
      fusion) to COMPLETED (pipeline path smoke)
- [ ] Resulting `run_record.json` sidecars land in the lake with
      `status: completed` and populated `metrics`
- [ ] `python -m graphids pipeline-status` shows the smoke run as green
- [ ] `python -m graphids rebuild-catalog` picks up the new run records
- [ ] `graphids/commands/profile.py` still runs (`scripts/submit.sh profile`)
- [ ] `tests/config/test_jsonnet_render.py` (small permanent guard) passes

### Documentation updated, not just appended

- [ ] ADR `docs/decisions/0010-jsonnet-binary.md` committed
- [ ] `docs/reference/3-chain.md` — `merge_yaml_chain` → `render_config` everywhere
- [ ] `docs/reference/config-architecture.md` — rewritten for the jsonnet tree
- [ ] `.claude/rules/config-system.md` — rewritten (YAML chain description gone)
- [ ] `CLAUDE.md` (project root) — `--config` examples updated to `.jsonnet`
- [ ] `PLAN.md` — Phase 1 marked complete, Phase 2 handoff noted
- [ ] `graphids/config/{README.md,VALIDATION_CHECKLIST.md,CONFIG_REFERENCE.md}` —
      either rewritten to reflect jsonnet or deleted (don't leave stale YAML docs)

When every box is ticked, Phase 1 is done. Phase 2 (Pydantic validation layer)
builds on top of the rendered dicts.

---

## 12. LOC budget (sanity check, not a contract)

Full migration — expect roughly balanced adds and deletes.

| Area | Adds | Deletes | Notes |
|---|---|---|---|
| `configs/` jsonnet tree | 400–550 | — | ~1:1 with the YAML it replaces |
| `graphids/config/jsonnet.py` | 60–80 | — | Subprocess shim |
| `graphids/config/{stages,models,fusion}/**.yaml` | — | 300–400 | 20 YAML files |
| `graphids/config/defaults/trainer.yaml` | — | 17 | Inlined into jsonnet |
| `graphids/config/yaml_utils.py` | — | 40–50 | `merge_yaml_chain`, `deep_merge`, `apply_dotted_overrides` |
| `graphids/core/contracts/{models,ops}.py` | 30–50 | 60–80 | `build_tla_dict`, `resolve_jsonnet_path` in; `to_override_dict`, `resolve_config_files`, `_CKPT_FLAG_BY_MODEL` out |
| `graphids/orchestrate/{resolve,planning,validate}.py` | 20–40 | 20–40 | Field renames, `merge_yaml_chain` → `render_config` |
| `graphids/core/train_entrypoint.py` | 5–10 | 5–10 | One call site |
| `graphids/_lightning.py` | 5 | 10–15 | Bootstrap + default_config_files |
| `graphids/cli.py` | 30 | — | `_preprocess_jsonnet_configs` |
| `graphids/commands/profile.py` | 5 | 5 | Field names |
| `tests/config/test_jsonnet_parity.py` | 150–200 | 150–200 | Created in Commit 2, deleted in Commit 5 |
| `tests/config/test_merge_parity.py` | — | 90 | Deleted |
| `tests/config/test_yaml_utils.py` | — | 30–50 | Trim merge tests |
| `tests/config/test_jsonnet_render.py` | 30 | — | Permanent guard |
| `docs/decisions/0010-jsonnet-binary.md` | 40 | — | ADR |
| `docs/reference/{3-chain,config-architecture}.md` | 50 | 150 | Rewrites |
| `graphids/config/{README,VALIDATION_CHECKLIST,CONFIG_REFERENCE}.md` | — | 100–200 | Delete or trim |
| **Total** | **~825–1090** | **~975–1360** | Net negative: ~−200 to −300 LOC |

**Phase 1 is net-negative.** The jsonnet source tree is roughly 1:1 with the
YAML it replaces, but the deleted Python plumbing (merge/override/stringify/
coerce) is not counterbalanced — jsonnet absorbs those responsibilities
natively.

If the net comes out positive, something is wrong. Most likely culprits:
leaving the parity harness in, leaving the YAML files in, or keeping both
`config_files` and `jsonnet_path` on `TrainingSpec`. Any of those means Phase 1
is not actually done — it's a shadow path. Go back and re-read §1.
