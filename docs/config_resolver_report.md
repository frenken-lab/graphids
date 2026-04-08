# config-resolver Library Assessment

Evaluated: `config-resolver` v5.1.0 (PyPI) by Michel Albert (exhuma).
Source: [GitHub](https://github.com/exhuma/config_resolver) |
[Docs](https://config-resolver.readthedocs.io/en/latest/) |
MIT license | Last release: 2021-07-13 | 4 stars, 1 direct dependency

## What It Is

A pure-Python library for **locating and loading config files** from standard
filesystem paths, following the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html).
It answers one question: "given an app name and group name, where on disk
should I look for config files, and in what order?"

It is **not** a config schema validator, a config composition/merge engine,
or a template language. It finds files and hands them to a format handler.

## Core Concepts

- **ConfigID**: a `(group, app)` pair identifying the application (e.g., `("acmecorp", "bird_feeder")`).
- **Handler**: pluggable file-format reader. Ships with `IniHandler` (ConfigParser) and `JsonHandler` (dict). Custom handlers subclass `Handler[T]` and implement `empty()`, `from_string()`, `from_filename()`, `get_version()`, `update_from_file()`.
- **LookupResult**: `NamedTuple` of `(config, meta)` where `meta` has `active_path`, `loaded_files`, `config_id`, `prefix_filter`.

## Resolution Algorithm

`get_config(app_name, group_name, lookup_options, handler)` builds a search
path in ascending priority order:

1. `/etc/<group>/<app>/app.ini`
2. `/etc/xdg/<group>/<app>/app.ini` (XDG system dirs)
3. `~/.config/<group>/<app>/app.ini` (XDG user dir)
4. `./.{group}/{app}/app.ini` (CWD-local)

Files loaded later override values from earlier files (merge semantics
depend on handler; INI does section-level update, JSON does dict update).
No file is required by default — returns empty config if nothing found.

### Override mechanisms

| Mechanism | Effect |
|-----------|--------|
| `lookup_options["search_path"]` | Replace default search path entirely |
| `lookup_options["filename"]` | Change basename (e.g., `db.ini` instead of `app.ini`) |
| Env var `{GROUP}_{APP}_PATH` | Override/append (`+` prefix) search path |
| Env var `{GROUP}_{APP}_FILENAME` | Override config file basename |
| `XDG_CONFIG_HOME` / `XDG_CONFIG_DIRS` | Standard XDG overrides |
| `lookup_options["require_load"] = True` | Raise `OSError` if no file found |
| `lookup_options["secure"] = True` | Reject world-readable files |

### Versioning

Config files can declare `[meta] version=2.1` (INI) or `{"meta": {"version": "2.1"}}` (JSON).
Major version mismatch skips the file with an ERROR log. Minor version
of file must be >= expected minor. Useful for schema evolution signaling.

## Features Summary

- **File discovery**: XDG-compliant multi-tier path search
- **Format handlers**: INI (default), JSON, extensible to YAML/TOML via subclass
- **Config versioning**: semver-based file acceptance/rejection
- **Security check**: optional rejection of world-readable files
- **Environment variable overrides**: per-app path and filename control
- **Logging**: all file load/skip decisions logged via stdlib `logging`
- **Metadata**: callers can inspect which files were loaded and in what order

## What It Does NOT Do

- No config **composition** (no merging multiple config layers into one typed object)
- No **templating** or **variable interpolation** beyond ConfigParser's built-in `%(key)s`
- No **schema validation** — returns raw ConfigParser or dict
- No **deep merge** — JSON handler does shallow `dict.update()`
- No **typed coercion** — everything is strings (INI) or JSON primitives
- No **conditional logic** — no `if env == "prod"` branching
- No **inheritance** between config fragments

## Comparison with GraphIDS Config Stack

| Capability | config-resolver | GraphIDS (Jsonnet + Pydantic + ConfigResolver) |
|------------|----------------|-----------------------------------------------|
| File location | XDG path search across /etc, ~/.config, CWD | Explicit `configs/stages/*.jsonnet` paths, resolved by `topology.py` |
| Composition | Sequential file override (shallow) | Jsonnet deep-merge (`+:`) with libsonnet imports, recipe expansion |
| Templating | None | Jsonnet functions, TLA parameters, `apply_dotted()` |
| Schema validation | Version check only | Pydantic `ValidatedConfig` + cross-field rules |
| Typed output | Raw dict or ConfigParser | `InstantiatedRun(trainer, model, datamodule)` — fully wired objects |
| Merge semantics | Handler-dependent shallow update | Jsonnet `+:` deep merge with explicit precedence |
| Environment overrides | `{GROUP}_{APP}_PATH` env vars | `KD_GAT_*` env vars in `config/constants.py` + `slurm/env.py` |
| Audit trail | `meta.loaded_files` list | `OverrideRecord` tuples with source attribution |
| Config groups | `(group, app)` → one file per concept | Stage jsonnet files + model/fusion libsonnets per family |
| Format support | INI, JSON, custom handlers | Jsonnet (renders to JSON/dict) |

## Assessment: Not Useful for This Project

**config-resolver solves a different problem.** It is a **file finder** for
applications that need to locate config files across XDG-standard directories
(system → user → CWD). It was designed for traditional deployed applications
(daemons, CLI tools) where config files live in `/etc/` or `~/.config/`.

GraphIDS does not have this problem:
1. **Config files are in-repo.** All Jsonnet sources live under `configs/` in
   the repository. There is no XDG search — `topology.py` knows exactly where
   every stage config lives, validated at import time.
2. **Composition is the hard part.** The project needs deep-merge of model
   scales, KD auxiliaries, recipe overrides, trainer overrides, and upstream
   checkpoint paths. Jsonnet's `+:` operator and TLA functions handle this.
   config-resolver's shallow `dict.update()` cannot.
3. **Schema validation is critical.** Pydantic `ValidatedConfig` catches null
   list fields, monitor mismatches, un-namespaced class_paths, and
   logger/callback wiring errors before any torch import. config-resolver has
   no schema validation beyond a version number check.
4. **Typed instantiation is the goal.** The pipeline needs `(trainer, model,
   datamodule)` triples, not raw dicts. `graphids.instantiate` does
   class_path import, signature-filtered link_arguments, forced callbacks.
   config-resolver returns a ConfigParser or dict.
5. **No deployment diversity.** The code runs on one cluster (OSC Pitzer) with
   one Python environment. There are no system-wide vs. user-level vs.
   per-instance config tiers to search through.

**Verdict: Do not adopt.** The library adds a dependency (albeit small) for a
capability the project does not need. The existing Jsonnet + Pydantic +
custom ConfigResolver stack handles composition, validation, and
instantiation — none of which config-resolver addresses. The naming
collision with `graphids.orchestrate.resolve.resolver.ConfigResolver` is
coincidental; the two classes solve fundamentally different problems.
