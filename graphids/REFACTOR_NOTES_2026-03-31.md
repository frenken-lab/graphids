# Refactor Notes (2026-03-31)

## Scope

This refactor focused on reducing orchestration sprawl, formalizing execution contracts,
reorganizing model/preprocessing modules by family, and replacing custom orchestration
patterns with more Dagster-native behavior.

## What Changed

1. Contract boundary extraction
- Added canonical training and analysis contracts under `core/contracts/`.
- Introduced `TrainingSpec` and `AnalysisSpec` envelopes for transport-safe execution.
- Added `TrainingContract` / `AnalysisContract` helpers for serialization and CLI bridging.

2. Programmatic train/analyze entrypoints
- Added `core/train_entrypoint.py` and `core/analyze_entrypoint.py`.
- Added `commands/train_from_spec.py` and `commands/analyze_from_spec.py` to execute from serialized spec files.
- Added shared payload loader in `commands/_spec_payload.py`.

3. Orchestrate decomposition + Dagster-native replacement
- Split orchestration responsibilities into:
	- `orchestrate/planning.py` (asset planning)
	- `orchestrate/execution.py` (path/spec/accounting helpers)
	- `orchestrate/assets.py` (asset factories)
	- `orchestrate/checks.py` (asset checks)
	- `orchestrate/analysis.py` (analysis support/helpers)
- Replaced custom checkpoint sidecar IO manager with Dagster `fs_io_manager`.
- Removed explicit `.complete` marker requirement from execution/check flow.
- Added asset-level retry policy in Dagster assets.

4. SLURM boundary hardening
- Added protocol boundary `SlurmJobClient` in `orchestrate/slurm.py`.
- Added default adapter `SubprocessSlurmJobClient`.
- Switched SLURM command execution to spec-file transport (`train-from-spec`) rather than direct config command assembly.

5. Config system modularization
- Replaced monolithic `config/__init__.py` internals with modular config API:
	- `config/base.py`, `config/runtime.py`, `config/topology.py`, `config/paths.py`, `config/contracts.py`.
- Added strict YAML loader helper `config/yaml_utils.py` and reused it in resource loading.
- Added recipe expansion module `config/recipe_expand.py`.

6. Recipe/KD ergonomics
- Added first-class KD sweep declaration support in recipe expansion (`kd` block emits `auxiliaries`).
- Added regression test coverage for KD sweep expansion.

7. Model and preprocessing package reorganization
- Moved model implementations into family namespaces:
	- `core/models/autoencoder/`
	- `core/models/supervised/`
	- `core/models/fusion/`
	- `core/models/temporal_family/`
- Moved preprocessing stage-specific modules into `core/preprocessing/stages/`.
- Updated import paths and registry/module mapping accordingly.

8. CLI routing cleanup
- Replaced dynamic command import fallback in `__main__.py` with explicit subcommand registry and parser routing.
- Added smoke test coverage for CLI routing behavior.

9. Resource config split
- Updated resource loading to use compact tree under `config/resources/`.
- Added submit profiles in `config/resources/submit_profiles.yaml`.

## Foreseen Risks

1. Runtime compatibility risk (high)
- Physical module moves can break external scripts/checkpoints that import old module paths directly.
- Risk surface: user scripts, notebooks, stale serialized references.

2. Dagster data continuity risk (medium)
- Switching to `fs_io_manager` and removing custom sidecar patterns may alter how existing local IO artifacts are discovered.
- Risk surface: existing local dev runs with previous IO conventions.

3. Recipe expansion semantics risk (medium)
- KD sweep expansion introduces new normalization behavior that may differ from legacy handwritten recipe configurations.
- Risk surface: recipe configs that rely on implicit defaults or legacy shape assumptions.

4. SLURM transport operational risk (medium)
- Spec-file lifecycle and shared filesystem assumptions are now critical for job start.
- Risk surface: permissions/cleanup under the configured SLURM log/spec path.

5. Config validation strictness risk (low-medium)
- Centralized strict YAML loading and modular topology checks can fail earlier than before.
- Risk surface: malformed or non-mapping YAML files that previously passed silently.

## Mitigations in Place

1. Added targeted tests
- `tests/test_cli_routing_smoke.py`
- `tests/test_recipe_expand_kd.py`

2. Contract-driven execution
- Both train/analyze transport now validate contract identity/version through envelope parsing.

3. Centralized planning/execution helpers
- Fewer ad-hoc code paths in orchestrate reduce divergent behavior across assets.

## Validation Status

1. Static status
- Workspace diagnostics were clean at the end of refactor iterations.

2. Runtime status
- Full runtime execution verification (Dagster materialization + test execution) still needs to be run in an environment with Python runtime/deps available.

## Recommended Post-Refactor Checks

1. Dagster validation
- `python -m graphids.orchestrate validate-dagster`

2. Recipe + Lightning parse validation
- `python -m graphids.orchestrate validate`

3. Smoke tests
- `python -m pytest tests/test_cli_routing_smoke.py tests/test_recipe_expand_kd.py`

4. One representative end-to-end materialization
- Materialize one dataset/seed partition for a non-fusion stage and one fusion stage.

