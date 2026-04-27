# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

### Render snapshots verified identical

Rendered four representative presets pre/post each stage and diff'd —
zero output drift across all 3 consolidations. 133 tests pass.

### Submit chain: render once, pickle the rendered dict

`_TrainingJob` now carries the rendered config dict instead of
`(config_path, tlas, sets)`. `submit()` renders ONCE on the login node
with both TLAs **and** `--set` overrides applied (was previously a
latent bug — login render dropped `--set`, so login-computed `run_dir`
disagreed with compute-computed `run_dir` for any `--set` touching it).
Compute node skips jsonnet eval entirely via `training.run_rendered`,
which is the new shared compute-node entrypoint for both the CLI and
submitit paths. `cli/submit.py` collapsed into a Typer-decorated forward.
End-to-end submitit fit smoke not yet run — only verified via dry-run +
pickle roundtrip + ResolvedConfig consumption.

### Submit: one entrypoint, render helper centralized

`submit_with_flags` merged into `submit()` — one function takes the
full CLI flag surface (preset/dataset/seed/depends_on/skip_if_finished/...)
and dispatches to submitit. The two-function indirection was a leftover
from when `dag.py` also called the low-level `submit()`; current
architecture has only one caller (the Typer wrapper). `dotted_to_nested`
moved from `cli/app.py` to `config/jsonnet.py` and gained a
`render_with_flags(preset, tla, set_)` companion — both training.py and
slurm/submit.py now share one render-input transformer instead of
inlining `render(..., dict(tla or []), dotted_to_nested(set_))`. Removes
the awkward `slurm → cli` import. 60/60 affected tests pass on SLURM
(jid 47107452).

## Open issues

- **#32** Add WaDi dataset module.
- **Verify submitit fit end-to-end**: a real `--smoke` fit on Pitzer to
  confirm `_TrainingJob.__call__ → run_rendered → train` works on a GPU
  compute node with the new pickled-rendered-dict payload.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
