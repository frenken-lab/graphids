# GraphIDS Session Plan

> Last updated: 2026-04-13 (session 45 — docs consolidation, fusion analysis, probe-plot rewrite)

PLAN.md is current-session work only. Historical session changelogs live in
git log; durable verdicts live in `docs/decisions/README.md`; living architecture
lives in `docs/reference/`.

## Active

- **probe-budget job 46752442** — queued on Pitzer (V100s saturated, est. start
  ~17h worst case but typically sooner). Sidecar will land at
  `experimentruns/slurm/budget_probe_46752442.jsonl`. Plot with
  `python -m graphids.plots.budget --jsonl <path> --gpu V100_16GB`.
- **#18 validate GPU-first auto-sizing** — blocked on probe-budget output.
- **`probe-budget --dry-run` UX** — currently exits at the GPU check before
  printing the plan (`_slurm.py:54-56`). 3-line reorder needed.

## Recently landed (last session)

- Campaign manifest subsystem (`graphids/campaigns/` + `cli/_campaign.py` +
  `campaigns/<name>.yaml`); recipes machinery deleted (-1421 LoC).
- probe-budget JSONL sidecar; `plots/` rewritten to direct-measurement plots
  only — dropped `ModelParams`/`fit_models`/regime classification.
- Fusion analysis enabled (#24): self-describing checkpoints
  (`class_path` written by `_build_checkpoint`, read by `safe_load_checkpoint`),
  upstream paths threaded into `analysis_spec_for`,
  `ANALYZABLE_MODEL_TYPES += "fusion"`.
- Closed: #24 (fusion analysis), #26 (per-test-set metrics — already shipped
  in 771376c), #28 (DGI gamma anomaly — already fixed in 0a4f6e1), #29 (open
  items audit).

## Open issues

- **#18** Validate GPU-first auto-sizing on SLURM (blocked on probe job)
- **#32** Add WaDi dataset module
- **#33** spam, can be closed/blocked

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Issues: `gh issue list`
