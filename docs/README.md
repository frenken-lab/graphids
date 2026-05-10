# Documentation Lifecycle

> Rules for maintaining `docs/`. Site entry point is [`index.md`](index.md);
> module map is [`responsibilities.md`](responsibilities.md).

## Structure

```
docs/
  decisions/    # ADRs — final verdicts, permanent
  reference/    # Living docs for current architecture
  drafts/       # Work-in-progress (not site-rendered)
  api/          # Auto-generated from docstrings
```

Work items are tracked as **GitHub issues** (`gh issue list`), not files.

## Lifecycle Rules

### Decision Records (`decisions/`)
- **Created when:** a tool/approach is evaluated and a final verdict is reached
- **Format:** title, context, decision, rationale, consequences, sources
- **Naming:** `NNNN-short-title.md` (sequential)
- **Never deleted.** If reversed, add "Superseded by NNNN" header
- **Max ~50 lines.** Decision + rationale, not full research

### Reference Docs (`reference/`)
- **Created when:** a subsystem is complex enough to need a map
- **Must describe current state, not historical evolution.** Stale = bug
- **If you change the code, update the doc in the same session**
- **Named by topic:** `cli-routes.md`, `write-paths.md`
- **Deleted when** the subsystem is removed
- Current maps worth reading first:
  - [`reference/data-architecture.md`](reference/data-architecture.md)
  - [`reference/config-architecture.md`](reference/config-architecture.md)

### Work Items (GitHub Issues)
- **Created when:** work is deferred or a design question is identified
- **Closed when resolved.** The fix is in git
- **Labels:** `config`, `orchestration`, `performance`, `models`, `evaluation`, `blocked`, `bug`
- When an issue becomes a decision, the decision goes to `decisions/` and the issue is closed

## What Goes Where

| Situation | Action |
|-----------|--------|
| Evaluated a tool/approach | `decisions/NNNN-*.md` |
| Describing how system X works today | `reference/*.md` |
| Work to do later | GitHub issue with appropriate labels |
| Bug found during a run | Fix it or file a GitHub issue |
| Plan for a session | `PLAN.md` (root), not docs/ |
| Implementation was completed | Close the issue. Update reference if architecture changed |

## Anti-Patterns

- **Don't mark files "RESOLVED" and keep them.** Delete them
- **Don't write implementation plans that outlive implementation.** They become stale instantly
- **Don't duplicate what's in code.** Config structure, file layouts, CLI commands — read the code
- **Don't accumulate session logs in docs/.** That's `PLAN.md`'s job
