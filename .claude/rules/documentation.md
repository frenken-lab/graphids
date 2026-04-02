# Documentation Lifecycle

## Structure

```
docs/
  decisions/    # ADRs — final verdicts, permanent
  reference/    # Living docs for current architecture
  backlog/      # Open work items, deleted when done
```

## Lifecycle Rules

### Decision Records (`decisions/`)
- **Created when:** a tool/approach is evaluated and a final verdict is reached
- **Format:** title, context, decision, rationale, consequences, sources
- **Naming:** `NNNN-short-title.md` (sequential). Next available: 0009
- **Never deleted.** If reversed, add "Superseded by NNNN" header
- **Max ~50 lines.** Decision + rationale, not full research

### Reference Docs (`reference/`)
- **Created when:** a subsystem is complex enough to need a map
- **Must describe current state, not historical evolution.** Stale = bug
- **If you change the code, update the doc in the same session**
- **Named by topic:** `cli-routes.md`, `write-paths.md`
- **Deleted when** the subsystem is removed

### Backlog Items (`backlog/`)
- **Created when:** work is deferred or a design question is identified
- **Deleted entirely when resolved.** Not compacted, not marked "RESOLVED" — gone. The fix is in git
- **Named by topic:** `kd-untested.md`, `per-stage-overrides.md`
- When a backlog item becomes a decision, the decision goes to `decisions/` and the backlog item is deleted

## Session Hygiene

At session start, scan `docs/backlog/` for items that are done. Delete them.
This takes 2 minutes and prevents multi-hour audits.

## What Goes Where

| Situation | Action |
|-----------|--------|
| Evaluated a tool/approach | `decisions/NNNN-*.md` |
| Describing how system X works today | `reference/*.md` |
| Work to do later | `backlog/*.md` |
| Bug found during a run | Fix it or add to `backlog/`, never leave as "RESOLVED" |
| Plan for a session | `PLAN.md` (root), not docs/ |
| Implementation was completed | Delete the backlog item. Update reference if architecture changed |

## Anti-Patterns

- **Don't mark files "RESOLVED" and keep them.** Delete them
- **Don't write implementation plans that outlive implementation.** They become stale instantly
- **Don't duplicate what's in code.** Config structure, file layouts, CLI commands — read the code
- **Don't accumulate session logs in docs/.** That's `PLAN.md`'s job
