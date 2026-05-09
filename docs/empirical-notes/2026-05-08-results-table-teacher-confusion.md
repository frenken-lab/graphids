# 2026-05-08 — Results table: three wrong moves on teacher model placement

**Context:** Building the interactive results-table Svelte figure for the paper.
Repeated failure to correctly model what "teacher" means in the KD pipeline.

---

## The mistake (three rounds)

**Round 1 — teacher in `MAIN_GROUPS`:**  
App.svelte was written with `MAIN_GROUPS = new Set(['fusion', 'teacher', 'student', 'student_kd'])`.
Teacher treated as just another deployment variant competing with fusion and student.
User correction: *"teacher is a scale oh my god you are all fucking crossed up."*

**Round 2 — teacher removed from export entirely:**  
Interpreted "teacher is a scale" as "exclude teachers." Removed teacher rows from
`_fetch_results_table_runs()` and deleted `teacher_vgae`/`teacher_gat` from `_VARIANT_LABELS`.
User correction: *"thats not what i meant."*

**Round 3 — asked for clarification, got the right answer:**  
User confirmed: reference row at the top, visually distinct from fusion/student comparison rows.
Then discovered teacher runs have no test phase in MLflow — only fit — so there's nothing
to show yet regardless of UI design.

---

## What teacher actually is

`teacher_gat` and `teacher_vgae` are the **full-size models you distill FROM**.
They define the performance ceiling the student is compressed toward.
- **Not a deployment variant** — teachers are training scaffolding, not shipped models.
- **Not a group to filter** — they are the scale/reference against which compression
  efficiency is measured (e.g. student = 1/20th the parameters, retains X% of teacher F1).
- **Belongs at the top of the main table as a non-competing reference row** — visually
  distinct (e.g. dashed border, "Reference" label), not sortable against fusion/student rows.

---

## Current state

- Teacher fit runs: FINISHED for all 5 datasets, checkpoints verified.
- Teacher test runs: **none submitted** — no test-phase MLflow entries exist.
- Until teacher test jobs run, the reference row has no data to show.
- Next step: render `training.main --filter '*teacher_gat*test*'` and submit for all 5 datasets.

---

## What to encode

When building any results table or figure that involves the KD pipeline:
1. Teachers are the scale, not a competitor.
2. Reference row treatment: fixed position (top), visually distinct, not in the sort/rank axis.
3. Check MLflow for `phase=test` runs before assuming data exists — fit ≠ test.
