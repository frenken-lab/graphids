# Orchestrate: Stage

Single-stage primitives: ``build`` resets GPU state and instantiates;
``train`` fits and writes the ``.train_complete`` marker + split
predictions; ``evaluate`` runs ``.test()``, writes the
``.test_complete`` marker + per-test-set prediction sidecars, and
persists the test-phase MLflow run row. ``_check_ckpt_compat`` guards
resume/test against silent topology drift (wrong Module class, wrong
``IdEncoder``).

## `graphids.orchestrate.stage`

::: graphids.orchestrate.stage
