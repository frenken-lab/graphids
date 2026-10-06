# Integrated signal catalog smoke run

The [catalog](toyota_prius-source_only-signal-catalog-v1.json) is a source-only Toyota Prius search artifact from the historical four-vehicle development manifest. It used two Prius fitting routes, source-vehicle DBCs, and `--max-ids 4`. The target DBC and final route were not read. It is a smoke run, not a full-cohort accuracy result.

The artifact records 110 observed message families with per-route period and bit-activity summaries, 36 source-role behavior prototypes, and 69 role or label profiles. `speed` has five ranked ID hypotheses, led by `0xB4`, but no bit-span hypothesis in this capped run. `acceleration` has 37 structural field candidates across five weakly ranked families. These behavior-prior rankings do not verify physical names. No cross-vehicle frame match was accepted and no new label was promoted.

Reproduce with:

```bash
.venv/bin/python -m scripts.benchmark_can_integrated \
  /tmp/graphids_candidate_graph_v11_repro/benchmark-manifest.json \
  --output-dir /tmp/graphids_catalog_smoke \
  --track source_only --only-vehicle TOYOTA_PRIUS --max-ids 4
```
