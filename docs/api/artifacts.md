# Run Records

`graphids.exp.records` owns saved configuration, manifests, lifecycle events and
status summaries. New runs save final metrics and failures in schema-2 manifests;
`resolved-run.yml` contains their validated input. Historical schema-1 manifests
remain readable.

`graphids.exp.tracking` derives MLflow exports from current records and reads
historical `mlflow_ingest.json` files. See [observability](../reference/observability.md)
and [write paths](../reference/write-paths.md).

::: graphids.exp.records
