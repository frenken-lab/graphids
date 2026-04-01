# Validation Checklist

1. Validate every YAML file against the nearest schema.
2. Expand matrix combinations and verify each resolves without missing files.
3. Assert every resolved run has a matching resource profile.
4. Assert fusion runs require `fusion_method`.
5. Assert no dead keys in resolved configs.
6. Assert `curriculum.data.init_args.max_epochs_ref == trainer.max_epochs`.
7. Materialize and save `resolved/config.yaml` and `resolved/provenance.yaml`.
