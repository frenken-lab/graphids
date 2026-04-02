# KD-GAT Compact Config System

This is a composition-first YAML config system designed to be compact, robust, and extensible.

## Composition order (lowest to highest precedence)

1. `defaults/global.yaml`
2. `defaults/trainer.yaml`
3. `defaults/io.yaml`
4. `datasets/{dataset}.yaml`
5. `stages/{stage}.yaml`
6. `models/{model_family}/base.yaml`
7. `models/{model_family}/scales/{scale}.yaml`
8. If `stage=fusion`:
   - `fusion/base.yaml`
   - `fusion/scales/{scale}.yaml`
   - `fusion/methods/{fusion_method}.yaml`
9. `resources/profiles/{family}.yaml` selected by cluster + stage + scale (+ method for fusion)
10. `recipes/{recipe}.yaml` overrides
11. CLI / environment overrides

## Design choices

- Axis-first: one file per concern, minimal duplication.
- Bandit is a first-class peer fusion method under `fusion/methods/bandit.yaml`.
- Run legality enforced by `topology.py` import-time assertions.

## Suggested resolver outputs

A resolver should materialize a single effective config per run and write:
- `resolved/config.yaml`
- `resolved/provenance.yaml` (which files and override order)

## Environment override convention

Use `KD_GAT__` prefix and double underscore path separators, for example:
- `KD_GAT__MODEL__INIT_ARGS__LR=0.001`
- `KD_GAT__TRAINER__MAX_EPOCHS=200`
