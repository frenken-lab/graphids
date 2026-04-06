# Config Reorg: jsonnet + Pydantic + jsonargparse

## What Moves Where

| Current | Destination | Why |
|---|---|---|
| `orchestrate/planning.py::StageConfig` | `graphids/config/shared.py` | schema not orchestration |
| `slurm/__init__.py::ResourceSpec` | `graphids/config/shared.py` | schema not slurm logic |
| `orchestrate/resolve.py` cross-field checks | `@model_validator` on per-stage configs | Pydantic's job |
| `orchestrate/validate.py` | deleted | absorbed by Pydantic |
| `configs/stages/*.yaml` | `configs/stages/*.jsonnet` | composition + self-refs |
| `configs/models/*.yaml` | `configs/models/*.libsonnet` | reusable, not rendered |
| `configs/resources/*.yaml` | `configs/resources/job_profiles.json` | pure static lookup |

---

## Target Layout

```
configs/
  _lib/                        # keep — already started
    base.libsonnet
    slurm.libsonnet            # absorbs ResourceSpec defaults
    training.libsonnet         # val_loss/min vs val_acc/max per stage
    utils.libsonnet
  datasets/
    dataset_registry.json      # keep — already done
  stages/                      # migrate YAMLs → jsonnet
    autoencoder.jsonnet
    normal.jsonnet
    curriculum.jsonnet
    fusion.jsonnet
  models/                      # migrate YAMLs → libsonnet
    gat.libsonnet
    dgi.libsonnet
    vgae.libsonnet
    fusion/
      dqn.libsonnet
      bandit.libsonnet
  resources/
    job_profiles.json          # GAT/DGI/etc walltime+CPU — pure JSON
  envs/
    cluster.libsonnet          # scratch paths, partition names

graphids/config/               # Pydantic schemas + shared contracts
  __init__.py                  # re-exports key configs
  shared.py                    # ResourceSpec, StageConfig
  cross_field.py               # cross-field rule table
  schemas.py                   # Pydantic models + validation helpers
  jsonnet.py                   # render_config() — keep as-is
```

---

## Pydantic Cross-Field Gate

```python
# graphids/config/schemas.py (StageValidation)
from pydantic import BaseModel, ConfigDict, model_validator
from graphids.config.cross_field import _RULES

class StageValidation(BaseModel):
    spec: TrainingSpec
    resources: ResourceSpec
    cfg: StageConfig
    merged: dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_cross_fields(self):
        # Applies _RULES (workers ≤ cpus-1, curriculum epoch sync,
        # GPU partition checks, RL dead config warnings).
        ...
        return self
```

---

## jsonargparse Usage

`jsonargparse` is retained for analyzer configs. `graphids.commands.analyze`
uses `ArgumentParser(parser_mode="jsonnet")` so analyzer configs can be
Jsonnet while CLI overrides still work.

```bash
python -m graphids analyze --config configs/stages/analyze_vgae.jsonnet \
  --analyzer.ckpt_path /path/to/best.ckpt --analyzer.dataset hcrl_sa
```

---

## Stage jsonnet Pattern

See `configs/stages/{autoencoder,normal,curriculum,fusion}.jsonnet` for the
authoritative pattern. These stages compose `configs/_lib/defaults.libsonnet`
and model/fusion libsonnets with TLAs for dataset/seed/scale and override
hooks (`trainer_overrides`, `stage_overrides`).

---

## Deletion List

| File | Reason |
|---|---|
| `orchestrate/validate.py` | Pydantic `@model_validator` replaces it |
| `orchestrate/resolve.py` cross-field logic | moved to `config/schemas.py` |
| `configs/stages/*.yaml` | replaced by `.jsonnet` |
| `configs/models/*.yaml` | replaced by `.libsonnet` |
| `configs/resources/*.yaml` | replaced by `job_profiles.json` |
