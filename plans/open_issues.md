## Config issues

- There is terrible configuration setting up, tracking, and overriding

- many different configurations with no defined responsibilities
- complicated merge mechanics in CLI
- dagster or user overrides or ignores on ad-hoc basis, no policy

## Experiment Tracking

- no assurance that everything that needs to be written (slurm logs, metrics, yaml, model artifacts)
  is actually being logged.
- Training jobs ran a whole pipeline and didnt bother to actually save checkpoint or final model weights
- poor communication between lighting, wandb, and dagster
- zero awareness, everything is a black box

## Half wired up metrics

- some wandb logs populate
- dagster UI broken
- No defined read and write roles
- database in share file is ad-hoc, buried code everywhere
