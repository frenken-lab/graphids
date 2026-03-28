"""Pipeline orchestrator: submit LightningCLI stages to SLURM with DAG ordering.

Usage as CPU job (interactive, can intervene):
    srun --partition=cpu --time=24:00:00 --mem=4G --account=PAS1266 --pty \
      python -m graphids.orchestrate configs/stages/ --datasets set_01 set_02

Usage from login node (fire and check later):
    python -m graphids.orchestrate configs/stages/ --datasets set_01 --dry-run
"""
