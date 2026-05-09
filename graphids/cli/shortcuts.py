"""Read-only shell shortcuts wired as Typer commands (Pattern 2).

Each entry in :data:`RECIPES` becomes a top-level CLI command via dynamic
registration on the root Typer ``app``. The body is a literal shell string
executed via ``subprocess.run(..., shell=True)``. Use exclusively for
**read-only or single-action shell wrappers** that would otherwise be
``alias`` lines in your shellrc.

What does NOT belong here:

- Anything that spawns multiple SLURM jobs in a loop (would violate
  ``chassis-invariants.md``). Stay a Justfile / bash function.
- Anything with user-supplied arguments interpolated into the shell string
  (injection risk on ``shell=True``). If you need params, write a real
  Typer command in ``commands.py`` with typed args.
- Personal preferences. Put those in ``~/.bashrc``, not in graphids' CLI.

Recipes live in this dict so the catalog is one eyeball-able block:
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from graphids.cli.app import app

# ---------------------------------------------------------------- recipes
# Keep this list short. New entries must be:
#   - read-only OR a single non-graphids action (cancel, tail)
#   - parameter-free
#   - documented by the shell command itself
RECIPES: dict[str, str] = {
    # Queue / job state
    "q": "squeue -u $USER -M ALL",
    "qme": "squeue -u $USER",
    "qpend": "squeue -u $USER --states=PENDING --format='%.18i %.30j %.8T %.10r %.20S %.20e'",
    "qhist": "sacct -u $USER --starttime=today -X "
    "--format=JobID%15,JobName%30,State%12,Elapsed,NodeList -P | column -t -s'|' | head -50",
    "cancel": "scancel -u $USER",
    # Cluster capacity
    "nodes": "sinfo -p gpu,cpu --format='%P %a %D %T %N'",
    "gpus": "sinfo -p gpu,gpuserial,gpu-debug --states=idle --format='%P %D %t %N' 2>/dev/null",
    # Filesystem
    "disk": 'du -sh "/fs/scratch/PAS1266/$USER" "$GRAPHIDS_RUN_ROOT" 2>/dev/null',
    "quota": "df -h /fs/ess/PAS1266 /fs/scratch/PAS1266",
    # Logs ({run_dir}/.slurm_scripts/, project may also have slurm_logs/)
    "tail-latest": "ls -t slurm_logs/*.err 2>/dev/null | head -1 | xargs -r tail -f",
    "logs": "ls -lt slurm_logs/ 2>/dev/null | head -20",
}


# ---------------------------------------------------------------- registration
def _make(cmd: str) -> Callable[[], None]:
    def recipe() -> None:
        subprocess.run(cmd, shell=True, check=False)

    recipe.__doc__ = f"`{cmd}`"
    return recipe


for _name, _cmd in RECIPES.items():
    app.command(_name, rich_help_panel="Shortcuts")(_make(_cmd))
