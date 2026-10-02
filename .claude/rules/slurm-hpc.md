# GraphIDS SLURM / HPC Conventions

OSC Pitzer/Cardinal/Ascend use account PAS1266, Python 3.12, and the project
`.venv`. Source `scripts/slurm/_preamble.sh` inside allocations. GPU diagnostics
share `_diagnostic.sh`; keep each script's resource and correctness gates.

- Never run pytest or training workloads on login nodes. Use a SLURM allocation.
- Static lint, git, config inspection, and dry-run rendering are login-node safe.
- Submit experiments with `gx exp submit CONFIG -C pitzer`; `--dry-run` prints
  the allocation script without writing snapshots or submitting.
- Submission saves expanded YAML next to the script. This freezes config,
  while Python source still comes from the chosen checkout.
- Use an immutable code snapshot for performance comparisons. Trials run in
  fresh processes on the same GPU with identical weights and owned workloads.
- SLURM logs belong under the configured log root; results and checkpoints
  belong under `GRAPHIDS_RUN_ROOT`. Use unique diagnostic output directories.
- Read raw CSVs and caches directly from the persistent ESS lake. Reintroduce
  staging only after measurements show I/O limits; do not call retired commands.

See `docs/reference/submit-flow.md` for the live submission path.
