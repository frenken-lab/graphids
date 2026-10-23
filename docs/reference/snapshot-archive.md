# Snapshot implementation archive

`snapshot-2026-10-16.tar.gz` preserves the exact pre-consolidation working tree,
including tracked and untracked source, experiment configurations, tests,
`pyproject.toml`, and `uv.lock`. Its companion JSON lists every file SHA256,
the archive SHA256, original Git status and revision, and extraction smoke checks.
Secrets, generated sites, build outputs, Git internals, datasets and run artifacts
are excluded. Existing lake datasets, reviewed manifests, clock repairs,
checkpoints, calibration, and results remain in their original locations.

October 23 maintenance subsequently consolidated retained event predictions and
archived obsolete run directories and logs. Cleanup inventories, archive paths
and checksums are recorded under
`/users/PAS2022/rf15/graphids-cleanup-audits/20261023T182040Z/`.
Reviewed inputs, the active event cache, the October 16 event baseline and the
current comparison suite remain in place. The source archive above is unchanged.

To reproduce snapshot experiments, extract into a separate directory:

```bash
mkdir snapshot-reproduction
tar -xzf archives/snapshot-2026-10-16.tar.gz -C snapshot-reproduction
```

Use that directory's configuration, lockfile, tests and scripts. Snapshot weights
and calibration are incompatible with the active event model. The active runtime
supports only current primitives; restore historical configurations with their archived source.
