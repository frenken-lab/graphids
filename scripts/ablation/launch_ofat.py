#!/usr/bin/env python
"""launch_ofat.py — submit the OFAT ablation DAG to SLURM.

Replaces launch_ofat.sh. Key differences from the bash version:

- Jids held in memory → Stage 3's "Job dependency problem" dep-race is
  gone. No second sbatch query needed between stages.
- Per-group walltime override. unsupervised/* (VGAE/GAE/DGI) have
  max_epochs=1200 at ~9 s/epoch on H100 ≈ 3 h, so the profile default
  (1:30 on Cardinal) walltimes them. This launcher passes --time 3:30:00
  for that group. Empirical: jids 8724431, 8724439, 8724441 walltimed
  2026-04-23 at 1:24:xx on the 1:30 wall.
- Idempotent — queries MLflow for each variant's latest fit status. If
  the latest attempt is FINISHED, skip re-submission. Any other status
  (FAILED/RUNNING/KILLED/missing) triggers a fresh submit. MLflow owns
  truth; no filesystem markers to drift.
- No MLflow parent-run plumbing. That's the existing `_mlflow.py`
  surgery (Phase B), not the launcher's concern.

Per-job sbatch plumbing still delegates to scripts/run — the launcher
only owns the DAG shape.

Usage:
  scripts/ablation/launch_ofat.py                          # set_01, seeds 42/123/777
  scripts/ablation/launch_ofat.py --dataset set_02         # different dataset
  scripts/ablation/launch_ofat.py --dry-run                # print commands, no submit
  scripts/ablation/launch_ofat.py --seed 42                # single seed
  scripts/ablation/launch_ofat.py --cluster cardinal       # target cluster
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Guard against running under the wrong venv. `.venv/bin/python` is a
# symlink to the system Python, so venv isolation lives in VIRTUAL_ENV,
# not in sys.executable. Fail loudly instead of silently loading the
# wrong site-packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPECTED_VENV = str(_PROJECT_ROOT / ".venv")
_ACTUAL_VENV = os.environ.get("VIRTUAL_ENV", "")
if _ACTUAL_VENV != _EXPECTED_VENV:
    sys.exit(
        f"launch_ofat.py: wrong virtualenv active.\n"
        f"  expected VIRTUAL_ENV={_EXPECTED_VENV}\n"
        f"  got      VIRTUAL_ENV={_ACTUAL_VENV or '(none)'}\n"
        f"  fix: `source {_EXPECTED_VENV}/bin/activate` and re-run."
    )

DEFAULT_SEEDS = (42, 123, 777)
DEFAULT_DATASET = "set_01"

# Groups whose presets exceed the profile's default `long` wall.
# Empirical (2026-04-23, Cardinal H100):
#   VGAE/GAE/DGI at max_epochs=1200, ~9 s/epoch → ~180 min needed.
# Everything else uses profile default.
LONG_WALL_GROUPS: frozenset[str] = frozenset({"unsupervised"})
LONG_WALL_TIME: str = "3:30:00"

# Stage 1 parallel variants, ordered by group.
STAGE1_VARIANTS: tuple[tuple[str, str], ...] = (
    ("conv_type", "gat"),
    ("conv_type", "gatv2"),
    ("conv_type", "gps"),
    ("unsupervised", "gae"),
    ("unsupervised", "dgi"),
    ("gat_sampling", "none"),
    ("gat_sampling", "curriculum_random"),
    ("gat_loss", "ce"),
    ("gat_loss", "weighted_ce"),
    ("id_encoding", "lookup"),
    ("id_encoding", "learned_unk"),
    ("id_encoding", "hash"),
)
FUSION_METHODS: tuple[str, ...] = ("bandit", "dqn", "mlp", "weighted_avg")

# Sentinel for "fit already complete; no new jid, but still chain test".
_FIT_ALREADY_COMPLETE = -1


def _run_dir(lake_root: Path, dataset: str, group: str, variant: str, seed: int) -> Path:
    return lake_root / dataset / "ablations" / group / variant / f"seed_{seed}"


def _fit_is_complete(dataset: str, group: str, variant: str, seed: int) -> bool:
    """Return True iff the *latest* fit attempt for this variant+seed is FINISHED.

    "Latest" matters: MLflow accumulates history across refactors, so an
    old FINISHED run from a prior code version can coexist with today's
    FAILED/RUNNING run. We only trust the most recent attempt.

    RUNNING (incl. zombies from SLURM-killed processes whose MLflow run
    never flipped to FAILED) returns False → safe re-run. The
    alternative was `mlflow_reap_zombies.py` (deleted 2026-04-23); we
    accept stale RUNNING rows in the UI in exchange for no reaper cron.
    """
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        lake = os.environ.get("GRAPHIDS_LAKE_ROOT")
        if lake:
            tracking_uri = f"sqlite:///{lake}/mlflow.db"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=(
            f"tags.`graphids.dataset` = '{dataset}' AND "
            f"tags.`graphids.group` = '{group}' AND "
            f"tags.`graphids.variant` = '{variant}' AND "
            f"tags.`graphids.seed` = '{seed}' AND "
            f"tags.`graphids.phase` = 'fit'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return not df.empty and df.iloc[0]["status"] == "FINISHED"


def _submit_fit(
    cfg: Path,
    seed: int,
    dataset: str,
    lake_root: Path,
    cluster: str,
    dep_jid: int | None = None,
    dry_run: bool = False,
) -> int:
    """Submit fit job. Returns jid (>0), 0 for dry-run, or _FIT_ALREADY_COMPLETE."""
    group = cfg.parent.name
    variant = cfg.stem

    if _fit_is_complete(dataset, group, variant, seed):
        print(
            f"  [skip] fit already FINISHED in MLflow: {group}/{variant} seed_{seed}",
            file=sys.stderr,
        )
        return _FIT_ALREADY_COMPLETE

    cmd: list[str] = [
        "scripts/run",
        str(cfg),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--lake-root",
        str(lake_root),
    ]
    if cluster:
        cmd += ["--cluster", cluster]
    if group in LONG_WALL_GROUPS:
        cmd += ["--time", LONG_WALL_TIME]
    if dry_run:
        cmd += ["--dry-run"]

    env = os.environ.copy()
    if dep_jid is not None and dep_jid > 0:
        env["SBATCH_DEP"] = f"afterok:{dep_jid}"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"scripts/run failed for {group}/{variant} seed_{seed} (exit {result.returncode})"
        )
    # scripts/run emits only the jid on stdout (banner goes to stderr).
    jid_str = result.stdout.strip()
    if not jid_str:
        raise RuntimeError(f"scripts/run returned empty stdout for {cfg}")
    return int(jid_str)


def _chain_test(
    cfg: Path,
    seed: int,
    dataset: str,
    lake_root: Path,
    fit_jid: int,
    cluster: str,
    dry_run: bool = False,
) -> None:
    """Submit afterok CPU test. Runs without dep if fit was already complete.

    --mem 32G: classification_test_metrics buffers full (N, K) probs on CPU;
    the 16G profile default OOM'd on seed 42 (test-gat 8724434 @ 16.77G).
    Long-term fix is stream-compute via torchmetrics; 32G is the interim.
    """
    if fit_jid == 0:  # dry-run sentinel — don't chain
        return

    group = cfg.parent.name
    variant = cfg.stem
    run_dir = _run_dir(lake_root, dataset, group, variant, seed)
    ckpt = run_dir / "checkpoints" / "best_model.ckpt"

    cmd: list[str] = [
        "scripts/run",
        str(cfg),
        "--action",
        "test",
        "--mode",
        "cpu",
        "--length",
        "short",
        "--mem",
        "32G",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--lake-root",
        str(lake_root),
        "--ckpt-path",
        str(ckpt),
    ]
    if cluster:
        cmd += ["--cluster", cluster]
    if dry_run:
        cmd += ["--dry-run"]

    env = os.environ.copy()
    if fit_jid > 0:
        env["SBATCH_DEP"] = f"afterok:{fit_jid}"
    # If fit already complete (_FIT_ALREADY_COMPLETE), submit test with no dep
    # — runs immediately against the existing ckpt.

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"test chain failed for {group}/{variant} seed_{seed} (exit {result.returncode})"
        )


def _submit_extract_states(
    seed: int,
    dataset: str,
    lake_root: Path,
    vgae_jid: int,
    focal_jid: int,
    cluster: str,
    dry_run: bool = False,
) -> int:
    """Stage 3 — extract VGAE+GAT fusion states. Afterok both upstream fits.

    When upstream fits are already complete (jid == _FIT_ALREADY_COMPLETE),
    submits with no dep so extraction runs immediately.
    """
    vgae_ckpt = (
        _run_dir(lake_root, dataset, "unsupervised", "vgae", seed)
        / "checkpoints"
        / "best_model.ckpt"
    )
    gat_ckpt = (
        _run_dir(lake_root, dataset, "gat_loss", "focal", seed) / "checkpoints" / "best_model.ckpt"
    )
    out = lake_root / dataset / "ablations" / "fusion_states" / f"seed_{seed}"
    inner = (
        f"python -m graphids extract-fusion-states "
        f"--vgae-ckpt {vgae_ckpt} --gat-ckpt {gat_ckpt} "
        f"--dataset {dataset} --seed {seed} --output-dir {out}"
    )

    cmd: list[str] = [
        "scripts/run",
        "--mode",
        "gpu",
        "--mem",
        "36G",
        "--time",
        "0:30:00",
        "--command",
        inner,
    ]
    if cluster:
        cmd += ["--cluster", cluster]
    if dry_run:
        cmd += ["--dry-run"]

    env = os.environ.copy()
    dep_jids = [str(j) for j in (vgae_jid, focal_jid) if j > 0]
    if dep_jids:
        env["SBATCH_DEP"] = "afterok:" + ":".join(dep_jids)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"extract-fusion-states failed for seed_{seed} (exit {result.returncode})"
        )
    return int(result.stdout.strip())


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a bash-style .env. Ignores comments/blanks."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Repeat for multiple seeds; default %s" % (DEFAULT_SEEDS,),
    )
    parser.add_argument("--cluster", default="", help="e.g. cardinal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    _load_dotenv(project_root / ".env")

    lake_env = os.environ.get("GRAPHIDS_LAKE_ROOT")
    if not lake_env:
        sys.exit("GRAPHIDS_LAKE_ROOT must be set in .env")
    lake_root = Path(lake_env) / "dev" / os.environ["USER"]

    seeds: tuple[int, ...] = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    cfg_root = Path("configs/ablations")

    vgae_jids: dict[int, int] = {}
    focal_jids: dict[int, int] = {}
    states_jids: dict[int, int] = {}

    # Stage 0: baseline VGAEs
    print(f"=== Stage 0: baseline VGAEs ({len(seeds)} jobs) ===", file=sys.stderr)
    for seed in seeds:
        cfg = cfg_root / "unsupervised" / "vgae.jsonnet"
        jid = _submit_fit(cfg, seed, args.dataset, lake_root, args.cluster, dry_run=args.dry_run)
        vgae_jids[seed] = jid
        _chain_test(cfg, seed, args.dataset, lake_root, jid, args.cluster, dry_run=args.dry_run)
        print(f"  seed={seed} -> vgae jid={jid}", file=sys.stderr)

    # Stage 1: parallel standalone variants + focal (baseline for fusion)
    print(
        f"=== Stage 1: standalone ({len(STAGE1_VARIANTS) + 1} x {len(seeds)} jobs) ===",
        file=sys.stderr,
    )
    for seed in seeds:
        for group, variant in STAGE1_VARIANTS:
            cfg = cfg_root / group / f"{variant}.jsonnet"
            jid = _submit_fit(
                cfg, seed, args.dataset, lake_root, args.cluster, dry_run=args.dry_run
            )
            _chain_test(cfg, seed, args.dataset, lake_root, jid, args.cluster, dry_run=args.dry_run)
        cfg = cfg_root / "gat_loss" / "focal.jsonnet"
        jid = _submit_fit(cfg, seed, args.dataset, lake_root, args.cluster, dry_run=args.dry_run)
        focal_jids[seed] = jid
        _chain_test(cfg, seed, args.dataset, lake_root, jid, args.cluster, dry_run=args.dry_run)
        print(f"  seed={seed} -> focal jid={jid}", file=sys.stderr)

    # Stage 2: curriculum_vgae afterok VGAE
    print(
        f"=== Stage 2: curriculum_vgae ({len(seeds)} jobs, afterok Stage 0) ===",
        file=sys.stderr,
    )
    for seed in seeds:
        cfg = cfg_root / "gat_sampling" / "curriculum_vgae.jsonnet"
        jid = _submit_fit(
            cfg,
            seed,
            args.dataset,
            lake_root,
            args.cluster,
            dep_jid=vgae_jids[seed],
            dry_run=args.dry_run,
        )
        _chain_test(cfg, seed, args.dataset, lake_root, jid, args.cluster, dry_run=args.dry_run)

    # Stage 3: extract-fusion-states afterok (vgae, focal)
    print(f"=== Stage 3: extract-fusion-states ({len(seeds)} jobs) ===", file=sys.stderr)
    for seed in seeds:
        jid = _submit_extract_states(
            seed,
            args.dataset,
            lake_root,
            vgae_jids[seed],
            focal_jids[seed],
            args.cluster,
            dry_run=args.dry_run,
        )
        states_jids[seed] = jid
        print(f"  seed={seed} -> states jid={jid}", file=sys.stderr)

    # Stage 4: fusion afterok Stage 3
    print(f"=== Stage 4: fusion ({len(FUSION_METHODS)} x {len(seeds)} jobs) ===", file=sys.stderr)
    for seed in seeds:
        for method in FUSION_METHODS:
            cfg = cfg_root / "fusion" / f"{method}.jsonnet"
            jid = _submit_fit(
                cfg,
                seed,
                args.dataset,
                lake_root,
                args.cluster,
                dep_jid=states_jids[seed],
                dry_run=args.dry_run,
            )
            _chain_test(cfg, seed, args.dataset, lake_root, jid, args.cluster, dry_run=args.dry_run)

    print(file=sys.stderr)
    print("=== Launched ===", file=sys.stderr)
    print(f"Dataset:      {args.dataset}", file=sys.stderr)
    print(f"Seeds:        {list(seeds)}", file=sys.stderr)
    print(f"VGAE jids:    {vgae_jids}", file=sys.stderr)
    print(f"Focal jids:   {focal_jids}", file=sys.stderr)
    print(f"States jids:  {states_jids}", file=sys.stderr)


if __name__ == "__main__":
    main()
