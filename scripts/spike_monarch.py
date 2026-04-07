"""Monarch compute-node spike: validate actor training chain.

Tests the PipelineActor's training logic on a GPU node without
Monarch's runtime (SlurmJob / spawn_procs). Each step is independent
so failures are easy to diagnose.

Steps:
  1. torchmonarch importability
  2. SLURM env var inheritance (TMPDIR, CUDA_VISIBLE_DEVICES)
  3. bootstrap_staging (NFS -> TMPDIR)
  4. Config rendering via _prepare_stage
  5. Lightning instantiation + single-batch fit (fast_dev_run)

Usage: scripts/slurm/submit.sh spike-monarch
"""

from __future__ import annotations

import os
import sys
import time
import traceback


def _header(step: int, title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"Step {step}: {title}")
    print("=" * 50)


def main() -> int:
    t0 = time.monotonic()
    failures: list[str] = []

    # -- Step 1: torchmonarch import ------------------------------------------
    _header(1, "torchmonarch import")
    try:
        import monarch  # noqa: F401

        print(f"  version: {getattr(monarch, '__version__', 'unknown')}")
        from monarch.actor import Actor, endpoint  # noqa: F401

        print("  Actor + endpoint imported")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIP (not installed): {e}")
        print("  Actor fallback will be used")

    # -- Step 2: SLURM environment --------------------------------------------
    _header(2, "SLURM environment")
    for var in ("TMPDIR", "SLURM_JOB_ID", "SLURM_PARTITION", "CUDA_VISIBLE_DEVICES"):
        print(f"  {var}={os.environ.get(var, '<unset>')}")

    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir:
        print("  WARN: TMPDIR not set")
    else:
        print("  PASS")

    # -- Step 3: bootstrap_staging --------------------------------------------
    _header(3, "bootstrap_staging")
    try:
        from graphids.monarch._setup import bootstrap_staging

        bootstrap_staging("hcrl_ch")
        stage_dir = os.environ.get("KD_GAT_STAGE_DIR", "<unset>")
        print(f"  KD_GAT_STAGE_DIR={stage_dir}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failures.append("bootstrap_staging")

    # -- Step 4: PipelineActor + config rendering -----------------------------
    _header(4, "PipelineActor + _prepare_stage")
    try:
        from graphids.monarch.actors import PipelineActor

        # Use TMPDIR as lake_root so we don't pollute the experiment lake
        spike_lake = os.path.join(tmpdir or "/tmp", "graphids-spike")
        actor = PipelineActor(
            dataset="hcrl_ch",
            seed=42,
            scale="small",
            lake_root=spike_lake,
            conv_type="gatv2",
            variational=True,
        )
        model_type, ckpt_path, run_dir, resolved = actor._prepare_stage(
            stage="autoencoder",
            fusion_method="bandit",
            tla_overrides={"trainer_overrides": {"trainer.fast_dev_run": 1}},
            vgae_ckpt_path=None,
            gat_ckpt_path=None,
        )
        print(f"  model_type={model_type}")
        print(f"  run_dir={run_dir}")
        fdr = resolved.rendered.get("trainer", {}).get("fast_dev_run")
        print(f"  trainer.fast_dev_run={fdr}")
        if fdr != 1:
            print("  WARN: fast_dev_run not propagated — training will be long")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failures.append("_prepare_stage")
        # Fatal — can't continue to step 5
        _report(failures, t0)
        return 1

    # -- Step 5: Lightning instantiation + fit --------------------------------
    _header(5, "Lightning fast_dev_run fit")
    try:
        run = actor._instantiate_and_inject(resolved)
        print(f"  trainer: {type(run.trainer).__name__}")
        print(f"  model:   {type(run.model).__name__}")
        print(f"  data:    {type(run.datamodule).__name__}")

        run.trainer.fit(run.model, datamodule=run.datamodule)
        actor._cache_datasets_from(run.datamodule)

        print(f"  global_step={run.trainer.global_step}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failures.append("lightning_fit")

    _report(failures, t0)
    return 1 if failures else 0


def _report(failures: list[str], t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    else:
        print("ALL STEPS PASSED")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
