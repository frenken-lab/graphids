"""Fit a typed source-learning experiment and test its source-selected checkpoint.

Run in a one-GPU Slurm allocation. The final target test is performed only
once, after fitting; no target metrics enter checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from graphids.exp.config import ExperimentConfig
from graphids.exp.ray_backend import launch_run


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/graph_mamba_ctat_learning.yml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPHIDS_MLFLOW_MODE"] = "disabled"
    cfg = ExperimentConfig.from_yaml(args.config)
    fit = cfg.build_run(name=cfg.experiment_name, stage="fit", run_dir=root / "fit")
    fit.write_config()
    report = dict(
        stage="fit",
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        implementation_commit=cfg.git_sha,
        worktree_status=subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for base in ("graphids", "scripts")
            for path in sorted(Path(base).rglob("*.py"))
        },
        config=cfg.model_dump(mode="json"),
        target_used_for_checkpoint_selection=False,
        selection_metric=cfg.config["callbacks"]["checkpoint"]["init_args"]["monitor"],
        selection_mode="max",
    )

    def save():
        (root / "learning-run.json").write_text(json.dumps(report, indent=2))

    save()
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False)
    started = time.monotonic()
    try:
        summary = launch_run(fit)
        if summary is None or summary.status != "finished":
            raise RuntimeError(f"fit did not finish: {summary}")
        report["fit"] = asdict(summary)
        report["fit_seconds"] = time.monotonic() - started
        checkpoint = root / "fit/checkpoints/best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        curve = json.loads((root / "fit/artifacts/learning-curve.json").read_text())
        report["epochs_completed"] = len(curve["epochs"])
        report["optimizer_steps"] = curve["epochs"][-1]["optimizer_steps"]
        if report["epochs_completed"] != cfg.config["trainer"]["max_epochs"]:
            raise AssertionError(
                "learning run did not complete the requested epoch budget"
            )
        if any(
            epoch["training_steps_this_rank"] != epoch["available_training_batches"]
            for epoch in curve["epochs"]
        ):
            raise AssertionError(
                "learning run must train the complete source schedule each epoch"
            )
        report["stage"] = "final_target_evaluation"
        save()
        # New output directory; inherit the exact source model/data config.
        trainer = dict(cfg.config["trainer"], enable_checkpointing=False, logger=False)
        test = cfg.build_run(
            name=cfg.experiment_name + "_test",
            stage="test",
            run_dir=root / "test",
            config={"ckpt_path": str(checkpoint), "trainer": trainer, "callbacks": {}},
        )
        began = time.monotonic()
        summary = launch_run(test)
        if summary is None or summary.status != "finished":
            raise RuntimeError(f"target evaluation did not finish: {summary}")
        report.update(
            stage="complete",
            test=asdict(summary),
            test_seconds=time.monotonic() - began,
            total_seconds=time.monotonic() - started,
            coverage=json.loads((root / "test/artifacts/coverage.json").read_text()),
        )
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(stage="failed", error=repr(exc))
        save()
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
