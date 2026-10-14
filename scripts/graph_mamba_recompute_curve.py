"""Correct source curves only from inventoried retained weights; never retrain/select on target."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from graphids.core.models.score_columns import ScoreTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    dm = CANSnapshotDataModule(
        manifest_path=str(args.manifest),
        datasets=["set_01"],
        cache_root=str(args.cache_root),
    )
    dm.setup()
    inventory = sorted(
        json.loads(args.inventory.read_text()), key=lambda row: row["epoch"]
    )
    report = dict(
        version=2,
        original_selected_epoch=4,
        selection_unchanged=True,
        corrected_from_retained_weights=True,
        epochs=[],
    )
    for entry in inventory:
        print(f"Recomputing retained epoch {entry['epoch'] + 1}", flush=True)
        if (
            hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError("retained checkpoint hash mismatch")
        net = VehicleGraphMambaJoint.load_from_checkpoint(
            entry["path"], map_location="cpu"
        )
        net.prepare_from_datamodule(dm)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.validate(net, datamodule=dm, verbose=False)
        table = ScoreTable(net._validation_scores)
        table.apply_bundles(net.calibration_bundles)
        root = args.output / f"epoch-{entry['epoch'] + 1:02d}"
        root.mkdir()
        metrics = table.report(net.calibration_bundles)
        (root / "source-metrics-v2.json").write_text(json.dumps(metrics, indent=2))
        (root / "calibration-v2.json").write_text(
            json.dumps(
                {k: v.to_dict() for k, v in net.calibration_bundles.items()}, indent=2
            )
        )
        with (root / "source-scores-v2.jsonl").open("w") as handle:
            for row in table:
                handle.write(json.dumps(row) + "\n")
        report["epochs"].append(
            dict(**entry, displayed_epoch=entry["epoch"] + 1, source=metrics)
        )
        (args.output / "corrected-learning-curve-v2.json").write_text(
            json.dumps(report, indent=2)
        )
        if entry == inventory[-1]:
            # Source calibration is frozen before this final-checkpoint diagnostic.
            os.environ["GRAPHIDS_RUN_DIR"] = str(root.resolve())
            trainer.test(net, datamodule=dm, verbose=False)
            report["actual_final_target"] = json.loads(
                (root / "artifacts/metrics-v2.json").read_text()
            )
            report["actual_final_target_calibrated"] = json.loads(
                (root / "artifacts/target-calibrated-metrics-v2.json").read_text()
            )
    (args.output / "corrected-learning-curve-v2.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
