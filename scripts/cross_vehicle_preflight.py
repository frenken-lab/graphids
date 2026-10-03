"""Exercise 1,024-lane raw and behavioral training batches on one V100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.exp.config import ExperimentConfig
from graphids.exp.runtime import build_component, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("configs", type=Path, nargs="+")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V100 CUDA preflight requires a GPU compute node")
    results = []
    for path in args.configs:
        experiment = ExperimentConfig.from_yaml(path)
        run = experiment.build_run(name=experiment.experiment_name, stage="fit")
        config = run.payload.model_dump(mode="json")
        dm = build_component(config["data"])
        dm.setup(None)
        if dm.stream_lanes != 1024 or dm.chunk_size != 32:
            raise ValueError(f"{path} is not the revised 1,024-lane configuration")
        if config["model"].get("scan", {}).get("cuda_graph") is not False:
            raise ValueError(f"{path} must disable CUDA graphs for V100 memory safety")
        dm.coverage_policy = "diagnostic"
        model = build_model(config, dm)
        # A partial probe intentionally bypasses the production full-epoch audit.
        dm.train_dataloader = lambda module=dm: module._lane_loader(module.train_data, training=True)
        probe_batches = 40
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1,
                             limit_train_batches=probe_batches, limit_val_batches=0,
                             num_sanity_val_steps=0, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(model, datamodule=dm)
        results.append({"config": str(path), "training_batches": probe_batches,
                        "optimizer_steps": trainer.global_step,
                        "gpu_memory_gib_peak": torch.cuda.max_memory_allocated() / 1024**3})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        del trainer, model, dm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
