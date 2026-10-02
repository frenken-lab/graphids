"""Evaluate a cross-vehicle checkpoint at thresholds fixed before target tests.

Run on a compute node after fitting a cross_vehicle/*_fit.yml experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from torchmetrics.functional.classification import binary_average_precision

from graphids.exp.config import ExperimentConfig
from graphids.exp.runtime import build_component, build_model


def threshold_at_fpr(scores: torch.Tensor, fpr: float) -> float:
    """Highest attainable recall with empirical benign FPR no greater than fpr."""
    if not 0 < fpr < 1 or scores.numel() == 0:
        raise ValueError("requires benign calibration scores and 0 < fpr < 1")
    ordered = torch.sort(scores.float().flatten()).values
    permitted = math.floor(fpr * len(ordered))
    boundary = ordered[-1] if permitted == 0 else ordered[-permitted - 1]
    return float(torch.nextafter(boundary, torch.tensor(math.inf)))


def alert_count(scores: torch.Tensor, times: torch.Tensor, streams: torch.Tensor,
                threshold: float, refractory_s: float) -> int:
    count = 0
    last: dict[int, float] = {}
    for score, time, stream in zip(scores.tolist(), times.tolist(), streams.tolist(), strict=True):
        key = int(stream)
        if score >= threshold and (key not in last or time - last[key] >= refractory_s):
            count += 1
            last[key] = time
    return count


def elapsed_hours(times: torch.Tensor, streams: torch.Tensor) -> float:
    duration = 0.0
    for key in streams.unique():
        values = times[streams == key]
        if values.numel() > 1:
            duration += max(0.0, float(values[-1] - values[0]))
    return duration / 3600.0


def episode_latencies(scores: torch.Tensor, labels: torch.Tensor, times: torch.Tensor,
                      streams: torch.Tensor, threshold: float, *, attack_gap_s: float = 1.0,
                      ) -> tuple[list[float], int]:
    """Group injected frames into episodes despite benign frames between them."""
    results = []
    episodes = 0
    for stream in streams.unique():
        selected = streams == stream
        order = torch.argsort(times[selected], stable=True)
        stream_scores = scores[selected][order]
        stream_labels = labels[selected][order].bool()
        stream_times = times[selected][order]
        positive_times = stream_times[stream_labels]
        if positive_times.numel() == 0:
            continue
        boundaries = (positive_times[1:] - positive_times[:-1] > attack_gap_s).nonzero().flatten() + 1
        starts = torch.cat((torch.zeros(1, dtype=torch.long), boundaries))
        ends = torch.cat((boundaries, torch.tensor([positive_times.numel()])))
        for first, end in zip(starts.tolist(), ends.tolist(), strict=True):
            episodes += 1
            start_time, end_time = positive_times[first], positive_times[end - 1]
            alert = (stream_times >= start_time) & (stream_times <= end_time) & (stream_scores >= threshold)
            if bool(alert.any()):
                results.append(float(stream_times[alert][0] - start_time))
    return results, episodes


def score_partition(data, predictions: dict) -> dict[str, torch.Tensor]:
    mask = data.is_scored.bool()
    scores = predictions["scores"].float()
    if scores.ndim == 2:
        scores = scores[:, 1]
    expected = int(mask.sum())
    if scores.numel() != expected:
        raise ValueError(f"prediction coverage mismatch: {scores.numel()} versus {expected}")
    if ("event_id" not in predictions or
            set(predictions["event_id"].tolist()) != set(data.event_id[mask].tolist())):
        raise ValueError("prediction event IDs do not match the scored partition")
    return {
        "scores": scores,
        "labels": predictions["labels"].long(),
        "times": predictions["times"].double(),
        "streams": predictions["streams"].long(),
        "attack_type": predictions["attack_type"].long(),
    }


def metrics(partition: dict[str, torch.Tensor], threshold: float, *, fpr: float,
            refractory_s: float) -> dict:
    scores, labels = partition["scores"], partition["labels"]
    benign, attacked = labels == 0, labels == 1
    result = {
        "events": int(labels.numel()), "attacks": int(attacked.sum()),
        "threshold": threshold, "calibration_fpr_target": fpr,
        "recall": float((scores[attacked] >= threshold).float().mean()) if attacked.any() else None,
        "false_positive_rate": float((scores[benign] >= threshold).float().mean()) if benign.any() else None,
        "auprc": float(binary_average_precision(scores, labels)) if attacked.any() and benign.any() else None,
    }
    times, streams = partition["times"], partition["streams"]
    benign_hours = elapsed_hours(times[benign], streams[benign])
    result["benign_hours"] = benign_hours
    result["false_alerts_per_hour"] = (
        alert_count(scores[benign], times[benign], streams[benign], threshold, refractory_s) / benign_hours
        if benign_hours > 0 else None
    )
    latencies, episodes = episode_latencies(scores, labels, times, streams, threshold)
    result["attack_episodes"] = episodes
    result["detected_episodes"] = len(latencies)
    result["detected_episode_latency_median_s"] = float(torch.tensor(latencies).median()) if latencies else None
    result["per_attack"] = {}
    for code in partition["attack_type"].unique().tolist():
        if code == 0:
            continue
        selected = benign | ((partition["attack_type"] == code) & attacked)
        if not (attacked & selected).any():
            continue
        sub_scores, sub_labels = scores[selected], labels[selected]
        result["per_attack"][str(code)] = {
            "attacks": int(sub_labels.sum()),
            "auprc": float(binary_average_precision(sub_scores, sub_labels)),
            "recall": float((sub_scores[sub_labels == 1] >= threshold).float().mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="fit experiment YAML")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fpr", type=float, default=0.001)
    parser.add_argument("--refractory-s", type=float, default=1.0)
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    run = experiment.build_run(name=experiment.experiment_name, stage="fit")
    config = run.payload.model_dump(mode="json")
    dm = build_component(config["data"])
    dm.setup(None)
    dm._tests = {"source_validation": dm.val_data, **dm.test_data}
    model = build_model(config, dm)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    trainer.test(model, datamodule=dm)
    partitions = {name: score_partition(data, model._test_predictions[name])
                  for name, data in dm.test_data.items()}
    source = partitions.pop("source_validation")
    calibration = partitions.pop("target_calibration")
    source_threshold = threshold_at_fpr(source["scores"][source["labels"] == 0], args.fpr)
    target_threshold = threshold_at_fpr(calibration["scores"][calibration["labels"] == 0], args.fpr)
    report = {
        "config": str(args.config.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "fpr_target": args.fpr, "refractory_s": args.refractory_s,
        "source_calibration_events": int((source["labels"] == 0).sum()),
        "target_calibration_events": int((calibration["labels"] == 0).sum()),
        "zero_shot": {name: metrics(p, source_threshold, fpr=args.fpr,
                                    refractory_s=args.refractory_s) for name, p in partitions.items()},
        "normal_target_adaptation": {name: metrics(p, target_threshold, fpr=args.fpr,
                                                    refractory_s=args.refractory_s) for name, p in partitions.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
