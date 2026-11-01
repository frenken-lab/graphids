"""Paired classifier adaptation on a frozen event model; never full-model resume.

python -m scripts.event_classifier_pilot --config WINNER.yml \
    --checkpoint WINNER.ckpt --output NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from graphids.evaluation import EventScores
from graphids.models.event_graph_mamba import EventGraphMambaState


def persist(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def positive_weights(recordings, cap):
    """Equalize positive recording mass subject to a FINAL cap and fixed total.

    Negative weights stay one. Capped water filling preserves the original
    positive mass, unlike clipping inverse counts and then rescaling above cap.
    Validation recordings and unknown labels cannot affect these weights.
    """
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("positive weight cap must be finite and >=1")
    groups = {
        r.recording.key: dict(
            rows=r.count,
            positives=int((r.arrays["labels"] == 1).sum()),
            negatives=int((r.arrays["labels"] == 0).sum()),
            unknown=int((r.arrays["labels"] < 0).sum()),
        )
        for r in recordings
        if r.recording.role == "train_query"
    }
    counts = np.array([g["positives"] for g in groups.values()], dtype=np.float64)
    positive = counts[counts > 0]
    if not len(positive) or not sum(g["negatives"] for g in groups.values()):
        raise ValueError("classifier pilot requires both training classes")
    low, high = 0.0, float(positive.max())
    for _ in range(64):
        scale = (low + high) / 2
        if np.minimum(cap * positive, scale).sum() < positive.sum():
            low = scale
        else:
            high = scale
    for group in groups.values():
        count = group["positives"]
        group["positive_weight"] = min(cap, high / count) if count else 1.0
    return groups


class FrozenReadout:
    """Capture the existing classifier input without changing production code."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.captured = []

        def capture(module, args):
            self.captured.append(args[0].detach())

        self.hook = self.model.classifier.register_forward_pre_hook(capture)
        return self

    def __exit__(self, *exc):
        self.hook.remove()

    @torch.no_grad()
    def scan(self, batches, states):
        self.captured.clear()
        outputs, following = self.model.scan(batches, states, detach=True)
        if len(self.captured) != 1:
            raise RuntimeError("expected one classifier readout per stream group")
        return outputs, following, self.captured.pop()


def head_loss(logits, batch, positive_weight, denominator):
    known = batch.scored & (batch.labels >= 0)
    loss = F.binary_cross_entropy_with_logits(
        logits, batch.labels.clamp_min(0).to(logits), reduction="none"
    )
    weight = torch.where(batch.labels == 1, positive_weight, 1.0)
    # Normalize over the complete optimizer window, never its sum of weights.
    return (loss * weight * known).sum() / max(denominator, 1)


def fit_heads(model, dm, *, epochs=5, cap=10.0, learning_rate=1e-3):
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    groups = positive_weights(dm.event_recordings.values(), cap)
    expected = {name: group["rows"] for name, group in groups.items()}
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    heads = {
        name: deepcopy(model.classifier).requires_grad_(True)
        for name in ("ordinary", "balanced")
    }
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=model.hparams.weight_decay
        )
        for name, head in heads.items()
    }
    loader, history = dm.train_dataloader(), []
    with FrozenReadout(model) as readout:
        for epoch in range(epochs):
            started, states, coverage, steps = time.monotonic(), {}, {}, 0
            totals = dict.fromkeys(heads, 0.0)
            for window in loader:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                for group in window.groups:
                    batches = [b.to(device, non_blocking=True) for b in group]
                    _, following, features = readout.scan(
                        batches,
                        [
                            states.get(b.recording, EventGraphMambaState())
                            for b in batches
                        ],
                    )
                    for batch, state, z in zip(
                        batches, following, features, strict=True
                    ):
                        if batch.row_start != coverage.get(batch.recording, 0):
                            raise ValueError("duplicate/missing training rows")
                        coverage[batch.recording] = batch.row_start + len(batch.ids)
                        if batch.recording_end:
                            states.pop(batch.recording, None)
                        else:
                            states[batch.recording] = state
                        for name, head in heads.items():
                            weight = (
                                groups[batch.recording]["positive_weight"]
                                if name == "balanced"
                                else 1.0
                            )
                            loss = head_loss(
                                head(z).squeeze(-1),
                                batch,
                                weight,
                                window.classification_targets,
                            )
                            loss.backward()
                            totals[name] += (
                                float(loss.detach()) * window.classification_targets
                            )
                if window.classification_targets:
                    for optimizer in optimizers.values():
                        optimizer.step()
                    steps += 1
            if coverage != expected or states:
                raise ValueError("incomplete training coverage/state reset")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            known = sum(g["positives"] + g["negatives"] for g in groups.values())
            history.append(
                dict(
                    epoch=epoch + 1,
                    steps=steps,
                    coverage=coverage,
                    loss={name: value / known for name, value in totals.items()},
                    seconds=time.monotonic() - started,
                )
            )
            print(json.dumps(history[-1]), flush=True)
    return heads, dict(training_groups=groups, epochs=history)


@torch.no_grad()
def evaluate(model, heads, dm, output):
    device = next(model.parameters()).device
    scores = {name: EventScores(output / name) for name in ("original", *heads)}
    state = None
    with FrozenReadout(model) as readout:
        for host in dm.val_dataloader():
            batch = host.to(device, non_blocking=True)
            if batch.row_start == 0:
                state = None
            outputs, following, features = readout.scan(
                [batch], [state or EventGraphMambaState()]
            )
            state = following[0]
            logits = {"original": outputs[0]["classification_logits"]}
            logits.update(
                {name: head(features[0]).squeeze(-1) for name, head in heads.items()}
            )
            mask = torch.arange(8, device=device)[None] < batch.lengths[:, None]
            mask &= batch.payload_length_verified
            losses = model._payload_nll(outputs[0]["payload_logits"], batch.payload)
            anomaly = (losses * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            for name, values in scores.items():
                rows = (
                    torch.stack(
                        (
                            batch.timestamps,
                            batch.ids,
                            batch.labels,
                            logits[name].sigmoid(),
                            anomaly,
                            mask.any(-1),
                        ),
                        -1,
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
                values.write(
                    rows,
                    recording=batch.recording,
                    source_digest=batch.source_digest,
                    row_start=batch.row_start,
                )
    expected = {
        r.recording.key: r.count
        for r in dm.event_recordings.values()
        if r.recording.role == "val_query"
    }
    if not expected:
        raise ValueError("classifier pilot requires validation recordings")
    results = {}
    for name, values in scores.items():
        metrics = values.finish(expected)
        aps = [
            g["classification_frame_ap"]
            for g in values.by_recording.values()
            if g["classification_positives"] and g["classification_negatives"]
        ]
        results[name] = dict(
            metrics=metrics,
            by_recording=values.by_recording,
            recording_macro_ap=float(np.mean(aps)) if aps else None,
            coverage=values.coverage,
        )
    return results


def main():
    from graphids.exp.config import ExperimentConfig
    from graphids.exp.run import build_component
    from graphids.exp.slurm import dependency_versions, file_hash
    from graphids.models.event_graph_mamba import EventGraphMamba
    from scripts.event_model_gate import source_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--positive-cap", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.positive_cap)
        or args.positive_cap < 1
    ):
        parser.error("use positive epochs/rate and a finite positive cap >=1")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("run this pilot inside a single-GPU Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="running",
        started_at=datetime.now(UTC).isoformat(),
        stage="frozen_classifier_adaptation",
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_hash(args.checkpoint),
        configuration_sha256=file_hash(args.config),
        head_epochs=args.epochs,
        positive_cap=args.positive_cap,
        learning_rate=args.learning_rate,
        selection="fixed final epoch; no threshold fitting",
        objective="positive recording balance; negative weights=1; fixed positive mass; known-window denominator",
        seed=42,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        device=torch.cuda.get_device_name(),
    )
    persist(args.output / "pilot.json", receipt)
    try:
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = ExperimentConfig.from_yaml(args.config)
        dm = build_component(cfg.config["data"])
        dm.setup()
        model = EventGraphMamba.load_from_checkpoint(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        before = fingerprint(model)
        model.prepare_from_datamodule(dm)
        model.cuda()
        if fingerprint(model) != before:
            raise ValueError("checkpoint timing buffers must already be fitted")
        receipt["source_manifest_digest"] = dm.manifest.digest
        persist(args.output / "source-manifest.json", asdict(dm.manifest))
        git_head, diff, hashes = source_provenance(Path(__file__).resolve().parents[1])
        persist(args.output / "source-sha256.json", hashes)
        persist(args.output / "dependency-versions.json", dependency_versions())
        (args.output / "working-tree.diff").write_bytes(diff)
        (args.output / "configuration.yml").write_bytes(args.config.read_bytes())
        receipt.update(
            git_head=git_head,
            frozen_model_sha256=before,
            classifier_parameters=sum(p.numel() for p in model.classifier.parameters()),
            weight_decay=model.hparams.weight_decay,
        )
        heads, training = fit_heads(
            model,
            dm,
            epochs=args.epochs,
            cap=args.positive_cap,
            learning_rate=args.learning_rate,
        )
        receipt.update(training)
        receipt["results"] = evaluate(model, heads, dm, args.output / "scores")
        if fingerprint(model) != before:
            raise RuntimeError("frozen model parameters/buffers changed")
        artifact = dict(
            stage=receipt["stage"],
            base_checkpoint_sha256=receipt["checkpoint_sha256"],
            source_manifest_digest=dm.manifest.digest,
            heads={
                name: {key: value.cpu() for key, value in head.state_dict().items()}
                for name, head in heads.items()
            },
        )
        torch.save(artifact, args.output / "classifier-heads.pt")
        receipt.update(
            status="finished",
            finished_at=datetime.now(UTC).isoformat(),
            frozen_model_unchanged=True,
            head_artifact_sha256=file_hash(args.output / "classifier-heads.pt"),
        )
        persist(args.output / "pilot.json", receipt)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist(args.output / "pilot.json", receipt)
        raise


if __name__ == "__main__":
    main()
