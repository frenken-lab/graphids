"""Shared causal Mamba temporal encoding and bounded same-time graph readout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from graphids.core.data.snapshots.batch import CANSnapshotBatch


@dataclass
class GraphMambaState:
    recording: str = ""
    context_digest: str = ""
    caches: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    version: int = 1
    last_bin: int = -1


class GraphReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width + 4, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.gate = nn.Linear(width * 2, width)
        self.update = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, batch, *, graph_enabled=True):
        b, t, n, d = h.shape
        k = batch.neighbors.shape[-1]
        index = batch.neighbors[:, None].expand(b, t, n, k)
        sender = h.gather(
            2, index.reshape(b, t, n * k, 1).expand(b, t, n * k, d)
        ).reshape(b, t, n, k, d)
        sender_live = batch.live.gather(2, index.reshape(b, t, n * k)).reshape(
            b, t, n, k
        )
        observed_age = batch.x[..., [20, 1]]
        metadata = observed_age.gather(
            2, index.reshape(b, t, n * k, 1).expand(b, t, n * k, 2)
        ).reshape(b, t, n, k, 2)
        edge = batch.edge_features[:, None].expand(b, t, n, k, 2)
        mask = batch.edge_mask[:, None] & sender_live & batch.live[:, :, :, None]
        if not graph_enabled:
            mask = torch.zeros_like(mask)
        messages = self.message(torch.cat((sender, metadata, edge), -1))
        aggregate = (messages * mask.unsqueeze(-1)).sum(3) / mask.sum(3).clamp_min(
            1
        ).unsqueeze(-1)
        combined = torch.cat((h, aggregate), -1)
        out = self.norm(
            h + torch.sigmoid(self.gate(combined)) * F.silu(self.update(combined))
        )
        return out * batch.live.unsqueeze(-1)


class VehicleGraphMambaJoint(pl.LightningModule):
    def __init__(
        self,
        *,
        width=128,
        layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        backend="official",
        graph_enabled=True,
        classification=True,
        forecasting=True,
        learning_rate=1e-3,
        weight_decay=0.01,
        loss_weights=None,
        alert_quantile=0.999,
        schema_version=1,
    ):
        super().__init__()
        if schema_version != 1 or backend not in ("official", "reference"):
            raise ValueError("unsupported graph-Mamba checkpoint schema/backend")
        if min(width, layers, d_state, d_conv, expand) < 1 or not (
            classification or forecasting
        ):
            raise ValueError(
                "positive model dimensions and at least one detection objective required"
            )
        if not 0 < alert_quantile < 1:
            raise ValueError("alert_quantile must be in (0,1)")
        self.save_hyperparameters()
        self.input_proj = nn.Linear(23 + 4 + 2, width)
        self.temporal_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        if backend == "official":
            from graphids.runtime_checks import assert_mamba_backend

            assert_mamba_backend()
            from mamba_ssm import Mamba

            self.temporal = nn.ModuleList(
                Mamba(
                    d_model=width,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=i,
                )
                for i in range(layers)
            )
        else:
            from .mamba_reference import ReferenceMamba

            self.temporal = nn.ModuleList(
                ReferenceMamba(width, d_state, d_conv, expand) for _ in range(layers)
            )
        self.dropout = nn.Dropout(dropout)
        self.graph = GraphReadout(width)
        self.classifier = nn.Linear(width * 2, 1)
        self.count_head = nn.Linear(width, 1)
        self.change_head = nn.Linear(width, 1)
        self.new_id_head = nn.Linear(width * 2, 1)
        self.register_buffer("score_center", torch.zeros(3))
        self.register_buffer("score_scale", torch.ones(3))
        self.register_buffer("alert_thresholds", torch.full((2,), float("inf")))
        self.register_buffer("calibrated", torch.tensor(False))
        self._eval_states = {}
        self._validation_scores = []
        self._test_outputs = []
        self._train_coverage = []

    def prepare_from_datamodule(self, dm):
        if getattr(dm, "in_channels", None) != 23:
            raise ValueError(
                "vehicle_graph_mamba_joint requires the version-1 CAN snapshot datamodule"
            )
        self.data_manifest_digest = dm.manifest.digest
        self._test_set_names = tuple(dm.test_data)
        self._recordings = dm.records

    def _tokens(self, batch):
        b, t, n, _ = batch.x.shape
        profiles = batch.profiles[:, None].expand(b, t, n, 4)
        flags = torch.stack(
            (
                batch.in_reference[:, None].expand(b, t, n),
                batch.context_present[:, None, None].expand(b, t, n),
            ),
            -1,
        ).to(batch.x.dtype)
        return self.input_proj(torch.cat((batch.x, profiles, flags), -1))

    def encode(self, batch):
        tokens = self._tokens(batch)
        b, t, n, d = tokens.shape
        segments = {}
        # Loop over stream segments, not timesteps. Each group uses a full CUDA scan.
        for bi in range(b):
            for ni in range(n):
                valid = batch.live[bi, :, ni].detach().cpu().tolist()
                generations = batch.generation[bi, :, ni].detach().cpu().tolist()
                start = 0
                while start < t:
                    if not valid[start]:
                        start += 1
                        continue
                    stop = start + 1
                    while (
                        stop < t
                        and valid[stop]
                        and generations[stop] == generations[start]
                    ):
                        stop += 1
                    segments.setdefault(stop - start, []).append((bi, ni, start, stop))
                    start = stop
        h = torch.zeros_like(tokens)
        for _, group in segments.items():
            x = torch.stack(
                [tokens[bi, start:stop, ni] for bi, ni, start, stop in group]
            )
            for norm, block in zip(self.temporal_norms, self.temporal, strict=True):
                x = x + self.dropout(block(norm(x)))
            for i, (bi, ni, start, stop) in enumerate(group):
                h[bi, start:stop, ni] = x[i]
        return h

    def _heads(self, h, batch):
        h = self.graph(h, batch, graph_enabled=self.hparams.graph_enabled)
        live = batch.live.unsqueeze(-1)
        mean = (h * live).sum(2) / live.sum(2).clamp_min(1)
        maximum = (
            h.masked_fill(~live, torch.finfo(h.dtype).min).max(2).values
            if h.shape[2]
            else torch.zeros_like(mean)
        )
        maximum = torch.where(live.any(2), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum), -1)
        return {
            "classification_logits": self.classifier(pooled).squeeze(-1),
            "count_rate": F.softplus(self.count_head(h).squeeze(-1)) + 1e-6,
            "change_logits": self.change_head(h).squeeze(-1),
            "new_id_rate": F.softplus(self.new_id_head(pooled).squeeze(-1)) + 1e-6,
        }

    def forward(self, batch):
        return self._heads(self.encode(batch), batch)

    def _distributed_mean(self, values, mask):
        total = (values * mask).sum()
        count = mask.sum().to(total.dtype)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            # DDP averages gradients; scale each local numerator accordingly.
            total = total * torch.distributed.get_world_size()
        return total / count.clamp_min(1)

    @staticmethod
    def poisson_surprise(rate, target, exposure):
        mean = (rate * exposure).clamp_min(1e-8)
        return mean - target * mean.log() + torch.lgamma(target + 1)

    def losses(self, output, batch):
        classification = F.binary_cross_entropy_with_logits(
            output["classification_logits"],
            batch.labels.clamp_min(0).float(),
            reduction="none",
        )
        cls_mask = batch.scored & (batch.labels >= 0)
        values = {"classification": self._distributed_mean(classification, cls_mask)}
        # Forecasts are generated before targets enter the temporal encoder.
        valid = (
            batch.live[:, :-1]
            & batch.live[:, 1:]
            & (batch.generation[:, :-1] == batch.generation[:, 1:])
        )
        verified_history = batch.normal.long().cumprod(1).bool()
        eligible = verified_history[:, :-1] & batch.normal[:, 1:] & batch.scored[:, 1:]
        eligible &= batch.exposure[:, 1:] > 0
        mask = valid & eligible[:, :, None]
        count = self.poisson_surprise(
            output["count_rate"][:, :-1],
            batch.counts[:, 1:],
            batch.exposure[:, 1:, None],
        )
        change = F.binary_cross_entropy_with_logits(
            output["change_logits"][:, :-1],
            batch.payload_change[:, 1:],
            reduction="none",
        )
        new_id = self.poisson_surprise(
            output["new_id_rate"][:, :-1], batch.new_ids[:, 1:], batch.exposure[:, 1:]
        )
        values["count"] = self._distributed_mean(count, mask)
        values["change"] = self._distributed_mean(
            change, mask & batch.payload_observed[:, 1:]
        )
        values["new_id"] = self._distributed_mean(new_id, eligible)
        weights = self.hparams.loss_weights or {}
        loss = sum(
            value
            * float(weights.get(key, 1.0))
            * int(
                self.hparams.classification
                if key == "classification"
                else self.hparams.forecasting
            )
            for key, value in values.items()
        )
        # Every parameter participates even for dummy/empty rank schedules.
        loss = loss + sum(p.sum() * 0 for p in self.parameters())
        return loss, values

    def training_step(self, batch, batch_idx):
        loss, components = self.losses(self(batch), batch)
        self.log("train/loss", loss, sync_dist=True, batch_size=batch.x.shape[0])
        for name, value in components.items():
            self.log(
                f"train/{name}", value, sync_dist=True, batch_size=batch.x.shape[0]
            )
        self._train_coverage.append(
            {
                "recordings": batch.recording_keys,
                "datasets": batch.dataset_keys,
                "scored_bins": int(batch.scored.sum()),
                "scored_events": int((batch.event_count * batch.scored).sum()),
                "scored_seconds": float((batch.exposure * batch.scored).sum()),
                "node_bin_tokens": batch.live.numel(),
                "live_node_bins": int(batch.live.sum()),
                "graph_messages": int(batch.edge_mask.sum()) * batch.x.shape[1],
            }
        )
        return loss

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        dm.train_sampler.set_epoch(self.current_epoch)
        self._train_coverage = []

    def on_train_epoch_end(self):
        import json
        import os

        records = self._train_coverage
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, records)
            records = [r for part in gathered for r in part]
        if self.trainer.is_global_zero:
            root = (
                Path(os.environ.get("GRAPHIDS_RUN_DIR", self.trainer.default_root_dir))
                / "artifacts"
            )
            root.mkdir(parents=True, exist_ok=True)
            (root / f"training-work-{self.current_epoch}.json").write_text(
                json.dumps(
                    {
                        "epoch": self.current_epoch,
                        "optimizer_steps": self.global_step,
                        "sampled_episodes": records,
                        "source_manifest": self.data_manifest_digest,
                        "bin_seconds": self.trainer.datamodule.representation_cfg.bin_seconds,
                        "anchor_capacity": self.trainer.datamodule.anchor_capacity,
                        "transient_capacity": self.trainer.datamodule.transient_capacity,
                    },
                    indent=2,
                )
            )

    @torch.no_grad()
    def stream(self, batch, state=None):
        if batch.x.shape[0] != 1:
            raise ValueError("continuous streaming requires one recording per batch")
        if self.training:
            raise ValueError("streaming inference requires eval mode")
        state = state or GraphMambaState()
        key, digest = batch.recording_keys[0], batch.context_digests[0]
        if (state.recording, state.context_digest) != (key, digest):
            state = GraphMambaState(key, digest)
        if state.last_bin >= 0 and int(batch.bin_indices[0, 0]) != state.last_bin + 1:
            raise ValueError(
                "stream bins must be contiguous; reset explicitly at a new inference episode"
            )
        tokens = self._tokens(batch)
        t, n = batch.live.shape[1:]
        all_outputs = []
        surprises = []
        eligible_scores = []
        component_masks = []
        for ti in range(t):
            live_nodes = batch.live[0, ti].nonzero().flatten().tolist()
            active = {
                (batch.node_keys[0][ni], int(batch.generation[0, ti, ni]))
                for ni in live_nodes
            }
            state.caches = {k: v for k, v in state.caches.items() if k in active}
            h = tokens.new_zeros(1, 1, n, tokens.shape[-1])
            if live_nodes:
                x = tokens[0, ti, live_nodes].unsqueeze(1)
                caches = []
                for ni in live_nodes:
                    cache_key = (
                        batch.node_keys[0][ni],
                        int(batch.generation[0, ti, ni]),
                    )
                    if cache_key not in state.caches:
                        state.caches[cache_key] = [
                            block.allocate_inference_cache(1, 1)
                            for block in self.temporal
                        ]
                    caches.append(state.caches[cache_key])
                for layer, (norm, block) in enumerate(
                    zip(self.temporal_norms, self.temporal, strict=True)
                ):
                    conv = torch.cat([c[layer][0] for c in caches])
                    ssm = torch.cat([c[layer][1] for c in caches])
                    y, conv, ssm = block.step(norm(x), conv, ssm)
                    x = x + y
                    for ci, cache in enumerate(caches):
                        cache[layer] = (
                            conv[ci : ci + 1].clone(),
                            ssm[ci : ci + 1].clone(),
                        )
                h[0, 0, live_nodes] = x[:, 0]
            single = self._slice_bin(batch, ti)
            output = self._heads(h, single)
            all_outputs.append(output)
            terms = tokens.new_zeros(3)
            component_mask = torch.zeros(3, dtype=torch.bool, device=tokens.device)
            forecast_eligible = False
            if state.pending and float(batch.exposure[0, ti]) > 0:
                prior = state.pending
                matched = [
                    (ni, prior["nodes"][node])
                    for ni in live_nodes
                    if (
                        node := (
                            batch.node_keys[0][ni],
                            int(batch.generation[0, ti, ni]),
                        )
                    )
                    in prior["nodes"]
                ]
                if matched:
                    component_mask[0] = True
                    rates = torch.stack([pred[0] for ni, pred in matched])
                    targets = torch.stack(
                        [batch.counts[0, ti, ni] for ni, pred in matched]
                    )
                    terms[0] = self.poisson_surprise(
                        rates, targets, batch.exposure[0, ti]
                    ).max()
                    payload = [
                        (ni, pred)
                        for ni, pred in matched
                        if batch.payload_observed[0, ti, ni]
                    ]
                    if payload:
                        component_mask[1] = True
                        terms[1] = torch.stack(
                            [
                                F.binary_cross_entropy_with_logits(
                                    pred[1],
                                    batch.payload_change[0, ti, ni],
                                    reduction="none",
                                )
                                for ni, pred in payload
                            ]
                        ).max()
                terms[2] = self.poisson_surprise(
                    prior["new_id"], batch.new_ids[0, ti], batch.exposure[0, ti]
                )
                forecast_eligible = True
                component_mask[2] = True
            surprises.append(terms)
            component_masks.append(component_mask)
            eligible_scores.append(forecast_eligible)
            state.pending = {
                "nodes": {
                    (batch.node_keys[0][ni], int(batch.generation[0, ti, ni])): (
                        output["count_rate"][0, 0, ni],
                        output["change_logits"][0, 0, ni],
                    )
                    for ni in live_nodes
                },
                "new_id": output["new_id_rate"][0, 0],
            }
        outputs = {
            name: torch.cat([o[name] for o in all_outputs], 1)
            for name in all_outputs[0]
        }
        outputs["anomaly_components"] = torch.stack(surprises).unsqueeze(0)
        outputs["anomaly_component_mask"] = torch.stack(component_masks).unsqueeze(0)
        outputs["anomaly_eligible"] = torch.tensor(
            eligible_scores, device=tokens.device
        ).unsqueeze(0)
        outputs["anomaly_score"] = (
            ((outputs["anomaly_components"] - self.score_center) / self.score_scale)
            * outputs["anomaly_component_mask"]
        ).sum(-1)
        state.last_bin = int(batch.bin_indices[0, -1])
        return outputs, state

    @staticmethod
    def _slice_bin(batch, t):
        from dataclasses import fields

        static = {
            "profiles",
            "in_reference",
            "context_present",
            "neighbors",
            "edge_features",
            "edge_mask",
        }
        return CANSnapshotBatch(
            **{
                f.name: (
                    getattr(batch, f.name)[:, t : t + 1]
                    if isinstance(getattr(batch, f.name), torch.Tensor)
                    and f.name not in static
                    else getattr(batch, f.name)
                )
                for f in fields(batch)
            }
        )

    def _evaluate(self, batch, loader_idx, stage):
        if not batch.scored.any():
            return  # distributed schedule placeholder, no state or coverage
        state = self._eval_states.get(loader_idx)
        output, state = self.stream(batch, state)
        self._eval_states[loader_idx] = state
        probability = torch.sigmoid(output["classification_logits"])
        mask = batch.scored
        for t in mask[0].nonzero().flatten().tolist():
            row = {
                "recording": batch.recording_keys[0],
                "dataset": batch.dataset_keys[0],
                "vehicle": self._recordings[batch.recording_keys[0]].vehicle,
                "role": self._recordings[batch.recording_keys[0]].role,
                "context": batch.context_digests[0],
                "bin": int(batch.bin_indices[0, t]),
                "close": float(batch.closes[0, t]),
                "exposure": float(batch.exposure[0, t]),
                "classification": float(probability[0, t]),
                "anomaly": float(output["anomaly_score"][0, t]),
                "anomaly_components": output["anomaly_components"][0, t].cpu().tolist(),
                "anomaly_component_mask": output["anomaly_component_mask"][0, t]
                .cpu()
                .tolist(),
                "anomaly_eligible": bool(output["anomaly_eligible"][0, t]),
                "label": int(batch.labels[0, t]),
                "normal": bool(batch.normal[0, t]),
                "events": int(batch.event_count[0, t]),
                "live_nodes": int(batch.live[0, t].sum()),
                "silent_anchors": int(
                    (
                        batch.in_reference[0]
                        & batch.live[0, t]
                        & (batch.counts[0, t] == 0)
                    ).sum()
                ),
                "unavailable_payload_fields": int(
                    (batch.live[0, t, :, None] & (batch.x[0, t, :, 12:20] == 0)).sum()
                ),
                "alert": bool(
                    (
                        self.hparams.classification
                        and probability[0, t] >= self.alert_thresholds[0]
                    )
                    or (
                        self.hparams.forecasting
                        and output["anomaly_eligible"][0, t]
                        and output["anomaly_score"][0, t] >= self.alert_thresholds[1]
                    )
                ),
            }
            if stage == "val":
                self._validation_scores.append(row)
            else:
                self._test_outputs.append(row)

    def on_validation_epoch_start(self):
        self._eval_states = {}
        self._validation_scores = []

    def validation_step(self, batch, batch_idx):
        # Source modes/captures have distinct state keys.
        self._evaluate(batch, 0, "val")

    def on_validation_epoch_end(self):
        rows = self._validation_scores
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        unique = {(r["recording"], r["context"], r["bin"]): r for r in rows}
        if len(unique) != len(rows):
            raise ValueError("duplicate scored validation bins across workers")
        rows = list(unique.values())
        normal = [r for r in rows if r["normal"] and r["anomaly_eligible"]]
        if normal:
            components = torch.tensor(
                [r["anomaly_components"] for r in normal], device=self.device
            )
            masks = torch.tensor(
                [r["anomaly_component_mask"] for r in normal], device=self.device
            )
            for component in range(3):
                valid_values = components[masks[:, component], component]
                if valid_values.numel():
                    self.score_center[component] = valid_values.mean()
                    self.score_scale[component] = valid_values.std(
                        unbiased=False
                    ).clamp_min(1e-4)
            anomaly = (
                ((components - self.score_center) / self.score_scale) * masks
            ).sum(-1)
            cls = torch.tensor(
                [r["classification"] for r in normal], device=self.device
            )
            self.alert_thresholds.copy_(
                torch.stack(
                    (
                        torch.quantile(cls, self.hparams.alert_quantile),
                        torch.quantile(anomaly, self.hparams.alert_quantile),
                    )
                )
            )
            self.calibrated.fill_(True)
            for row in rows:
                component = self.score_center.new_tensor(row["anomaly_components"])
                row["anomaly"] = float(
                    (
                        ((component - self.score_center) / self.score_scale)
                        * component.new_tensor(row["anomaly_component_mask"])
                    ).sum()
                )
                row["alert"] = bool(
                    (
                        self.hparams.classification
                        and row["classification"] >= self.alert_thresholds[0]
                    )
                    or (
                        self.hparams.forecasting
                        and row["anomaly_eligible"]
                        and row["anomaly"] >= self.alert_thresholds[1]
                    )
                )
        self._log_metrics(rows, "val")

    def on_test_epoch_start(self):
        self._eval_states = {}
        self._test_outputs = []
        if not bool(self.calibrated):
            raise ValueError(
                "test requires a checkpoint calibrated on verified-normal source validation"
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._evaluate(batch, dataloader_idx, "test")

    def on_test_epoch_end(self):
        rows = self._test_outputs
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        keys = [(r["recording"], r["context"], r["bin"]) for r in rows]
        expected = set()
        dm = self.trainer.datamodule
        for dataset in dm.test_data.values():
            for key, start, stop, warmup, mode in dataset.entries:
                context = dm.context_for(key) if mode == "conditioned" else None
                digest = context.digest if context else "zero-shot"
                expected.update((key, digest, i) for i in range(start + warmup, stop))
        if set(keys) != expected:
            raise ValueError(
                f"incomplete test coverage: {len(expected - set(keys))} missing, {len(set(keys) - expected)} unexpected bins"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(
                "duplicate scored bins across data-parallel evaluation workers"
            )
        self._log_metrics(rows, "test")
        if self.trainer.is_global_zero:
            import json
            import os

            root = (
                Path(os.environ.get("GRAPHIDS_RUN_DIR", self.trainer.default_root_dir))
                / "artifacts"
            )
            root.mkdir(parents=True, exist_ok=True)
            (root / "bin-scores.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "coverage.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "expected_bins": len(expected),
                        "scored_bins": len(keys),
                        "duplicate_bins": 0,
                        "missing_bins": 0,
                        "scored_event_observations": sum(r["events"] for r in rows),
                        "classification_bins": sum(r["label"] >= 0 for r in rows),
                        "anomaly_bins": sum(r["anomaly_eligible"] for r in rows),
                        "terminal_zero_exposure_bins": sum(
                            r["exposure"] == 0 for r in rows
                        ),
                        "manifest_digest": dm.manifest.digest,
                    },
                    indent=2,
                )
            )

    def _log_metrics(self, rows, prefix):
        from torchmetrics.functional.classification import binary_average_precision

        macro = {}
        datasets = sorted({r["dataset"] for r in rows})
        for dataset in datasets:
            selected = [
                r
                for r in rows
                if r["dataset"] == dataset and r["role"] != "target_calibration"
            ]
            for mode in ("zero-shot", "conditioned"):
                group = [
                    r
                    for r in selected
                    if (r["context"] == "zero-shot") == (mode == "zero-shot")
                ]
                normal = [r for r in group if r["normal"]]
                hours = sum(r["exposure"] for r in normal) / 3600
                base = f"{prefix}/{dataset}/{mode}"
                self.log(f"{base}/scored_bins", float(len(group)), sync_dist=False)
                self.log(f"{base}/normal_hours", hours, sync_dist=False)
                if hours:
                    # Initial alert policy emits one candidate alert per closed bin.
                    self.log(
                        f"{base}/false_alerts_per_hour",
                        sum(r["alert"] for r in normal) / hours,
                        sync_dist=False,
                    )
                for score in ("classification", "anomaly"):
                    if not (
                        self.hparams.classification
                        if score == "classification"
                        else self.hparams.forecasting
                    ):
                        continue
                    eligible = [
                        r
                        for r in group
                        if r["label"] >= 0
                        and (score != "anomaly" or r["anomaly_eligible"])
                    ]
                    if {r["label"] for r in eligible} == {0, 1}:
                        p = torch.tensor(
                            [r[score] for r in eligible], device=self.device
                        )
                        y = torch.tensor(
                            [r["label"] for r in eligible], device=self.device
                        )
                        ap = binary_average_precision(p, y)
                        macro.setdefault((mode, score), []).append(ap)
                        self.log(
                            f"{base}/{score}_ap",
                            ap,
                            sync_dist=False,
                        )
                        for vehicle in sorted({r["vehicle"] for r in eligible}):
                            subset = [r for r in eligible if r["vehicle"] == vehicle]
                            if {r["label"] for r in subset} == {0, 1}:
                                vp = torch.tensor(
                                    [r[score] for r in subset], device=self.device
                                )
                                vy = torch.tensor(
                                    [r["label"] for r in subset], device=self.device
                                )
                                self.log(
                                    f"{base}/{vehicle}/{score}_ap",
                                    binary_average_precision(vp, vy),
                                    sync_dist=False,
                                )
        for (mode, score), values in macro.items():
            stack = torch.stack(values)
            self.log(f"{prefix}/{mode}/{score}_ap_macro", stack.mean(), sync_dist=False)
            self.log(
                f"{prefix}/{mode}/{score}_ap_worst_dataset",
                stack.min(),
                sync_dist=False,
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["graphids_schema"] = {
            "model": "vehicle_graph_mamba_joint",
            "version": 1,
            "data_manifest": getattr(self, "data_manifest_digest", None),
            "backend": self.hparams.backend,
        }

    def on_load_checkpoint(self, checkpoint):
        schema = checkpoint.get("graphids_schema", {})
        if (
            schema.get("model") != "vehicle_graph_mamba_joint"
            or schema.get("version") != 1
        ):
            raise ValueError(
                "legacy/incompatible checkpoint: use the graphids-temporal-legacy revision to evaluate old weights"
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
