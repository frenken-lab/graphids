"""Shared causal Mamba temporal encoding and bounded same-time graph readout."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from graphids.core.data.snapshots.batch import CANSnapshotBatch
from graphids.core.data.snapshots.execution import execution_plan
from graphids.core.data.snapshots.policy import ReferencePolicy

from .calibration import CalibrationBundle, weights_digest
from .mamba_chunk import mamba_chunk
from .score_columns import ScoreBlock, ScoreTable


@dataclass
class GraphMambaState:
    recording: str = ""
    context_digest: str = ""
    caches: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)
    scratch: list = field(default_factory=list)
    version: int = 2
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
        stream_execution="chunked",
        graph_enabled=True,
        classification=True,
        forecasting=True,
        learning_rate=1e-3,
        weight_decay=0.01,
        loss_weights=None,
        alert_quantile=0.999,
        schema_version=1,
        reference_policy=None,
        legacy_calibration_replay=False,
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
        if stream_execution not in ("chunked", "recurrent"):
            raise ValueError("stream_execution must be chunked or recurrent")
        self.save_hyperparameters()
        self.reference_policy = ReferencePolicy.parse(reference_policy)
        self.calibration_bundles = {}
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
        self.phase_timings = {}

    def prepare_from_datamodule(self, dm):
        if getattr(dm, "in_channels", None) != 23:
            raise ValueError(
                "vehicle_graph_mamba_joint requires the version-1 CAN snapshot datamodule"
            )
        if self.reference_policy != dm.reference_policy:
            raise ValueError("model and datamodule reference policies differ")
        import hashlib
        import json
        from dataclasses import asdict

        self.source_manifest_digest = hashlib.sha256(
            json.dumps(
                [
                    asdict(r)
                    for r in dm.manifest.records
                    if r.role.startswith(("train_", "val_"))
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
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
        if not self.reference_policy.profiles:
            profiles = torch.zeros_like(profiles)
        if not self.reference_policy.reference_flags:
            flags = torch.zeros_like(flags)
        return self.input_proj(torch.cat((batch.x, profiles, flags), -1))

    def encode(self, batch):
        tokens = self._tokens(batch)
        flat = tokens.flatten(0, 2)
        h = torch.zeros_like(flat)
        for group in execution_plan(batch).groups:
            indices = group.indices.to(tokens.device, non_blocking=True)
            x = flat[indices]
            for norm, block in zip(self.temporal_norms, self.temporal, strict=True):
                x = x + self.dropout(block(norm(x)))
            h = h.index_copy(0, indices.flatten(), x.flatten(0, 1))
        return h.reshape_as(tokens)

    def _heads(self, h, batch):
        h = self.graph(
            h,
            batch,
            graph_enabled=self.hparams.graph_enabled
            and self.reference_policy.graph_edges,
        )
        live = batch.live.unsqueeze(-1)
        mean = (h * live).sum(2) / live.sum(2).clamp_min(1)
        maximum = (
            h.masked_fill(~live, torch.finfo(h.dtype).min).max(2).values
            if h.shape[2]
            else torch.zeros_like(mean)
        )
        maximum = torch.where(live.any(2), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum), -1)
        classifier_pool = pooled
        if self.reference_policy.recent_classifier_pooling:
            recent = live & (
                (batch.counts > 0) | ~batch.in_reference[:, None]
            ).unsqueeze(-1)
            recent_mean = (h * recent).sum(2) / recent.sum(2).clamp_min(1)
            recent_max = h.masked_fill(~recent, torch.finfo(h.dtype).min).max(2).values
            recent_max = torch.where(
                recent.any(2), recent_max, torch.zeros_like(recent_max)
            )
            classifier_pool = torch.cat((recent_mean, recent_max), -1)
        return {
            "classification_logits": self.classifier(classifier_pool).squeeze(-1),
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
        if state.version != 2:
            raise ValueError(
                "version-1 live state is incompatible; reset the stream state"
            )
        key, digest = (
            batch.recording_keys[0],
            batch.context_digests[0] + ":" + self.reference_policy.digest,
        )
        if (state.recording, state.context_digest) != (key, digest):
            state = GraphMambaState(key, digest)
        plan = execution_plan(batch)
        bins = plan.bins[0]
        if (state.last_bin >= 0 and int(bins[0]) != state.last_bin + 1) or (
            len(bins) > 1 and not ((bins[1:] - bins[:-1]) == 1).all()
        ):
            raise ValueError(
                "stream bins must be contiguous; reset explicitly at a new inference episode"
            )
        tokens = self._tokens(batch)
        _, t, n, _ = tokens.shape
        final_nodes = [int(i) for i in plan.live[0, -1].nonzero()[0]]
        final_keys = [
            (batch.node_keys[0][ni], int(plan.generation[0, -1, ni]))
            for ni in final_nodes
        ]
        next_slots = {identity: slot for slot, identity in enumerate(final_keys)}
        max_live = int(plan.live.sum(-1).max())
        capacity = batch.state_capacity or max(1, max_live, len(state.caches))
        if max(len(state.caches), len(next_slots), max_live) > capacity:
            raise ValueError(
                "live state capacity exceeded; observations were not truncated"
            )
        if not state.layers or state.layers[0][0].shape[0] < capacity:
            previous = state.layers
            state.layers = [
                block.allocate_inference_cache(capacity, 1) for block in self.temporal
            ]
            for old, new in zip(previous, state.layers):
                for src, dst in zip(old, new):
                    dst[: src.shape[0]].copy_(src)
            state.scratch = [
                block.allocate_inference_cache(capacity, 1) for block in self.temporal
            ]
        flat = tokens.flatten(0, 2)
        h = torch.zeros_like(flat)
        for group in plan.groups:
            indices = group.indices.to(tokens.device, non_blocking=True)
            x = flat[indices]
            carried, old_slots, finishing, slots = [], [], [], []
            for gi, (_, ni, start, stop, generation) in enumerate(group.segments):
                identity = (batch.node_keys[0][ni], generation)
                if start == 0 and identity in state.caches:
                    carried.append(gi)
                    old_slots.append(state.caches[identity])
                if stop == t:
                    finishing.append(gi)
                    slots.append(next_slots[identity])

            def index(values):
                return torch.tensor(values, device=tokens.device, dtype=torch.long)

            carried, old_slots, finishing, slots = map(
                index, (carried, old_slots, finishing, slots)
            )
            for layer, (norm, block) in enumerate(
                zip(self.temporal_norms, self.temporal, strict=True)
            ):
                conv, ssm = block.allocate_inference_cache(
                    len(group.segments), x.shape[1]
                )
                conv.index_copy_(
                    0, carried, state.layers[layer][0].index_select(0, old_slots)
                )
                ssm.index_copy_(
                    0, carried, state.layers[layer][1].index_select(0, old_slots)
                )
                normalized = norm(x)
                if self.hparams.stream_execution == "chunked":
                    y, conv, ssm = mamba_chunk(
                        block,
                        normalized,
                        conv,
                        ssm,
                        official=self.hparams.backend == "official",
                    )
                else:
                    outputs = []
                    for ti in range(x.shape[1]):
                        y, conv, ssm = block.step(normalized[:, ti : ti + 1], conv, ssm)
                        outputs.append(y)
                    y = torch.cat(outputs, 1)
                x = x + y
                state.scratch[layer][0].index_copy_(
                    0, slots, conv.index_select(0, finishing)
                )
                state.scratch[layer][1].index_copy_(
                    0, slots, ssm.index_select(0, finishing)
                )
            h.index_copy_(0, indices.flatten(), x.flatten(0, 1))
        output = self._heads(h.reshape_as(tokens), batch)
        self._forecast_scores(output, batch, state, plan, final_nodes, final_keys)
        state.layers, state.scratch = state.scratch, state.layers
        state.caches = next_slots
        state.last_bin = int(bins[-1])
        return output, state

    def _forecast_scores(self, output, batch, state, plan, final_nodes, final_keys):
        """Match prior forecasts to current identities, including the chunk edge."""
        b, t, n = batch.live.shape
        rates = torch.zeros_like(output["count_rate"])
        changes = torch.zeros_like(output["change_logits"])
        matched = torch.zeros_like(batch.live)
        rates[:, 1:] = output["count_rate"][:, :-1]
        changes[:, 1:] = output["change_logits"][:, :-1]
        matched[:, 1:] = (
            batch.live[:, 1:]
            & batch.live[:, :-1]
            & (batch.generation[:, 1:] == batch.generation[:, :-1])
        )
        prior_new = torch.zeros_like(output["new_id_rate"])
        prior_new[:, 1:] = output["new_id_rate"][:, :-1]
        eligible = batch.exposure > 0
        eligible[:, 0] &= bool(state.pending)
        if state.pending:
            prior_new[:, 0] = state.pending["new_id"]
            prior_map = state.pending["identities"]
            current, previous = [], []
            for ni in plan.live[0, 0].nonzero()[0]:
                identity = (batch.node_keys[0][ni], int(plan.generation[0, 0, ni]))
                if identity in prior_map:
                    current.append(int(ni))
                    previous.append(prior_map[identity])
            ci = torch.tensor(current, device=batch.x.device, dtype=torch.long)
            pi = torch.tensor(previous, device=batch.x.device, dtype=torch.long)
            rates[0, 0, ci] = state.pending["count"].index_select(0, pi)
            changes[0, 0, ci] = state.pending["change"].index_select(0, pi)
            matched[0, 0, ci] = True
        matched &= eligible[..., None]
        payload = matched & batch.payload_observed
        count = self.poisson_surprise(
            rates, batch.counts, batch.exposure[..., None].to(rates.dtype)
        )
        change = F.binary_cross_entropy_with_logits(
            changes, batch.payload_change, reduction="none"
        )

        def maximum(values, mask):
            # Support recordings/chunks with no live IDs.
            if n == 0:
                return values.new_zeros(b, t)
            return (
                values.masked_fill(~mask, -torch.inf)
                .max(-1)
                .values.masked_fill(~mask.any(-1), 0)
            )

        components = torch.stack(
            (
                maximum(count, matched),
                maximum(change, payload),
                self.poisson_surprise(
                    prior_new, batch.new_ids, batch.exposure
                ).masked_fill(~eligible, 0),
            ),
            -1,
        )
        components = components.to(output["count_rate"].dtype)
        masks = torch.stack((matched.any(-1), payload.any(-1), eligible), -1)
        mode = "conditioned" if bool(batch.context_present[0]) else "zero-shot"
        bundle = self.calibration_bundles.get(mode)
        center, scale = self.score_center, self.score_scale
        if bundle:
            center = components.new_tensor(bundle.center)
            scale = components.new_tensor(bundle.scale)

        output.update(
            anomaly_components=components,
            anomaly_component_mask=masks,
            anomaly_eligible=eligible,
            anomaly_score=(
                ((components - center) / scale)
                * masks
                * (components.new_tensor(bundle.available) if bundle else 1)
            ).sum(-1),
        )
        nodes = torch.tensor(final_nodes, device=batch.x.device, dtype=torch.long)
        state.pending = {
            "identities": {identity: i for i, identity in enumerate(final_keys)},
            "count": output["count_rate"][0, -1].index_select(0, nodes).clone(),
            "change": output["change_logits"][0, -1].index_select(0, nodes).clone(),
            "new_id": output["new_id_rate"][:, -1].clone(),
        }

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
                    else None
                    if f.name == "execution"
                    else getattr(batch, f.name)
                )
                for f in fields(batch)
            }
        )

    def _evaluate(self, batch, loader_idx, stage):
        if not execution_plan(batch).scored.any():
            return  # distributed placeholder never advances state
        with self._timed(f"{stage}/inference_and_collection"):
            output, state = self.stream(batch, self._eval_states.get(loader_idx))
            self._eval_states[loader_idx] = state
            block = ScoreBlock.from_batch(self, batch, output)
        (self._validation_scores if stage == "val" else self._test_outputs).append(
            block
        )

    @contextmanager
    def _timed(self, name):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            entry = self.phase_timings.setdefault(
                name, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0}
            )
            entry["wall_seconds"] += time.perf_counter() - wall
            entry["cpu_seconds"] += time.process_time() - cpu
            entry["calls"] += 1

    def _gather_scores(self, blocks):
        if not torch.distributed.is_initialized():
            return ScoreTable(blocks)
        rank = torch.distributed.get_rank()
        gathered = [None] * torch.distributed.get_world_size() if rank == 0 else None
        torch.distributed.gather_object(blocks, gathered, dst=0)
        return (
            ScoreTable([block for part in gathered for block in part])
            if rank == 0
            else None
        )

    def on_validation_epoch_start(self):
        self._eval_states = {}
        self._validation_scores = []

    def validation_step(self, batch, batch_idx):
        # Source modes/captures have distinct state keys.
        self._evaluate(batch, 0, "val")

    def on_validation_epoch_end(self):
        with self._timed("val/gather"):
            table = self._gather_scores(self._validation_scores)
        metrics = {}
        if table is not None:
            table.check_unique("validation")
            with self._timed("val/calibration"):
                table.calibrate(self)
            with self._timed("val/metrics"):
                metrics = table.log_metrics(self, "val", emit_to_model=False)
        if torch.distributed.is_initialized():
            objects = [
                {k: v.to_dict() for k, v in self.calibration_bundles.items()},
                metrics,
            ]
            torch.distributed.broadcast_object_list(objects, src=0)
            self.calibration_bundles = {
                k: CalibrationBundle(**v) for k, v in objects[0].items()
            }
            metrics = objects[1]
            for buffer in (
                self.score_center,
                self.score_scale,
                self.alert_thresholds,
                self.calibrated,
            ):
                torch.distributed.broadcast(buffer, src=0)
        # Checkpoint callbacks run on every rank; each needs the same global metric.
        for name, value in metrics.items():
            self.log(name, value, sync_dist=False)

    def on_test_epoch_start(self):
        self._eval_states = {}
        self._test_outputs = []
        if not self.calibration_bundles and not self.hparams.legacy_calibration_replay:
            raise ValueError(
                "legacy pooled calibration requires explicit legacy_calibration_replay or fresh source validation"
            )
        if self.calibration_bundles and set(self.calibration_bundles) != {
            "zero-shot",
            "conditioned",
        }:
            raise ValueError("missing calibration mode; both modes are required")
        for mode, bundle in self.calibration_bundles.items():
            bundle.validate(
                weights_digest(self),
                self.source_manifest_digest,
                self.reference_policy.digest,
                mode,
            )
        if not bool(self.calibrated):
            raise ValueError(
                "test requires a checkpoint calibrated on verified-normal source validation"
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._evaluate(batch, dataloader_idx, "test")

    def on_test_epoch_end(self):
        with self._timed("test/gather"):
            rows = self._gather_scores(self._test_outputs)
        if rows is None:
            return
        keys = rows.keys()
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
        if self.calibration_bundles:
            rows.apply_bundles(
                self.calibration_bundles,
                classification=self.hparams.classification,
                forecasting=self.hparams.forecasting,
            )
        with self._timed("test/metrics"):
            self._log_metrics(rows, "test")
        if self.trainer.is_global_zero:
            import json
            import os

            root = (
                Path(os.environ.get("GRAPHIDS_RUN_DIR", self.trainer.default_root_dir))
                / "artifacts"
            )
            root.mkdir(parents=True, exist_ok=True)
            (root / "calibration-v2.json").write_text(
                json.dumps(
                    {k: v.to_dict() for k, v in self.calibration_bundles.items()},
                    indent=2,
                )
            )
            report = rows.report(
                self.calibration_bundles,
                classification=self.hparams.classification,
                forecasting=self.hparams.forecasting,
            )
            report["calibration_track"] = (
                "source-normal" if self.calibration_bundles else "legacy-pooled-replay"
            )
            (root / "metrics-v2.json").write_text(json.dumps(report, indent=2))
            if "conditioned" in self.calibration_bundles:
                target_bundle = CalibrationBundle.fit(
                    rows,
                    checkpoint_hash=self.calibration_bundles[
                        "conditioned"
                    ].checkpoint_hash,
                    source_manifest=self.source_manifest_digest,
                    reference_policy=self.reference_policy.digest,
                    mode="conditioned",
                    source=self.calibration_bundles["conditioned"],
                    quantile=self.hparams.alert_quantile,
                )
                tracks = dict(self.calibration_bundles, conditioned=target_bundle)
                (root / "target-calibration-v2.json").write_text(
                    json.dumps(target_bundle.to_dict(), indent=2)
                )
                (root / "target-calibrated-metrics-v2.json").write_text(
                    json.dumps(
                        rows.report(
                            tracks,
                            classification=self.hparams.classification,
                            forecasting=self.hparams.forecasting,
                        ),
                        indent=2,
                    )
                )
            with self._timed("test/artifacts"):
                with (root / "bin-scores.jsonl").open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                (root / "coverage.json").write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "expected_bins": len(expected),
                            "scored_bins": len(keys),
                            "duplicate_bins": 0,
                            "missing_bins": 0,
                            "scored_event_observations": int(
                                rows.column("events").sum()
                            ),
                            "classification_bins": int(
                                (rows.column("label") >= 0).sum()
                            ),
                            "anomaly_bins": int(rows.column("anomaly_eligible").sum()),
                            "terminal_zero_exposure_bins": int(
                                (rows.column("exposure") == 0).sum()
                            ),
                            "manifest_digest": dm.manifest.digest,
                        },
                        indent=2,
                    )
                )

    def _log_metrics(self, table, prefix):
        table.log_metrics(self, prefix)

    def on_save_checkpoint(self, checkpoint):
        from dataclasses import asdict

        checkpoint.setdefault("hyper_parameters", {})["reference_policy"] = asdict(
            self.reference_policy
        )
        checkpoint["calibration_v2"] = {
            k: v.to_dict() for k, v in self.calibration_bundles.items()
        }
        checkpoint["graphids_schema"] = {
            "model": "vehicle_graph_mamba_joint",
            "version": 1,
            "data_manifest": getattr(self, "data_manifest_digest", None),
            "backend": self.hparams.backend,
        }

    def on_load_checkpoint(self, checkpoint):
        self.calibration_bundles = {
            k: CalibrationBundle(**v)
            for k, v in checkpoint.get("calibration_v2", {}).items()
        }
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
