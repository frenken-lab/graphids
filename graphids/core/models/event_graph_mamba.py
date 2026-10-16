"""Message-resolution Mamba and learned causal CAN-ID graph readout.

Every observed frame enters Mamba before any learned compression. Graph keys are
latest *preceding* per-ID temporal states, with learned attention rather than
count/change correlation edges. This is a separate, incompatible checkpoint type.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graphids import event_defaults as defaults
from graphids.core.metrics import raw_average_precision

from .mamba_chunk import mamba_chunk

_GRAPH_WORKSPACE_BYTES = 256 * 1024**2


@dataclass
class EventGraphMambaState:
    recording: str = ""
    source_digest: str = ""
    next_row: int = 0
    last_timestamp: float = -math.inf
    caches: list = field(default_factory=list)
    ids: tuple[int, ...] = ()
    nodes: torch.Tensor | None = None
    previous_bus: torch.Tensor | None = None


class EventGraphMamba(pl.LightningModule):
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
        learning_rate=1e-3,
        weight_decay=0.01,
        payload_weight=1.0,
        node_capacity=512,
        max_graph_tokens=defaults.MAX_GRAPH_TOKENS,
        schema_version=1,
    ):
        super().__init__()
        if backend not in ("official", "reference") or schema_version != 1:
            raise ValueError("unsupported event Graph-Mamba backend/schema")
        if (
            min(width, layers, d_state, d_conv, expand, node_capacity, max_graph_tokens)
            < 1
        ):
            raise ValueError("positive event model dimensions/capacities required")
        if (
            not 0 <= dropout < 1
            or not math.isfinite(payload_weight)
            or payload_weight < 0
        ):
            raise ValueError("invalid event dropout/payload loss weight")
        self.save_hyperparameters()
        # Exact frame fields encoded numerically: bytes, byte-presence masks,
        # all 29 ID bits, byte length, elapsed time, preceding interarrival time.
        self.input_proj = nn.Linear(48, width)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
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
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.graph_gate = nn.Linear(width * 2, width)
        self.graph_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 1)
        # Header/current arrival are known; current payload is absent from every
        # path feeding this forecast. Predictions precede the frame-state update.
        self.payload_head = nn.Sequential(
            nn.Linear(width * 2 + 32, width), nn.SiLU(), nn.Linear(width, 8 * 256)
        )
        self.automatic_optimization = False
        self._training_states = {}
        self._epoch_complete = True
        self._stream_state = None
        self._score_parts = []
        self._score_index = []
        self._source_manifest_digest = None

    def prepare_from_datamodule(self, dm):
        if getattr(dm.representation_cfg, "kind", None) != "can_event":
            raise ValueError("event_graph_mamba requires can_event_dm")
        digest = dm.manifest.digest
        if (
            self._source_manifest_digest is not None
            and self._source_manifest_digest != digest
        ):
            raise ValueError(
                "event checkpoint source manifest differs from current data"
            )
        self._source_manifest_digest = digest
        self._recordings = dm.records

    def fields(self, batch):
        bits = (
            (batch.ids[:, None] >> torch.arange(29, device=batch.ids.device)) & 1
        ).float()
        mask = torch.arange(8, device=batch.ids.device)[None] < batch.lengths[:, None]
        # Scaling is invertible for every byte and length; no message aggregates.
        header = torch.cat(
            (bits, batch.lengths[:, None].float() / 8, batch.times.float()), -1
        )
        return torch.cat(
            (batch.payload.float() / 255, mask.float(), header), -1
        ), header

    def _routing(self, batch, state):
        prepared = batch.routing
        if prepared is not None:
            if (
                prepared.recording,
                prepared.source_digest,
                prepared.row_start,
                prepared.event_ids,
                prepared.incoming_ids,
            ) != (
                batch.recording,
                batch.source_digest,
                batch.row_start,
                batch.routing_ids,
                state.ids,
            ):
                raise ValueError("prepared routing provenance/history mismatch")
            if len(prepared.ids) > self.hparams.node_capacity:
                raise ValueError(
                    "event node capacity exceeded; no IDs were silently evicted"
                )
            if len(prepared.ids) * len(batch.ids) > self.hparams.max_graph_tokens:
                raise ValueError(
                    "event graph capacity exceeded; use smaller event chunks"
                )
            return prepared.ids, prepared.previous, prepared.own, prepared.final
        # Check capacity before the shared CPU helper allocates dense indices.
        from graphids.core.data.events.data import prepare_routing

        ids = tuple(dict.fromkeys((*state.ids, *batch.routing_ids)))
        if len(ids) > self.hparams.node_capacity:
            raise ValueError(
                "event node capacity exceeded; no IDs were silently evicted"
            )
        if len(ids) * len(batch.ids) > self.hparams.max_graph_tokens:
            raise ValueError("event graph capacity exceeded; use smaller event chunks")
        route = prepare_routing(
            batch.recording,
            batch.source_digest,
            batch.row_start,
            batch.routing_ids,
            state.ids,
        )
        return route.ids, route.previous, route.own, route.final

    @staticmethod
    def _gather_rows(bank, index):
        # index_select uses a dedicated index-add backward rather than the
        # generic multidimensional indexing gradient kernel (profiled bottleneck).
        return bank.index_select(0, index.reshape(-1)).reshape(
            *index.shape, bank.shape[-1]
        )

    def _readout_batched(self, h, header, batches, states, routes):
        """Pack banks with disjoint offsets; tile attention under 256 MiB."""
        width = h.shape[-1]
        banks, previous, own, final, offsets = [], [], [], [], []
        offset = 0
        max_ids = max(len(r[0]) for r in routes)
        for i, (state, route) in enumerate(zip(states, routes, strict=True)):
            ids, prev, local_own, local_final = route
            past = state.nodes if state.nodes is not None else h.new_empty((0, width))
            bank = torch.cat((past, h[i]), 0)
            banks.append(bank)
            offsets.append(offset)
            prev = torch.as_tensor(prev, device=h.device)
            # Sentinel -1 remains invalid after packing and padding.
            prev = torch.where(prev >= 0, prev + offset, -1)
            previous.append(F.pad(prev, (0, max_ids - len(ids)), value=-1))
            local_own = torch.as_tensor(local_own, device=h.device)
            own.append(torch.where(local_own >= 0, local_own + offset, -1))
            final.append(torch.as_tensor(local_final, device=h.device) + offset)
            offset += len(bank)
        bank = torch.cat(banks)
        # Split once before tile gathers. Autograd combines these local bank
        # gradients once; gathering from the full packed bank in every tile
        # otherwise zeros a streams-times-larger gradient buffer for each tile.
        bank_lengths = [len(b) for b in banks]
        keys = self.key(bank).split(bank_lengths)
        values = self.value(bank).split(bank_lengths)
        queries = self.query(h).flatten(0, 1)
        index = torch.stack(previous).flatten(0, 1)
        # Two gathered banks plus attention/backward workspace per tile.
        tile = max(
            1, _GRAPH_WORKSPACE_BYTES // (4 * max_ids * width * h.element_size())
        )

        def gather_tile(parts, lo, hi):
            pieces = []
            length = h.shape[1]
            for stream in range(lo // length, (hi - 1) // length + 1):
                start, stop = max(lo, stream * length), min(hi, (stream + 1) * length)
                local = index[start:stop] - offsets[stream]
                pieces.append(self._gather_rows(parts[stream], local.clamp_min(0)))
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

        aggregates = []
        for lo in range(0, len(queries), tile):
            ix = index[lo : lo + tile]
            mask = ix >= 0
            hi = min(len(queries), lo + tile)
            k = gather_tile(keys, lo, hi)
            logits = torch.bmm(k, queries[lo : lo + tile, :, None]).squeeze(-1)
            logits = logits / math.sqrt(width)
            weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(
                -1
            )
            # Entirely empty history produces exactly zero aggregate.
            weights = weights * mask
            v = gather_tile(values, lo, hi)
            aggregates.append(torch.bmm(weights[:, None], v).squeeze(1))
        aggregate = torch.cat(aggregates).reshape_as(h)
        z = self.graph_norm(
            h
            + torch.sigmoid(self.graph_gate(torch.cat((h, aggregate), -1))) * aggregate
        )
        own_index = torch.stack(own)
        previous_id = (
            self._gather_rows(bank, own_index.clamp_min(0))
            * (own_index >= 0)[..., None]
        )
        preceding = torch.stack(
            [
                s.previous_bus if s.previous_bus is not None else h.new_zeros(width)
                for s in states
            ]
        )
        previous_bus = torch.cat((preceding[:, None], h[:, :-1]), 1)
        payload = self.payload_head(torch.cat((previous_bus, previous_id, header), -1))
        payload = payload.reshape(*h.shape[:2], 8, 256)
        classification = self.classifier(z).squeeze(-1)
        outputs = [
            dict(classification_logits=classification[i], payload_logits=payload[i])
            for i in range(len(batches))
        ]
        nodes = [self._gather_rows(bank, ix) for ix in final]
        return outputs, nodes

    def forward(self, batch):
        # Stateless execution intentionally accepts arbitrary physical slices.
        # Prepared routes describe recording-prefix history, which is absent here.
        outputs, _ = self._execute(
            [replace(batch, routing=None)], [EventGraphMambaState()], detach=False
        )
        return outputs[0]

    def _validate_state(self, batch, state):
        if state.recording:
            if (state.recording, state.source_digest) != (
                batch.recording,
                batch.source_digest,
            ):
                raise ValueError("event state belongs to another recording/source")
        elif batch.row_start != 0:
            raise ValueError("a new event stream must start at physical row zero")
        if state.next_row != batch.row_start:
            raise ValueError(
                "event stream is not contiguous: duplicate or missing rows"
            )
        first_timestamp = (
            batch.host_timestamps[0]
            if batch.host_timestamps
            else float(batch.timestamps[0])
        )
        if first_timestamp < state.last_timestamp:
            raise ValueError("event stream clock decreased")

    def scan(self, batches, states=None, *, detach=False):
        """Differentiable equal-length independent streams, with explicit state.

        Detach only returned learned history for truncated backpropagation;
        current chunk outputs retain their complete parameter gradients.
        """
        states = (
            states if states is not None else [EventGraphMambaState() for _ in batches]
        )
        if (
            not batches
            or any(len(b.ids) == 0 for b in batches)
            or len(states) != len(batches)
            or len({len(b.ids) for b in batches}) != 1
        ):
            raise ValueError("scan requires one state per equal-length stream")
        if len({(b.recording, b.source_digest) for b in batches}) != len(batches):
            raise ValueError("independent streams must name distinct recordings")
        for batch, state in zip(batches, states, strict=True):
            self._validate_state(batch, state)
        return self._execute(batches, states, detach=detach)

    def _execute(self, batches, states, *, detach):
        routes = [self._routing(b, s) for b, s in zip(batches, states, strict=True)]
        # Encode all independent frames in a single field/projection operation.
        from types import SimpleNamespace

        packed = SimpleNamespace(
            **{
                name: torch.cat([getattr(b, name) for b in batches])
                for name in ("ids", "lengths", "times", "payload")
            }
        )
        fields, header = self.fields(packed)
        h = self.input_proj(fields).reshape(len(batches), len(batches[0].ids), -1)
        header = header.reshape(len(batches), len(batches[0].ids), -1)
        caches = []
        for i, (norm, block) in enumerate(zip(self.norms, self.temporal, strict=True)):
            incoming = [
                s.caches[i]
                if s.caches
                else block.allocate_inference_cache(1, h.shape[1])
                for s in states
            ]
            conv, memory = (torch.cat([c[j] for c in incoming]) for j in range(2))
            out, conv, memory = mamba_chunk(
                block,
                norm(h),
                conv,
                memory,
                official=self.hparams.backend == "official",
                differentiable_state=not detach,
            )
            caches.append((conv, memory))
            h = h + self.dropout(out)
        outputs, nodes_by_stream = self._readout_batched(
            h, header, batches, states, routes
        )
        following = []
        for i, batch in enumerate(batches):
            ids, nodes = routes[i][0], nodes_by_stream[i]

            def learn(x):
                return x.detach() if detach else x

            following.append(
                EventGraphMambaState(
                    batch.recording,
                    batch.source_digest,
                    batch.row_start + len(batch.ids),
                    batch.host_timestamps[-1]
                    if batch.host_timestamps
                    else float(batch.timestamps[-1]),
                    [(learn(c[i : i + 1]), learn(m[i : i + 1])) for c, m in caches],
                    ids,
                    learn(nodes),
                    learn(h[i, -1]),
                )
            )
        return outputs, following

    @torch.no_grad()
    def stream(self, batch, state=None):
        if self.training:
            raise ValueError("stream requires eval mode")
        outputs, states = self.scan(
            [batch], [state or EventGraphMambaState()], detach=True
        )
        return outputs[0], states[0]

    def loss_sums(self, output, batch):
        return self.batched_loss_sums([output], [batch])

    def batched_loss_sums(self, outputs, batches):
        """One classification reduction and one eligible normal-byte reduction."""
        logits = torch.cat([o["classification_logits"] for o in outputs])
        labels = torch.cat([b.labels for b in batches])
        scored = torch.cat([b.scored for b in batches])
        known = scored & (labels >= 0)
        cls = (
            F.binary_cross_entropy_with_logits(
                logits, labels.clamp_min(0).float(), reduction="none"
            )
            * known
        ).sum()
        eligible = [
            (o, b)
            for o, b in zip(outputs, batches, strict=True)
            if b.verified_normal and b.payload_length_verified
        ]
        # Preserve a connected zero even when the entire objective is masked.
        payload = sum(
            (o["payload_logits"].reshape(-1)[:1].sum() * 0 for o in outputs),
            logits.new_zeros(()),
        )
        count = torch.zeros((), dtype=torch.long, device=logits.device)
        if eligible:
            predictions = torch.cat([o["payload_logits"] for o, _ in eligible])
            targets = torch.cat([b.payload for _, b in eligible]).long()
            lengths = torch.cat([b.lengths for _, b in eligible])
            selected = torch.cat([b.scored for _, b in eligible])
            valid = (
                torch.arange(8, device=logits.device)[None] < lengths[:, None]
            ) & selected[:, None]
            nll = F.cross_entropy(
                predictions.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape(-1, 8)
            payload = payload + (nll * valid).sum()
            count = valid.sum()
        return cls, known.sum(), payload, count

    def losses(self, output, batch):
        cls, nc, payload, npayload = self.loss_sums(output, batch)
        cls, payload = cls / nc.clamp_min(1), payload / npayload.clamp_min(1)
        return cls + self.hparams.payload_weight * payload, dict(
            classification=cls, payload=payload
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device, non_blocking=True)

    def on_train_epoch_start(self):
        if self.trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "use data.effective_batch_events for objective-normalized accumulation"
            )
        self._training_states = {}
        self._training_coverage = {}
        self._epoch_complete = False

    def training_step(self, window, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        cls_total = next(self.parameters()).new_zeros(())
        payload_total = cls_total.clone()
        for batches in window.groups:
            incoming = [
                self._training_states.get(b.recording, EventGraphMambaState())
                for b in batches
            ]
            outputs, states = self.scan(batches, incoming, detach=True)
            for batch, state in zip(batches, states, strict=True):
                if batch.row_start != self._training_coverage.get(batch.recording, 0):
                    raise ValueError("duplicate/missing scored training rows")
                self._training_coverage[batch.recording] = state.next_row
                if batch.recording_end:
                    self._training_states.pop(batch.recording, None)
                else:
                    self._training_states[batch.recording] = state
            cls, _, payload, _ = self.batched_loss_sums(outputs, batches)
            cls = cls / max(1, window.classification_targets)
            payload = payload / max(1, window.payload_targets)
            cls_total += cls.detach()
            payload_total += payload.detach()
            self.manual_backward(cls + self.hparams.payload_weight * payload)
        self.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer.step()
        self.log_dict(
            {"train/classification": cls_total, "train/payload": payload_total},
            batch_size=sum(len(b.ids) for g in window.groups for b in g),
        )
        expected = {
            k: r.count
            for k, r in self.trainer.datamodule.event_recordings.items()
            if r.recording.role == "train_query"
        }
        self._epoch_complete = (
            batch_idx + 1 == self.trainer.num_training_batches
            and self._training_coverage == expected
            and not self._training_states
        )
        return cls_total + self.hparams.payload_weight * payload_total

    def on_train_epoch_end(self):
        if not self._epoch_complete:
            raise ValueError(
                "incomplete event training epoch; refusing an epoch checkpoint"
            )
        self._epoch_complete = True

    def _start_scores(self):
        self._stream_state, self._score_parts, self._score_index = None, [], []

    def on_validation_epoch_start(self):
        self._start_scores()

    def on_test_epoch_start(self):
        self._start_scores()

    def _evaluate(self, batch, prefix):
        if batch.row_start == 0:
            self._stream_state = None
        output, self._stream_state = self.stream(batch, self._stream_state)
        _, parts = self.losses(output, batch)
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in parts.items()}, batch_size=len(batch.ids)
        )
        mask = torch.arange(8, device=batch.ids.device)[None] < batch.lengths[:, None]
        mask &= batch.payload_length_verified
        nll = F.cross_entropy(
            output["payload_logits"].flatten(0, 1),
            batch.payload.long().flatten(),
            reduction="none",
        ).reshape(-1, 8)
        anomaly = (nll * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        values = (
            torch.stack(
                (
                    batch.timestamps,
                    batch.ids,
                    batch.labels,
                    output["classification_logits"].sigmoid(),
                    anomaly,
                    mask.any(-1),
                ),
                -1,
            )
            .double()
            .cpu()
            .numpy()
        )
        # Bounded artifacts, one shard per input chunk. No million-object score list.
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            directory = (
                Path(self.trainer.default_root_dir)
                / "artifacts"
                / f"{prefix}-events-epoch{self.current_epoch:04d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            name = f"{len(self._score_index):08d}.npz"
            np.savez(
                directory / name,
                values=values,
                row_start=batch.row_start,
                recording=batch.recording,
                source_digest=batch.source_digest,
            )
        self._score_index.append(
            dict(
                recording=batch.recording,
                row_start=batch.row_start,
                count=len(batch.ids),
            )
        )
        # Only labels and scalar scores for exact source AP; no node states retained.
        if self.trainer.sanity_checking:
            return
        self._score_parts.append(str(directory / name))

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _finish_scores(self, prefix):
        if not self._score_parts:
            return
        import tempfile

        total = sum(c["count"] for c in self._score_index)
        with tempfile.NamedTemporaryFile(dir=self.trainer.default_root_dir) as backing:
            values = np.memmap(
                backing.name, mode="w+", dtype=np.float64, shape=(total, 4)
            )
            offset = 0
            for path in self._score_parts:
                with np.load(path) as shard:
                    block = shard["values"][:, 2:]
                    values[offset : offset + len(block)] = block
                    offset += len(block)
            values.flush()
        # The open mapping remains valid after unlink on POSIX; metrics retain
        # scalar columns only, never node histories or per-chunk score arrays.
        labels = values[:, 0]
        known = labels >= 0
        metrics = {}
        for head, column in (("classification", 1), ("payload", 2)):
            eligible = (
                known if head == "classification" else known & values[:, 3].astype(bool)
            )
            ap = raw_average_precision(values[eligible, column], labels[eligible])
            if ap is not None:
                metrics[f"{head}_frame_ap"] = ap
                self.log(f"{prefix}/{head}_frame_ap", ap)
        expected = {}
        for chunk in self._score_index:
            key = chunk["recording"]
            if chunk["row_start"] != expected.get(key, 0):
                raise ValueError("duplicate/missing event evaluation rows")
            expected[key] = chunk["row_start"] + chunk["count"]
        if not self.trainer.sanity_checking:
            dm = self.trainer.datamodule
            counts = {
                k: r.count
                for k, r in dm.event_recordings.items()
                if r.recording.role == "val_query"
            }
            if expected != counts:
                raise ValueError(
                    "incomplete event evaluation; every validation frame is required"
                )
            directory = (
                Path(self.trainer.default_root_dir)
                / "artifacts"
                / f"{prefix}-events-epoch{self.current_epoch:04d}"
            )
            (directory / "index.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        prediction_unit="observed_frame",
                        metrics=metrics,
                        coverage=expected,
                        columns=[
                            "timestamp",
                            "can_id",
                            "label",
                            "classification",
                            "payload_nll",
                            "payload_supported",
                        ],
                        chunks=self._score_index,
                        calibration="unfitted; classification threshold 0.5 is diagnostic only",
                    ),
                    indent=2,
                )
                + "\n"
            )
        self._score_parts = []

    def on_validation_epoch_end(self):
        self._finish_scores("val")

    def on_test_epoch_end(self):
        self._finish_scores("test")

    def on_save_checkpoint(self, checkpoint):
        import random

        if not self._epoch_complete:
            raise ValueError(
                "mid-epoch checkpoint unsupported; save after complete recording coverage"
            )
        checkpoint["event_input_contract"] = dict(
            schema_version=1,
            source_manifest_digest=self._source_manifest_digest,
            model_config=dict(self.hparams),
            epoch_boundary=True,
            data_config=self._data_identity() if self._trainer is not None else None,
        )
        numpy_state = np.random.get_state()
        checkpoint["event_rng_state"] = dict(
            python=random.getstate(),
            numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )

    def _data_identity(self):
        dm = self.trainer.datamodule
        return {
            name: getattr(dm, name)
            for name in (
                "chunk_events",
                "stream_batch_size",
                "effective_batch_events",
            )
        }

    def on_train_start(self):
        contract = getattr(self, "_resume_contract", None)
        if contract and contract.get("data_config") != self._data_identity():
            raise ValueError("checkpoint training data configuration differs")
        rng = getattr(self, "_resume_rng", None)
        if rng:
            import random

            random.setstate(rng["python"])
            ns = rng["numpy"]
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])

    def on_load_checkpoint(self, checkpoint):
        contract = checkpoint.get("event_input_contract", {})
        if contract.get("schema_version") != 1:
            raise ValueError(
                "checkpoint lacks event input identity; snapshot weights are incompatible"
            )
        if (
            self._source_manifest_digest is not None
            and self._source_manifest_digest != contract["source_manifest_digest"]
        ):
            raise ValueError(
                "event checkpoint source manifest differs from current data"
            )
        if not contract.get("epoch_boundary", False):
            raise ValueError(
                "mid-epoch/legacy resume unsupported; use an epoch-boundary event checkpoint"
            )
        if contract.get("model_config") != dict(self.hparams):
            raise ValueError("checkpoint model configuration differs")
        self._resume_contract = contract
        self._resume_rng = checkpoint.get("event_rng_state")
        self._source_manifest_digest = contract["source_manifest_digest"]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
