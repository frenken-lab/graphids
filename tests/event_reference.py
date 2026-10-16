"""Independent serial graph/routing/loss oracle for tests and the Slurm gate.

Shares parameters and (on CUDA) official temporal operators, never production
routing, scan, readout or loss helpers. CPU temporal recurrence is explicit.
"""

import math

import torch
from torch.nn import functional as F

from graphids.core.models.event_graph_mamba import EventGraphMambaState
from graphids.core.models.mamba_chunk import mamba_chunk


def serial_routing(batch, state):
    ids = list(state.ids)
    for key in batch.routing_ids:
        if key not in ids:
            ids.append(key)
    latest = {key: i for i, key in enumerate(state.ids)}
    previous, own = [], []
    for i, key in enumerate(batch.routing_ids):
        previous.append([latest.get(k, -1) for k in ids])
        own.append(latest.get(key, -1))
        latest[key] = len(state.ids) + i
    device = batch.ids.device
    return tuple(ids), *(
        torch.tensor(x, device=device, dtype=torch.long)
        for x in (previous, own, [latest[k] for k in ids])
    )


def serial_readout(model, h, header, batch, state):
    ids, previous, own, final = serial_routing(batch, state)
    past = state.nodes if state.nodes is not None else h.new_empty((0, h.shape[-1]))
    bank = torch.cat((past, h))
    keys, values, query = model.key(bank), model.value(bank), model.query(h)
    # Serial multiply/reduce formula, independently tiled to fit large checks.
    tile = max(1, (256 * 1024**2) // (4 * len(ids) * h.shape[-1] * h.element_size()))
    aggregates = []
    for lo in range(0, len(h), tile):
        ix = previous[lo : lo + tile]
        mask = ix >= 0
        logits = (query[lo : lo + tile, None] * keys[ix.clamp_min(0)]).sum(
            -1
        ) / math.sqrt(h.shape[-1])
        weights = (
            logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1) * mask
        )
        aggregates.append((weights[..., None] * values[ix.clamp_min(0)]).sum(1))
    aggregate = torch.cat(aggregates)
    z = model.graph_norm(
        h + torch.sigmoid(model.graph_gate(torch.cat((h, aggregate), -1))) * aggregate
    )
    previous_id = bank[own.clamp_min(0)] * (own >= 0)[:, None]
    preceding = (
        state.previous_bus
        if state.previous_bus is not None
        else h.new_zeros(h.shape[-1])
    )
    previous_bus = torch.cat((preceding[None], h[:-1]))
    output = dict(
        classification_logits=model.classifier(z).squeeze(-1),
        payload_logits=model.payload_head(
            torch.cat((previous_bus, previous_id, header), -1)
        ).reshape(-1, 8, 256),
    )
    return output, ids, bank[final]


def reference_temporal(block, x, conv, memory):
    projected, gate = block.in_proj(x).chunk(2, -1)
    history = torch.cat((conv, projected.transpose(1, 2)), -1)
    values = F.silu(
        F.conv1d(
            history[:, :, -(len(x[0]) + block.d_conv - 1) :],
            block.conv1d.weight,
            block.conv1d.bias,
            groups=block.d_inner,
        )
    ).transpose(1, 2)
    outputs = []
    for t in range(x.shape[1]):
        delta, b, c = block.x_proj(values[:, t]).split(
            (block.dt_rank, block.d_state, block.d_state), -1
        )
        delta = F.softplus(F.linear(delta, block.dt_proj.weight, block.dt_proj.bias))
        memory = (
            torch.exp(delta[..., None] * -block.A_log.exp()) * memory
            + delta[..., None] * b[:, None] * values[:, t, :, None]
        )
        y = (memory * c[:, None]).sum(-1) + block.D * values[:, t]
        outputs.append(block.out_proj(y * F.silu(gate[:, t])))
    return torch.stack(outputs, 1), history[:, :, -block.d_conv :], memory


def serial_scan(model, batches, states=None, *, detach=False):
    states = states if states is not None else [EventGraphMambaState() for _ in batches]
    outputs, following = [], []
    for batch, state in zip(batches, states, strict=True):
        fields, header = model.fields(batch)
        h = model.input_proj(fields)[None]
        caches = []
        for i, (norm, block) in enumerate(
            zip(model.norms, model.temporal, strict=True)
        ):
            conv, memory = (
                state.caches[i]
                if state.caches
                else block.allocate_inference_cache(1, h.shape[1])
            )
            if model.hparams.backend == "official":
                out, conv, memory = mamba_chunk(
                    block,
                    norm(h),
                    conv,
                    memory,
                    official=True,
                    differentiable_state=not detach,
                )
            else:
                out, conv, memory = reference_temporal(block, norm(h), conv, memory)
            caches.append((conv, memory))
            h = h + model.dropout(out)
        output, ids, nodes = serial_readout(model, h[0], header, batch, state)

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
                [(learn(c), learn(m)) for c, m in caches],
                ids,
                learn(nodes),
                learn(h[0, -1]),
            )
        )
        outputs.append(output)
    return outputs, following


def serial_loss_sums(output, batch):
    known = batch.scored & (batch.labels >= 0)
    classification = F.binary_cross_entropy_with_logits(
        output["classification_logits"],
        batch.labels.clamp_min(0).float(),
        reduction="none",
    )
    valid = (
        (torch.arange(8, device=batch.ids.device)[None] < batch.lengths[:, None])
        & batch.scored[:, None]
        & batch.verified_normal
        & batch.payload_length_verified
    )
    nll = F.cross_entropy(
        output["payload_logits"].flatten(0, 1),
        batch.payload.long().flatten(),
        reduction="none",
    ).reshape(-1, 8)
    return (classification * known).sum(), known.sum(), (nll * valid).sum(), valid.sum()


def serial_loss(model, output, batch):
    c, nc, p, npayload = serial_loss_sums(output, batch)
    return c / nc.clamp_min(1) + model.hparams.payload_weight * p / npayload.clamp_min(
        1
    )
