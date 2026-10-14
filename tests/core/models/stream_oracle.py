"""Frozen pre-vectorization recurrence: independent correctness oracle."""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


@dataclass
class OracleState:
    recording: str = ""
    context_digest: str = ""
    caches: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    last_bin: int = -1


@torch.no_grad()
def oracle_stream(self, batch, state=None):
    if batch.x.shape[0] != 1:
        raise ValueError("continuous streaming requires one recording per batch")
    if self.training:
        raise ValueError("streaming inference requires eval mode")
    state = state or OracleState()
    key, digest = batch.recording_keys[0], batch.context_digests[0]
    if (state.recording, state.context_digest) != (key, digest):
        state = OracleState(key, digest)
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
                        block.allocate_inference_cache(1, 1) for block in self.temporal
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
                targets = torch.stack([batch.counts[0, ti, ni] for ni, pred in matched])
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
        name: torch.cat([o[name] for o in all_outputs], 1) for name in all_outputs[0]
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


def oracle_encode(self, batch):
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
                    stop < t and valid[stop] and generations[stop] == generations[start]
                ):
                    stop += 1
                segments.setdefault(stop - start, []).append((bi, ni, start, stop))
                start = stop
    h = torch.zeros_like(tokens)
    for _, group in segments.items():
        x = torch.stack([tokens[bi, start:stop, ni] for bi, ni, start, stop in group])
        for norm, block in zip(self.temporal_norms, self.temporal, strict=True):
            x = x + self.dropout(block(norm(x)))
        for i, (bi, ni, start, stop) in enumerate(group):
            h[bi, start:stop, ni] = x[i]
    return h
