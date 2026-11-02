"""Sparse vehicle/ID/byte-position normal-payload controls, with frozen counts.

Run via scripts.event_causal_controls --payload-transition. Both models predict
each current byte before observing its value; evaluation scores the observation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from scripts.event_failure_report import BLOCK, byte_keys

MARGINAL_MASS = 1.0
TRANSITION_MASS = 32.0
TABLES = ("marginal", "position", "transition", "context")


def _table(counts):
    keys = np.array(sorted(counts), dtype=np.int64)
    return keys, np.array([counts[int(k)] for k in keys], dtype=np.int64)


def _totals(keys, counts):
    parents, inverse = np.unique(keys // 256, return_inverse=True)
    totals = np.zeros(len(parents), dtype=np.int64)
    np.add.at(totals, inverse, counts)
    return parents, totals


def lookup(keys, counts, query):
    """Sparse integer lookup; absent states have zero training observations."""
    if not len(keys):
        return np.zeros_like(query, dtype=np.int64)
    index = np.searchsorted(keys, query).clip(0, len(keys) - 1)
    return np.where(keys[index] == query, counts[index], 0)


def fit_payload_profiles(recordings, *, block=BLOCK):
    """Count every trusted training byte/pair, resetting history per recording."""
    if block < 1:
        raise ValueError("block must be positive")
    vehicles = {}
    for events in recordings:
        rec = events.recording
        if (
            rec.role != "train_query"
            or not rec.verified_normal
            or not rec.payload_length_verified
        ):
            continue
        if not np.all(events.arrays["labels"] == 0):
            raise ValueError("verified-normal training contains nonnormal labels")
        profile = vehicles.setdefault(
            rec.vehicle,
            dict(
                marginal=Counter(),
                transition=Counter(),
                recordings=[],
                rows=0,
            ),
        )
        profile["recordings"].append(rec.key)
        profile["rows"] += events.count
        for lo in range(0, events.count, block):
            hi = min(events.count, lo + block)
            for name in ("marginal", "transition"):
                keys, valid = byte_keys(events, lo, hi, transition=name == "transition")
                unique, counts = np.unique(keys[valid], return_counts=True)
                profile[name].update(
                    {int(k): int(c) for k, c in zip(unique, counts, strict=True)}
                )
    if not vehicles:
        raise ValueError("verified-normal payload training required")
    for profile in vehicles.values():
        for name, parent in (("marginal", "position"), ("transition", "context")):
            keys, counts = _table(profile.pop(name))
            profile[f"{name}_keys"], profile[f"{name}_counts"] = keys, counts
            profile[f"{parent}_keys"], profile[f"{parent}_counts"] = _totals(
                keys, counts
            )
    return vehicles


def byte_probabilities(profile, positions, previous, current):
    """Predictive probabilities with uniform -> marginal -> transition backoff.

    positions = CAN_ID * 8 + byte_offset; previous=-1 means unavailable history.
    The current byte indexes a frozen predictive distribution, never its counts.
    """
    position_counts = lookup(
        profile["position_keys"], profile["position_counts"], positions
    )
    marginal_counts = lookup(
        profile["marginal_keys"], profile["marginal_counts"], positions * 256 + current
    )
    marginal = (marginal_counts + MARGINAL_MASS / 256) / (
        position_counts + MARGINAL_MASS
    )
    contexts = positions * 256 + previous.clip(0)
    context_counts = np.where(
        previous >= 0,
        lookup(profile["context_keys"], profile["context_counts"], contexts),
        0,
    )
    pair_counts = np.where(
        previous >= 0,
        lookup(
            profile["transition_keys"],
            profile["transition_counts"],
            contexts * 256 + current,
        ),
        0,
    )
    transition = (pair_counts + TRANSITION_MASS * marginal) / (
        context_counts + TRANSITION_MASS
    )
    return marginal, transition, position_counts, context_counts


def payload_scores(events, profiles, *, block=BLOCK):
    """Mean byte NLL; abstain unless every current valid position is trained.

    First occurrences and missing preceding byte positions use the marginal
    distribution. Unknown labels and attacks still supply preceding observations,
    but no query row ever updates fitted counts. Missing/unverified payloads,
    untrained positions and unprofiled vehicles abstain for both controls.
    """
    if block < 1:
        raise ValueError("block must be positive")
    result = dict(
        byte_marginal=np.zeros(events.count),
        byte_transition=np.zeros(events.count),
        byte_supported=np.zeros(events.count, bool),
        byte_backoff=np.zeros(events.count, bool),
        byte_fully_conditioned=np.zeros(events.count, bool),
        byte_scored_bytes=np.zeros(events.count, dtype=np.uint8),
        byte_context_bytes=np.zeros(events.count, dtype=np.uint8),
    )
    profile = profiles.get(events.recording.vehicle)
    if profile is None or not events.recording.payload_length_verified:
        return result
    offsets = np.arange(8)[None]
    arrays = events.arrays
    for lo in range(0, events.count, block):
        hi = min(events.count, lo + block)
        prior_rows = events.previous_rows[lo:hi]
        valid = offsets < arrays["lengths"][lo:hi, None]
        prior_valid = (prior_rows >= 0)[:, None] & (
            offsets < arrays["lengths"][prior_rows.clip(0), None]
        )
        previous = np.where(
            prior_valid, arrays["payload"][prior_rows.clip(0)].astype(np.int64), -1
        )
        positions = arrays["ids"][lo:hi, None] * 8 + offsets
        marginal, transition, position_counts, context_counts = byte_probabilities(
            profile,
            positions,
            previous,
            arrays["payload"][lo:hi],
        )
        length = valid.sum(-1)
        supported = (length > 0) & ((position_counts > 0) | ~valid).all(-1)
        contexts = (valid & (context_counts > 0)).sum(-1)
        for name, probability in (
            ("byte_marginal", marginal),
            ("byte_transition", transition),
        ):
            score = -(np.log(probability) * valid).sum(-1) / length.clip(1)
            result[name][lo:hi] = np.where(supported, score, 0)
        result["byte_supported"][lo:hi] = supported
        result["byte_backoff"][lo:hi] = supported & (contexts < length)
        result["byte_fully_conditioned"][lo:hi] = supported & (contexts == length)
        result["byte_scored_bytes"][lo:hi] = np.where(supported, length, 0)
        result["byte_context_bytes"][lo:hi] = np.where(supported, contexts, 0)
    return result


def save_profiles(profiles, output=None):
    """Compact JSON summaries and lossless sparse integer count artifacts."""
    summaries = {}
    for vehicle, profile in profiles.items():
        summary = {name: profile[name] for name in ("recordings", "rows")}
        summary.update(
            positions=len(profile["position_keys"]),
            marginal_states=len(profile["marginal_keys"]),
            transition_states=len(profile["transition_keys"]),
            contexts=len(profile["context_keys"]),
            training_bytes=int(profile["position_counts"].sum()),
            training_pairs=int(profile["context_counts"].sum()),
        )
        if output is not None:
            path = output / f"payload-profile-{len(summaries):02d}.npz"
            np.savez_compressed(
                path,
                **{
                    f"{name}_{part}": profile[f"{name}_{part}"]
                    for name in TABLES
                    for part in ("keys", "counts")
                },
            )
            summary["file"] = path.name
        summaries[vehicle] = summary
    return summaries
