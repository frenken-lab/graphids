"""Rebuild dataset cache for ``(dataset, vocab_scope ∈ {train, all})``.

Two CacheRows per ``graphids run`` invocation — both partitions warmed so
ablations comparing ``vocab_scope`` regimes don't pay first-run cost. CPU
only; submit on the cluster's short CPU profile.
"""

from __future__ import annotations

from typing import Any


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    return [
        {
            "name": f"cache_{dataset}_voc{scope}",
            "action": "cache",
            "dataset": dataset,
            "vocab_scope": scope,
            "seed": seed,
            "resources": {"mode": "cpu", "length": "short"},
        }
        for scope in ("train", "all")
    ]
