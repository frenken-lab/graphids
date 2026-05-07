"""Leaf building blocks for plans — class-path catalog + composing primitives.

Plans write::

    from graphids.plan import spec, GAT, FOCAL, graph_dm, can_bus
    spec(GAT, scale="large", dropout=0.3)
    graph_dm(source=can_bus(dataset="hcrl_sa", seed=42))

Why:
- ``spec(cls_path, **init_args)`` is a 3-line dict builder. The 11 trivial
  primitive functions (``gat``, ``focal``, ``ce``, …) collapsed into
  string constants here. Defaults live with the model class
  (``GAT.__init__``), not duplicated in a config wrapper.
- The four primitives that compose / validate stay as functions:
  ``can_bus`` (registry validation), ``graph_dm`` (conditional optional
  knobs), ``fusion_dm`` (path derivation), ``curriculum`` (deepcopy +
  reduction injection).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from graphids.paths import load_catalog
from graphids.paths import states_dir as _states_dir

# ---------------------------------------------------------------- class_paths

# Models
GAT = "graphids.core.models.supervised.gat.GAT"
VGAE = "graphids.core.models.autoencoder.vgae.VGAE"
DGI = "graphids.core.models.autoencoder.dgi.DGI"
BANDIT = "graphids.core.models.fusion.bandit.BanditFusionModule"
DQN = "graphids.core.models.fusion.dqn.DQNFusionModule"
MLP_FUSION = "graphids.core.models.fusion.mlp.MLPFusionModule"
MOE_FUSION = "graphids.core.models.fusion.moe.MoEFusionModule"
WAVG_FUSION = "graphids.core.models.fusion.weighted_avg.WeightedAvgModule"

# Losses
FOCAL = "graphids.core.losses.FocalLoss"
CE = "graphids.core.losses.CrossEntropyLoss"
WEIGHTED_CE = "graphids.core.losses.WeightedCrossEntropyLoss"
VGAE_TASK = "graphids.core.losses.VGAETaskLoss"
CURRICULUM_LOSS = "graphids.core.losses.CurriculumWeightedLoss"
LINEAR_RAMP = "graphids.core.curriculum.LinearRampSchedule"

# Difficulty scorers
SCORE_RANDOM = "graphids.core.data.preprocessing.curriculum.ScoreRandom"
SCORE_VGAE = "graphids.core.data.preprocessing.curriculum.ScoreVGAE"

# Data sources / datamodules
CAN_BUS = "graphids.core.data.datasets.can_bus.CANBusSource"
GRAPH_DM = "graphids.core.data.datamodule.GraphDataModule"
FUSION_DM = "graphids.core.data.datamodule.fusion.FusionDataModule"

# Fusion reward shaping. Two variants — a fusion plan picks one by passing the
# corresponding dict as ``reward_kwargs`` to the bandit/DQN spec.
#
# ``REWARD`` is the legacy 2-way (correct/incorrect) base reward with agreement
# / confidence / disagreement / overconfidence / balance shaping terms. Pre-
# 2026-05-07 default. Diagnosed (2026-05-06 fusion analysis) as the cause of
# the all-benign equilibrium under 86% benign hcrl_sa and the constant-arm-20
# RL collapse — the agreement bonus rewards the majority-class rule, and the
# balance term penalizes the data-optimal alpha=1.0 direction.
REWARD: dict[str, Any] = {
    "vgae_weights": [0.4, 0.3, 0.3],
    "correct": 3.0,
    "incorrect": -3.0,
    "confidence_weight": 0.5,
    "combined_conf_weight": 0.3,
    "disagreement_penalty": -1.0,
    "overconf_penalty": -1.5,
    "balance_weight": 0.3,
}

# ``REWARD_MINIMAL`` — Phase 1.1 of the fusion improvement plan. PBRS-compliant
# 4-way (TP/TN/FP/FN) base reward + attack-gated confidence bonus. No shaping
# terms that violate Ng-Harada-Russell 1999 policy invariance. FN cost = 4×FP
# cost encodes IDS F2-optimization (Davis & Goadrich 2006). Submit alongside
# the legacy variant once Phase 0 diagnostic logging confirms r_agreement
# dominance under the legacy reward.
REWARD_MINIMAL: dict[str, Any] = {
    "mode": "minimal",
    "vgae_weights": [0.4, 0.3, 0.3],
    "tp_reward": 3.0,
    "tn_reward": 1.5,
    "fp_cost": -1.5,
    "fn_cost": -6.0,
    "confidence_weight": 0.3,
}


# ------------------------------------------------------------------ spec helper


def spec(cls_path: str, **init_args: Any) -> dict[str, Any]:
    """Build a ``{class_path, init_args}`` block. Defaults live with the class."""
    return {"class_path": cls_path, "init_args": init_args}


# ------------------------------------------------------------ composing primitives


def can_bus(*, dataset: str, seed: int, **overrides: Any) -> dict[str, Any]:
    """``CANBusSource`` block with registry validation.

    Login-node fail-fast — surface unknown datasets before SLURM ingest.
    """
    registry = load_catalog()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset} (registry: {', '.join(sorted(registry))})")
    init_args: dict[str, Any] = {
        "name": dataset,
        "seed": seed,
        "window_size": 100,
        "stride": 100,
        "val_fraction": 0.2,
    }
    init_args.update(overrides)
    return {"class_path": CAN_BUS, "init_args": init_args}


def graph_dm(
    *,
    source: dict[str, Any],
    label_filter: str | None = None,
    difficulty: dict[str, Any] | None = None,
    scope_label: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    """``GraphDataModule`` block. Optional knobs are absent (not ``None``)
    when not passed, so curriculum vs. plain runs differ in dict shape.
    """
    init_args: dict[str, Any] = {"dataset": source}
    if label_filter is not None:
        init_args["label_filter"] = label_filter
    if difficulty is not None:
        init_args["difficulty"] = difficulty
        init_args["scope_label"] = scope_label
    init_args.update(overrides)
    return {"class_path": GRAPH_DM, "init_args": init_args}


def fusion_dm(
    *,
    dataset: str,
    seed: int,
    method: str,
    batch_size: int = 128,
    episode_sample_size: int = 20_000,
) -> dict[str, Any]:
    """``FusionDataModule`` block — derives ``cached_states_dir`` from catalog."""
    return {
        "class_path": FUSION_DM,
        "init_args": {
            "cached_states_dir": _states_dir(dataset, seed),
            "method": method,
            "batch_size": batch_size,
            "episode_sample_size": episode_sample_size,
        },
    }


def curriculum(base: dict[str, Any], schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap ``base`` (a bare ``{class_path, init_args}`` block) in a curriculum loss.

    Forces ``reduction='none'`` on the base so the wrapper can apply
    per-example weights before reducing.
    """
    base_per_example = deepcopy(base)
    base_per_example.setdefault("init_args", {})["reduction"] = "none"
    if schedule is None:
        schedule = spec(LINEAR_RAMP, start_ratio=1.0, end_ratio=10.0, max_epochs=300)
    return {
        "class_path": CURRICULUM_LOSS,
        "init_args": {"base_loss": base_per_example, "schedule": deepcopy(schedule)},
    }
