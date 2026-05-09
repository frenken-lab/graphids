"""GraphIDS plan layer — Python plans → typed row dicts → JSON.

Plan authors import from this package:

    from graphids.plan import (
        fit_row, test_row,                           # row builders
        extract, analyze, hf_push,                   # one-shot ops rows
        FUSION_TRAINER,                              # trainer overlay for fusion rows
        gat, vgae, dgi,                              # model factories
        focal, ce, weighted_ce, vgae_task,           # loss factories
        curriculum, soft_label_distillation,         # compound loss factories
        feature_distillation,
        can_bus, graph_dm, fusion_dm,                # data factories
        score_random, score_vgae,                    # difficulty scorer factories
        REWARD, REWARD_MINIMAL,                      # RL reward-shaping dicts
    )
"""

from graphids.plan.primitives import (
    REWARD,
    REWARD_MINIMAL,
    BanditCfg,
    CANBusCfg,
    CELossCfg,
    CurriculumLossCfg,
    DataCfg,
    DGICfg,
    DifficultyCfg,
    DQNCfg,
    FeatureDistillationCfg,
    FocalLossCfg,
    FusionDMCfg,
    GATCfg,
    GraphDMCfg,
    LinearRampCfg,
    LossFn,
    MLPFusionCfg,
    ModelCfg,
    MoECfg,
    ScoreRandomCfg,
    ScoreVGAECfg,
    SimpleLossFn,
    SoftLabelDistillationCfg,
    VGAECfg,
    VGAETaskLossCfg,
    WeightedAvgCfg,
    WeightedCELossCfg,
    bandit,
    can_bus,
    ce,
    curriculum,
    dgi,
    dqn,
    feature_distillation,
    focal,
    fusion_dm,
    gat,
    graph_dm,
    mlp_fusion,
    moe,
    score_random,
    score_vgae,
    soft_label_distillation,
    vgae,
    vgae_task,
    weighted_avg,
    weighted_ce,
)
from graphids.plan.rows import (
    FUSION_TRAINER,
    analyze,
    extract,
    fit_row,
    hf_push,
    test_row,
)

__all__ = [
    # row builders
    "fit_row",
    "test_row",
    # one-shot ops rows
    "extract",
    "analyze",
    "hf_push",
    # fusion trainer overlay
    "FUSION_TRAINER",
    # model factories
    "gat",
    "vgae",
    "dgi",
    "bandit",
    "dqn",
    "mlp_fusion",
    "moe",
    "weighted_avg",
    # loss factories
    "focal",
    "ce",
    "weighted_ce",
    "vgae_task",
    "curriculum",
    "soft_label_distillation",
    "feature_distillation",
    # data factories
    "can_bus",
    "graph_dm",
    "fusion_dm",
    # difficulty scorer factories
    "score_random",
    "score_vgae",
    # reward shaping dicts
    "REWARD",
    "REWARD_MINIMAL",
    # config types (for type annotations in plan modules)
    "ModelCfg",
    "LossFn",
    "SimpleLossFn",
    "DataCfg",
    "DifficultyCfg",
    "GATCfg",
    "VGAECfg",
    "DGICfg",
    "BanditCfg",
    "DQNCfg",
    "MLPFusionCfg",
    "MoECfg",
    "WeightedAvgCfg",
    "FocalLossCfg",
    "CELossCfg",
    "WeightedCELossCfg",
    "VGAETaskLossCfg",
    "CurriculumLossCfg",
    "LinearRampCfg",
    "SoftLabelDistillationCfg",
    "FeatureDistillationCfg",
    "ScoreRandomCfg",
    "ScoreVGAECfg",
    "CANBusCfg",
    "GraphDMCfg",
    "FusionDMCfg",
]
