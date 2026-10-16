"""Actionable rejection of retired event-model contracts."""

LEGACY_REVISION = "daa218eb262fb1c9f9dde067f68d4cc5309d3213"
RETIRED_COMPONENTS = frozenset(
    {
        "vehicle_graph_mamba_joint",
        "can_snapshot_dm",
        "ce",
        "focal",
        "weighted_ce",
        "temporal_event_classifier",
        "temporal_rnn_classifier",
        "temporal_gat",
        "temporal_vgae",
        "temporal_hybrid",
        "temporal_dm",
        "can_bus",
        "cross_vehicle",
    }
)


def reject_retired(spec):
    if not isinstance(spec, dict):
        return
    component = spec.get("type", "")
    path = spec.get("class_path", "")
    retired_classes = {
        "CELossCfg",
        "FocalLossCfg",
        "WeightedCELossCfg",
        "CANSnapshotRepresentationCfg",
        "SnapshotBatchProfilerCallback",
        "GraphMambaLearningCurveCallback",
        "VehicleGraphMambaJoint",
        "VehicleGraphMambaJointCfg",
        "CANSnapshotDataModule",
        "CANSnapshotDMCfg",
        "TemporalEventClassifierCfg",
        "TemporalGATCfg",
        "TemporalRNNClassifierCfg",
        "TemporalVGAECfg",
        "TemporalHybridCfg",
        "TemporalDMCfg",
        "CANBusCfg",
        "CrossVehicleCfg",
    }
    if (
        component in RETIRED_COMPONENTS
        or path.rsplit(".", 1)[-1] in retired_classes
        or ".core.losses." in path
        or ".primitives_losses." in path
        or ".snapshots." in path
        or ".graph_mamba." in path
        or ".models.temporal." in path
        or ".datamodule.temporal." in path
    ):
        raise ValueError(
            f"retired component {component or path!r}: use event_graph_mamba with can_event_dm; restore snapshot artifacts from archives/snapshot-2026-10-16.tar.gz; "
            f"evaluate legacy artifacts at graphids-temporal-legacy ({LEGACY_REVISION})"
        )


RETIRED_EXPERIMENTS = frozenset(
    [
        "graph_mamba_ctat_expanded_source.yml",
        "graph_mamba_ctat_learning.yml",
        "graph_mamba_ctat_learning_corrected.yml",
        "graph_mamba_ctat_multiattack_source.yml",
        "graph_mamba_ctat_pilot.yml",
        "graph_mamba_ctat_pilot_test.yml",
        "graph_mamba_joint.yml",
        "graph_mamba_joint_cardinal.yml",
        "graph_mamba_joint_parallel.yml",
        "graph_mamba_joint_test.yml",
    ]
)
