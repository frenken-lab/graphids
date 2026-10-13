"""Actionable rejection of retired event-model contracts."""

LEGACY_REVISION = "daa218eb262fb1c9f9dde067f68d4cc5309d3213"
RETIRED_COMPONENTS = frozenset(
    {
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
        or ".models.temporal." in path
        or ".datamodule.temporal." in path
    ):
        raise ValueError(
            f"retired component {component or path!r}: use vehicle_graph_mamba_joint with can_snapshot_dm; "
            f"evaluate legacy artifacts at graphids-temporal-legacy ({LEGACY_REVISION})"
        )
