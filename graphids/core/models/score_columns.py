"""Compact evaluation columns: bulk transfers, calibration and rank-zero metrics."""

from dataclasses import dataclass

import numpy as np
import torch

COLUMNS = (
    "bin",
    "close",
    "exposure",
    "classification",
    "anomaly",
    "component0",
    "component1",
    "component2",
    "mask0",
    "mask1",
    "mask2",
    "anomaly_eligible",
    "label",
    "normal",
    "events",
    "live_nodes",
    "silent_anchors",
    "unavailable_payload_fields",
    "alert",
    "reference_coverage",
    "active_edges",
)
COL = {name: i for i, name in enumerate(COLUMNS)}
BOOLS = {"normal", "anomaly_eligible", "alert"}
INTS = {
    "bin",
    "label",
    "events",
    "live_nodes",
    "silent_anchors",
    "unavailable_payload_fields",
    "active_edges",
}


def raw_average_precision(scores, labels):
    """Exact noninterpolated AP, grouping equal raw scores at their rank endpoint.

    Unknown labels (-1) are excluded. Single-class groups return None.
    No probability conversion is applied, even outside [0, 1].
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("AP requires finite raw scores")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("labels must be -1, 0 or 1")
    known = labels >= 0
    scores, labels = scores[known], labels[known]
    if set(labels) != {0, 1}:
        return None
    order = np.argsort(-scores, kind="stable")
    scores, positives = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(positives)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives.sum())


@dataclass
class ScoreBlock:
    identity: dict
    values: np.ndarray

    @classmethod
    def from_batch(cls, model, batch, output):
        probability = output["classification_logits"].sigmoid()
        components = output["anomaly_components"]
        masks = output["anomaly_component_mask"]
        mode = "conditioned" if batch.context_digests[0] != "zero-shot" else "zero-shot"
        bundle = model.calibration_bundles.get(mode)
        thresholds = bundle.thresholds if bundle else model.alert_thresholds
        anomaly_threshold = thresholds[1] if thresholds[1] is not None else float("inf")
        anomaly_available = masks.bool().any(-1)
        if bundle:
            anomaly_available = (
                masks.bool() & masks.new_tensor(bundle.available).bool()
            ).any(-1)
        alert = (model.hparams.classification & (probability >= thresholds[0])) | (
            model.hparams.forecasting
            & anomaly_available
            & output["anomaly_eligible"]
            & (output["anomaly_score"] >= anomaly_threshold)
        )
        tensors = (
            batch.bin_indices,
            batch.closes,
            batch.exposure,
            probability,
            output["anomaly_score"],
            *components.unbind(-1),
            *masks.unbind(-1),
            output["anomaly_eligible"],
            batch.labels,
            batch.normal,
            batch.event_count,
            batch.live.sum(-1),
            (batch.in_reference[:, None] & batch.live & (batch.counts == 0)).sum(-1),
            (batch.live[..., None] & (batch.x[..., 12:20] == 0)).sum((-1, -2)),
            alert,
            (batch.live & batch.in_reference[:, None]).sum(-1)
            / batch.live.sum(-1).clamp_min(1),
            (
                batch.edge_mask[:, None]
                & batch.live[..., None]
                & batch.live.gather(
                    2,
                    batch.neighbors[:, None]
                    .expand(-1, batch.live.shape[1], -1, -1)
                    .flatten(2),
                ).reshape(*batch.live.shape, -1)
            ).sum((-1, -2))
            * model.reference_policy.graph_edges
            * model.hparams.graph_enabled,
        )
        # One transfer for all numeric columns, retaining FP64 recording clocks.
        values = (
            torch.stack([v[0].double() for v in tensors], -1)[batch.scored[0]]
            .detach()
            .cpu()
            .numpy()
        )
        rec = model._recordings[batch.recording_keys[0]]
        return cls(
            dict(
                recording=rec.key,
                dataset=rec.dataset,
                vehicle=rec.vehicle,
                role=rec.role,
                context=batch.context_digests[0],
            ),
            values,
        )


class ScoreTable:
    def __init__(self, blocks):
        self.blocks = blocks
        self.values = (
            np.concatenate([b.values for b in blocks])
            if blocks
            else np.empty((0, len(COLUMNS)))
        )
        self.identities = {
            name: np.concatenate(
                [np.repeat(b.identity[name], len(b.values)) for b in blocks]
            )
            if blocks
            else np.array([], dtype=str)
            for name in ("recording", "dataset", "vehicle", "role", "context")
        }

    def __len__(self):
        return len(self.values)

    def column(self, name):
        return self.values[:, COL[name]]

    def keys(self):
        return list(
            zip(
                self.identities["recording"],
                self.identities["context"],
                self.column("bin").astype(np.int64),
                strict=True,
            )
        )

    def check_unique(self, stage):
        keys = self.keys()
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate scored {stage} bins across workers")

    def __iter__(self):
        for i, values in enumerate(self.values):
            row = {name: identity[i] for name, identity in self.identities.items()}
            for name, value in zip(COLUMNS, values, strict=True):
                if name.startswith(("component", "mask")):
                    continue
                row[name] = (
                    bool(value)
                    if name in BOOLS
                    else int(value)
                    if name in INTS
                    else float(value)
                )
            row["anomaly_components"] = values[5:8].tolist()
            row["anomaly_component_mask"] = values[8:11].astype(bool).tolist()
            yield row

    def mode_mask(self, mode):
        if mode not in ("zero-shot", "conditioned"):
            raise ValueError(f"unsupported inference mode: {mode}")
        return (self.identities["context"] == "zero-shot") == (mode == "zero-shot")

    @classmethod
    def from_rows(cls, rows):
        # Keep one compact block per recording/context, rather than allocating
        # a Python object and five identity arrays for every individual bin.
        fields = ("recording", "dataset", "vehicle", "role", "context")
        groups = {}
        for row in rows:
            identity = tuple(row[k] for k in fields)
            values = [row.get(name, 0) for name in COLUMNS]
            values[5:8] = row["anomaly_components"]
            values[8:11] = row["anomaly_component_mask"]
            groups.setdefault(identity, []).append(values)
        return cls(
            [
                ScoreBlock(
                    dict(zip(fields, key, strict=True)),
                    np.asarray(values, dtype=np.float64),
                )
                for key, values in groups.items()
            ]
        )

    def calibrate(self, model):
        from .calibration import MODES, CalibrationBundle, weights_digest

        fingerprint = weights_digest(model)
        model.calibration_bundles = {
            mode: CalibrationBundle.fit(
                self,
                checkpoint_hash=fingerprint,
                source_manifest=getattr(
                    model, "source_manifest_digest", model.data_manifest_digest
                ),
                reference_policy=model.reference_policy.digest,
                mode=mode,
                quantile=model.hparams.alert_quantile,
            )
            for mode in MODES
        }
        model.calibrated.fill_(True)
        self.apply_bundles(
            model.calibration_bundles,
            classification=model.hparams.classification,
            forecasting=model.hparams.forecasting,
        )

    def apply_bundles(self, bundles, *, classification=True, forecasting=True):
        for mode in ("zero-shot", "conditioned"):
            selected = self.mode_mask(mode)
            if not selected.any():
                continue
            if mode not in bundles:
                raise ValueError(f"missing calibration mode: {mode}")
            bundle = bundles[mode]
            if bundle.mode != mode:
                raise ValueError("calibration bundle mode mismatch")
            self.values[selected, COL["anomaly"]] = bundle.scores(
                self.values[selected, 5:8], self.values[selected, 8:11]
            )
            cls_alert = classification & (
                self.column("classification")[selected] >= bundle.thresholds[0]
            )
            anomaly = (
                forecasting
                & (
                    self.values[selected, 8:11].astype(bool)
                    & np.asarray(bundle.available)
                ).any(-1)
                & self.column("anomaly_eligible")[selected].astype(bool)
                & (
                    self.column("anomaly")[selected]
                    >= (
                        bundle.thresholds[1]
                        if bundle.thresholds[1] is not None
                        else np.inf
                    )
                )
            )
            self.values[selected, COL["alert"]] = cls_alert | anomaly

    def report(
        self, bundles=None, *, classification=True, forecasting=True, partition=None
    ):
        for head, enabled in (
            ("classification", classification),
            ("anomaly", forecasting),
        ):
            if enabled and not np.isfinite(self.column(head)).all():
                raise ValueError(f"nonfinite scored {head} values")
        groups = {}
        roles = self.identities["role"]
        allowed = (
            ("val_query",)
            if partition == "source"
            else ("target_normal_test", "target_attack_test")
            if partition == "target"
            else ("target_normal_test", "target_attack_test", "val_query")
        )
        test = np.isin(roles, allowed)
        for mode in ("zero-shot", "conditioned"):
            base = test & self.mode_mask(mode)
            selectors = {"all": base}
            for field in ("vehicle", "dataset", "recording"):
                for key in sorted(set(self.identities[field][base])):
                    selectors[f"{field}/{key}"] = base & (self.identities[field] == key)
            for key, selected in selectors.items():
                if not selected.any():
                    continue
                normal = (
                    selected
                    & self.column("normal").astype(bool)
                    & (self.column("label") == 0)
                )
                hours = float(self.column("exposure")[normal].sum()) / 3600
                result = dict(
                    scored_bins=int(selected.sum()), normal_hours=hours, heads={}
                )
                bundle = bundles.get(mode) if bundles else None
                if bundles and bundle is None:
                    raise ValueError(f"missing calibration mode: {mode}")
                if bundle and bundle.mode != mode:
                    raise ValueError("calibration bundle mode mismatch")
                cls = self.column("classification") >= (
                    bundle.thresholds[0] if bundle else np.inf
                )
                anomaly_available = self.values[:, 8:11].astype(bool)
                if bundle:
                    anomaly_available &= np.asarray(bundle.available)
                ano = (
                    anomaly_available.any(-1)
                    & self.column("anomaly_eligible").astype(bool)
                    & (
                        self.column("anomaly")
                        >= (
                            bundle.thresholds[1]
                            if bundle and bundle.thresholds[1] is not None
                            else np.inf
                        )
                    )
                )
                cls &= classification
                ano &= forecasting
                if bundle:
                    for name, alerts in (
                        ("classification_only", cls & ~ano),
                        ("anomaly_only", ano & ~cls),
                        ("overlap", cls & ano),
                        ("combined", cls | ano),
                        ("classification", cls),
                        ("anomaly", ano),
                    ):
                        result[name] = dict(
                            bin_alerts=int((alerts & selected).sum()),
                            normal_bin_alerts=int((alerts & normal).sum()),
                            false_bin_alerts_per_hour=float((alerts & normal).sum())
                            / hours
                            if hours
                            else None,
                        )
                for head in ("classification", "anomaly"):
                    if not (
                        classification if head == "classification" else forecasting
                    ):
                        result["heads"][head] = dict(
                            ap=None,
                            status="disabled",
                            eligible_bins=0,
                            positive_prevalence=None,
                        )
                        continue
                    eligible = selected & (self.column("label") >= 0)
                    unavailable = False
                    if head == "anomaly":
                        eligible &= self.column("anomaly_eligible").astype(bool)
                        eligible &= self.values[:, 8:11].astype(bool).any(-1)
                        if bundle:
                            available = np.asarray(bundle.available)
                            eligible &= (
                                self.values[:, 8:11].astype(bool) & available
                            ).any(-1)
                            unavailable = not available.any()
                    labels, scores = (
                        self.column("label")[eligible],
                        self.column(head)[eligible],
                    )
                    value = raw_average_precision(scores, labels)
                    result["heads"][head] = dict(
                        ap=value,
                        status="unavailable_components"
                        if unavailable
                        else "ok"
                        if value is not None
                        else "missing_both_classes",
                        eligible_bins=int(eligible.sum()),
                        positive_prevalence=float(labels.mean())
                        if len(labels)
                        else None,
                    )
                distributions = {}
                for label in (-1, 0, 1):
                    subset = selected & (self.column("label") == label)
                    if subset.any():
                        distributions[str(label)] = {
                            name: np.quantile(
                                self.column(name)[subset], [0, 0.5, 0.9, 0.99, 1]
                            ).tolist()
                            for name in (
                                "classification",
                                "anomaly",
                                "live_nodes",
                                "silent_anchors",
                                "component0",
                                "component1",
                                "component2",
                                "reference_coverage",
                                "active_edges",
                            )
                            if not (
                                (name == "classification" and not classification)
                                or (
                                    (name == "anomaly" or name.startswith("component"))
                                    and not forecasting
                                )
                            )
                        }
                result["distributions_by_label"] = distributions
                groups[f"{mode}/{key}"] = result
        return {"version": 2, "alert_unit": "bin", "groups": groups}

    def log_metrics(self, model, prefix, *, emit_to_model=True):
        logged = {}

        def emit(name, value, **kwargs):
            logged[name] = float(value)
            if emit_to_model:
                model.log(name, value, **kwargs)

        report = self.report(
            getattr(model, "calibration_bundles", None),
            classification=model.hparams.classification,
            forecasting=model.hparams.forecasting,
            partition="source" if prefix == "val" else "target",
        )
        macro = {}
        for group, result in report["groups"].items():
            mode, key = group.split("/", 1)
            base = f"{prefix}/{mode}/{key}"
            for name in ("scored_bins", "normal_hours"):
                emit(f"{base}/{name}", float(result[name]), sync_dist=False)
            for head, metrics in result["heads"].items():
                if not getattr(
                    model.hparams,
                    "classification" if head == "classification" else "forecasting",
                ):
                    continue
                if metrics["ap"] is not None:
                    emit(f"{base}/{head}_ap", metrics["ap"], sync_dist=False)
                    if key.startswith("vehicle/"):
                        macro.setdefault((mode, head), []).append(metrics["ap"])
        means = []
        for (mode, head), values in macro.items():
            mean = float(np.mean(values))
            means.append(mean)
            emit(f"{prefix}/{mode}/{head}_ap_macro", mean, sync_dist=False)
        if len(means) == 4:
            emit(f"{prefix}/source_min_vehicle_macro_ap", min(means), sync_dist=False)
        return logged
