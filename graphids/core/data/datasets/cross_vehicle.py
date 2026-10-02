"""Leave-one-vehicle-out can-train-and-test source with explicit provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import polars as pl
from filelock import FileLock

from graphids._fs import atomic_save, atomic_write_text
from graphids.core.data.datasets.can_bus import (
    ATTACK_TYPE_NAMES,
    CAN_TRAIN_TEST_VEHICLES,
    load_can_rows,
)
from graphids.core.data.datasets.recording_split import recording_manifest
from graphids.core.data.preprocessing.representations import (
    TemporalRepresentationCfg,
    representation_digest,
)
from graphids.core.data.preprocessing.temporal import (
    add_temporal_split_masks,
    build_temporal_event_table,
    temporal_to_pyg,
)
from graphids.core.data.preprocessing.vocab import persist_vocab
from graphids.core.data.state import DatasetState

HOLDOUTS = {
    "chevrolet_silverado": ("set_01", "set_03"),
    "subaru_forester": ("set_02", "set_04"),
    "chevrolet_traverse": ("set_04", "set_02"),
}
LOCAL_ID_CAPACITY = 8192


def assign_local_ids(rows: pl.DataFrame) -> pl.DataFrame:
    """Assign IDs on first occurrence within each capture, without collisions.

    The offline join implements the same mapping a streaming detector obtains by
    incrementing a counter whenever a new arbitration ID first appears.
    """
    keys = ["vehicle_id", "source_file", "arb_id"]
    first_seen = rows.select(keys).unique(subset=keys, maintain_order=True)
    first_seen = first_seen.with_columns(
        pl.cum_count("arb_id").over(["vehicle_id", "source_file"])
        .cast(pl.Int64).alias("node_id")
    )
    max_id = first_seen["node_id"].max()
    if max_id is not None and max_id > LOCAL_ID_CAPACITY:
        raise ValueError(f"capture has {max_id} IDs; local capacity is {LOCAL_ID_CAPACITY}")
    return rows.join(first_seen, on=keys, how="left", maintain_order="left")


def _recording_hashes(root: Path, files: list[str]) -> set[str]:
    result = set()
    for name in files:
        with (root / name).open("rb") as handle:
            result.add(hashlib.file_digest(handle, "sha256").hexdigest())
    return result


@dataclass(frozen=True)
class CrossVehicleTemporalSource:
    """Train on three verified vehicles; reserve another for testing.

    The target normal split is published separately and never enters ``train``
    or ``val``. Its use in calibration must be declared by the evaluator.
    """

    KIND: ClassVar[str] = "cross_vehicle"
    held_out_vehicle: str
    source_vehicle: str | None = None
    lake_root: str | None = None
    representation_cfg: TemporalRepresentationCfg = field(default_factory=TemporalRepresentationCfg)
    val_fraction: float = 0.2
    split_seed: int = 42
    val_warmup_events: int = 64
    test_warmup_events: int = 64

    def __post_init__(self) -> None:
        if self.held_out_vehicle not in HOLDOUTS:
            raise ValueError(f"held_out_vehicle must be one of {sorted(HOLDOUTS)}")
        known = {pair[0] for pair in CAN_TRAIN_TEST_VEHICLES.values()}
        if self.source_vehicle is not None and (
            self.source_vehicle not in known or self.source_vehicle == self.held_out_vehicle
        ):
            raise ValueError("source_vehicle must be one known vehicle distinct from the holdout")
        if not 0 < self.val_fraction < 1:
            raise ValueError("cross-vehicle validation requires 0 < val_fraction < 1")

    @property
    def attack_type_names(self) -> dict[int, str]:
        return dict(ATTACK_TYPE_NAMES)

    @property
    def num_ids(self) -> int | None:
        return LOCAL_ID_CAPACITY + 1 if self.representation_cfg.features == "behavioral" else None

    @property
    def cache_key(self) -> str:
        return str(self.cache_root_path())

    def resolved_lake_root(self) -> str:
        if self.lake_root:
            return self.lake_root
        from graphids.paths import lake_root
        return lake_root()

    def cache_root_path(self) -> Path:
        from graphids.paths import cache_dir
        return (cache_dir(self.resolved_lake_root(), "cross_vehicle") /
                f"v2_{self.held_out_vehicle}_from_{self.source_vehicle or 'pooled'}_"
                f"{representation_digest(self.representation_cfg)}_"
                f"seed{self.split_seed}_val{self.val_fraction:g}_"
                f"warm{self.val_warmup_events}_{self.test_warmup_events}")

    def cache_ready(self) -> bool:
        root = self.cache_root_path()
        expected = ["data_train.pt", "data_val.pt", "data_target_calibration.pt",
                    "data_target_normal_test.pt",
                    "data_target_known_attack.pt", "data_target_unknown_attack.pt"]
        return ((root / "processed" / ".complete").is_file()
                and (root / "split_manifest.json").is_file()
                and (root / "vocab.json").is_file()
                and all((root / "processed" / name).is_file() for name in expected))

    def build(self) -> DatasetState:
        import torch

        from graphids.paths import data_dir, load_catalog

        root = self.cache_root_path()
        processed = root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        with FileLock(str(processed / ".lock")):
            if not self.cache_ready():
                catalog = load_catalog()
                raw_lake = self.resolved_lake_root()
                target_set, calibration_set = HOLDOUTS[self.held_out_vehicle]
                source_sets = [name for name, pair in CAN_TRAIN_TEST_VEHICLES.items()
                               if pair[0] != self.held_out_vehicle and
                               (self.source_vehicle is None or pair[0] == self.source_vehicle)]
                manifest = {"version": 1, "held_out_vehicle": self.held_out_vehicle,
                            "source_vehicle": self.source_vehicle,
                            "source_sets": source_sets, "target_set": target_set,
                            "calibration_set": calibration_set, "source_manifests": {}}
                train_frames, val_frames = [], []
                source_paths: list[tuple[Path, list[str]]] = []
                for set_name in source_sets:
                    raw = data_dir(raw_lake, set_name)
                    split = recording_manifest(
                        raw, catalog[set_name], fraction=self.val_fraction,
                        seed=self.split_seed, warmup=self.val_warmup_events,
                        train_source_mode="mixed",
                    )
                    manifest["source_manifests"][set_name] = split
                    for partition, destination in (("train", train_frames), ("val", val_frames)):
                        files = [r["source_file"] for r in split["recordings"] if r["split"] == partition]
                        dirs = sorted({Path(name).parts[0] for name in files})
                        frame = load_can_rows(raw, dirs, source_files=files, preserve_id_text=True).with_columns(
                            pl.concat_str([pl.lit(set_name), pl.col("source_file")], separator="/").alias("source_file")
                        )
                        destination.append(frame)
                        source_paths.append((raw, files))
                target_raw = data_dir(raw_lake, target_set)
                calibration_raw = data_dir(raw_lake, calibration_set)
                target_dirs = [catalog[target_set]["test_subdirs"][i] for i in (1, 3)]
                calibration_dir = catalog[calibration_set]["train_subdir"]
                normal_files = sorted(str(p.relative_to(calibration_raw)) for p in
                                           (calibration_raw / calibration_dir).rglob("*.csv"))
                target_files = sorted(str(p.relative_to(target_raw)) for directory in target_dirs
                                      for p in (target_raw / directory).rglob("*.csv"))
                if len(normal_files) < 2 or not target_files:
                    raise ValueError("target normal or test recordings are missing")
                normal_files.sort(key=lambda name: hashlib.sha256(name.encode()).hexdigest())
                midpoint = len(normal_files) // 2
                calibration_files, normal_test_files = normal_files[:midpoint], normal_files[midpoint:]
                source_hashes = set().union(*(_recording_hashes(path, files) for path, files in source_paths))
                calibration_hashes = _recording_hashes(calibration_raw, calibration_files)
                normal_test_hashes = _recording_hashes(calibration_raw, normal_test_files)
                target_hashes = _recording_hashes(target_raw, target_files)
                if (source_hashes & (calibration_hashes | normal_test_hashes | target_hashes)
                        or calibration_hashes & (normal_test_hashes | target_hashes)
                        or normal_test_hashes & target_hashes):
                    raise ValueError("recording content overlaps across training, calibration, and target test")
                manifest["calibration_files"] = calibration_files
                manifest["normal_test_files"] = normal_test_files
                manifest["target_files"] = target_files
                manifest["digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

                train_rows = pl.concat(train_frames, how="vertical")
                val_rows = pl.concat(val_frames, how="vertical")
                # Raw models use a namespace; target IDs remain genuinely unknown.
                keys = train_rows.select(pl.concat_str(["vehicle_id", "arb_id"], separator="::").alias("key"))
                vocab = {token: i + 1 for i, token in enumerate(sorted(keys["key"].unique().to_list()))}
                persist_vocab(vocab, root / "vocab.json")

                def pack(rows: pl.DataFrame, split_name: str, warmup: int):
                    # Keep subsecond timing before conversion to TemporalData's float32 t.
                    rows = rows.with_columns((
                        pl.col("timestamp") - pl.col("timestamp").min().over(["vehicle_id", "source_file"])
                    ).alias("timestamp"))
                    keys = pl.concat_str([pl.col("vehicle_id"), pl.col("arb_id").cast(pl.Utf8)], separator="::")
                    if self.representation_cfg.features == "behavioral":
                        rows = assign_local_ids(rows)
                    else:
                        rows = rows.with_columns(keys.replace_strict(vocab, default=0).cast(pl.Int64).alias("node_id"))
                    table = add_temporal_split_masks(build_temporal_event_table(rows),
                                                     split_name=split_name, warmup_events=warmup)
                    return temporal_to_pyg(table, features=self.representation_cfg.features)

                target_calibration = load_can_rows(calibration_raw, [calibration_dir], source_files=calibration_files,
                                                   preserve_id_text=True)
                if target_calibration["vehicle_id"].unique().to_list() != [self.held_out_vehicle]:
                    raise ValueError("calibration vehicle identity differs from held-out vehicle")
                partitions = {
                    "train": pack(train_rows, "train", 0),
                    "val": pack(val_rows, "val", self.val_warmup_events),
                    "target_calibration": pack(target_calibration, "test", self.test_warmup_events),
                    "target_normal_test": pack(load_can_rows(calibration_raw, [calibration_dir], source_files=normal_test_files,
                                                        preserve_id_text=True), "test", self.test_warmup_events),
                    "target_known_attack": pack(load_can_rows(target_raw, [target_dirs[0]], preserve_id_text=True),
                                                "test", self.test_warmup_events),
                    "target_unknown_attack": pack(load_can_rows(target_raw, [target_dirs[1]], preserve_id_text=True),
                                                  "test", self.test_warmup_events),
                }
                offset = 0
                for name, data in partitions.items():
                    data.event_id += offset
                    offset += len(data.event_id)
                    atomic_save(data, processed / f"data_{name}.pt")
                atomic_write_text(root / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")
                atomic_write_text(processed / ".complete", "ok")
        def read(name):
            return torch.load(processed / f"data_{name}.pt", map_location="cpu", mmap=True, weights_only=False)
        return DatasetState(train=read("train"), val=read("val"), test={
            name: read(name) for name in ("target_calibration", "target_normal_test",
                                           "target_known_attack", "target_unknown_attack")
        })
