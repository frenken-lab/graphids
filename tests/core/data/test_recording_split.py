"""Recording holdout leakage and attack-coverage contracts."""

from __future__ import annotations

import csv
import json
import shutil

import pytest
import torch

import graphids.primitives_data as primitives_data
from graphids.core.data.datasets.can_bus import CANBusTemporalSource
from graphids.core.data.datasets.recording_split import recording_manifest


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    entry = {
        "train_subdir": "clean",
        "train_attack_subdir": "mixed",
        "test_subdirs": ["test"],
    }

    def write(name, unique_id, labels):
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "arb_id", "data_field", "attack"])
            writer.writerows(
                [i, "shared" if i == 0 else unique_id, "0102030405060708", label]
                for i, label in enumerate(labels)
            )

    for i in range(2):
        write(f"clean/normal_{i}.csv", f"normal_{i}", [0, 0, 0, 0])
        write(f"mixed/dos_{i}.csv", f"attack_{i}", [0, 1, 1, 0])
    (raw / "mixed").mkdir(exist_ok=True)
    shutil.copy(raw / "clean/normal_0.csv", raw / "mixed/normal_copy.csv")
    write("test/dos.csv", "test_only", [0, 1, 0, 1])
    monkeypatch.setattr("graphids.paths.load_catalog", lambda: {"dummy": entry})
    monkeypatch.setattr(primitives_data, "load_catalog", lambda: {"dummy": entry})
    monkeypatch.setattr("graphids.paths.data_dir", lambda lake, name: raw)
    monkeypatch.setattr(
        "graphids.paths.cache_dir", lambda lake, name: tmp_path / "cache"
    )
    return raw, entry


def test_holdout_deduplicates_and_excludes_validation_and_test_from_vocab(recordings):
    raw, _ = recordings
    source = CANBusTemporalSource(
        name="dummy", lake_root="lake", val_split="recordings", val_warmup_events=1
    )
    state = source.build()
    root = source.cache_root_path()
    manifest = json.loads((root / "split_manifest.json").read_text())
    vocab = json.loads((root / "vocab.json").read_text())["entries"]
    assert len(manifest["recordings"]) == 4
    assert any(len(r["aliases"]) == 2 for r in manifest["recordings"])
    assert state.train.num_events + state.val.num_events == 16
    assert state.train.num_events == state.val.num_events == 8
    assert torch.bincount(state.train.y, minlength=2).tolist() == [6, 2]
    assert torch.bincount(state.val.y[state.val.is_scored], minlength=2).tolist() == [
        4,
        2,
    ]
    assert int(state.val.is_warmup.sum()) == 2
    assert set(state.train.event_id.tolist()).isdisjoint(state.val.event_id.tolist())
    assert state.val.reset_after.tolist() == [False, False, False, True] * 2
    assert state.val.src[0] == state.val.dst[0]
    assert state.val.msg[0, 8:17].eq(0).all()
    assert state.val.dst.reshape(2, 4)[:, 1:].eq(0).all()
    assert "test_only" not in vocab
    for record in manifest["recordings"]:
        if record["split"] == "val":
            with (raw / record["source_file"]).open() as handle:
                unique_id = list(csv.DictReader(handle))[1]["arb_id"]
            assert unique_id not in vocab
    assert source.cache_ready()
    assert torch.equal(source.build().val.y, state.val.y)


def test_manifest_is_deterministic_and_independent_of_test_contents(recordings):
    raw, entry = recordings
    options = dict(fraction=0.2, seed=42, warmup=1, train_source_mode="mixed")
    first = recording_manifest(raw, entry, **options)
    shutil.rmtree(raw / "test")
    second = recording_manifest(raw, entry, **options)
    assert first == second
    assert all(
        not alias.startswith("test/")
        for r in first["recordings"]
        for alias in r["aliases"]
    )


def test_attack_free_training_uses_same_attack_inclusive_validation(recordings):
    source = CANBusTemporalSource(
        name="dummy",
        lake_root="lake",
        val_split="recordings",
        train_source_mode="attack_free",
        val_warmup_events=1,
    )
    state = source.build()
    assert state.train.y.eq(0).all()
    assert state.train.num_events == 4
    assert torch.bincount(state.val.y[state.val.is_scored], minlength=2).tolist() == [
        4,
        2,
    ]
    mixed = CANBusTemporalSource(
        name="dummy", lake_root="lake", val_split="recordings", val_warmup_events=1
    ).build()
    assert torch.equal(state.val.y, mixed.val.y)


def test_impossible_attack_holdout_fails_without_using_test(recordings):
    raw, entry = recordings
    (raw / "mixed/dos_1.csv").unlink()
    with pytest.raises(ValueError, match="attack-inclusive|no attacked training"):
        recording_manifest(
            raw, entry, fraction=0.2, seed=42, warmup=1, train_source_mode="mixed"
        )


def test_warmup_attacks_cannot_satisfy_holdout_coverage(recordings):
    raw, entry = recordings
    with pytest.raises(ValueError, match="attack-inclusive"):
        recording_manifest(
            raw, entry, fraction=0.2, seed=42, warmup=3, train_source_mode="mixed"
        )


def test_recording_split_has_separate_cache_and_rejects_prefixes(recordings):
    from graphids.core.data.datamodule.temporal import TemporalDataModule
    from graphids.primitives_data import can_bus, temporal_dm

    legacy = CANBusTemporalSource(name="dummy", lake_root="lake")
    fixed = CANBusTemporalSource(name="dummy", lake_root="lake", val_split="recordings")
    other_seed = CANBusTemporalSource(
        name="dummy", lake_root="lake", val_split="recordings", split_seed=43
    )
    assert len({s.cache_root_path() for s in (legacy, fixed, other_seed)}) == 3
    assert len({s.cache_key for s in (legacy, fixed, other_seed)}) == 3
    with pytest.raises(ValueError, match="complete streams"):
        TemporalDataModule(fixed, val_prefix_events=8192)
    cfg = can_bus(dataset="dummy", seed=99, val_split="recordings", split_seed=42)
    dm = temporal_dm(source=cfg).build()
    assert dm.source.val_split == "recordings"
    assert dm.source.split_seed == 42
    with pytest.raises(ValueError, match="complete streams"):
        temporal_dm(source=cfg, val_prefix_events=8192)


def test_set_01_final_configs_and_lane_config_use_complete_recording_validation():
    from pathlib import Path

    from graphids.exp.config import ExperimentConfig
    from graphids.exp.ray_backend import resolve_spec

    paths = list(Path("configs/experiments/final").glob("*/*set_01_train.yml"))
    paths.append(Path("configs/experiments/temporal_set_01_recording_holdout.yml"))
    assert len(paths) == 9
    for path in paths:
        experiment = ExperimentConfig.from_yaml(path)
        run = experiment.build_run(
            name=experiment.experiment_name,
            stage=experiment.stage,
            config=experiment.config,
        )
        config = resolve_spec(run.payload.data)
        assert config.source.val_split == "recordings"
        assert config.source.split_seed == 42
        assert config.val_prefix_events is None
        assert config.val_warmup_events == 64
