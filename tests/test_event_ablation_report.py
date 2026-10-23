import json

import pytest
import yaml

from scripts.event_ablation_report import suite_report


def write_arm(root, name, *, finished=True, cover=10):
    cfg = dict(
        experiment_name=name,
        dataset="fixture",
        config=dict(
            model=dict(payload_weight=0 if name == "classification" else 1),
            trainer=dict(max_epochs=2),
        ),
    )
    path = root / f"{name}.yml"
    path.write_text(yaml.safe_dump(cfg))
    run = root / "runs/fixture" / name / name
    (run / ".graphids").mkdir(parents=True)
    (run / ".graphids/manifest.json").write_text(
        json.dumps(dict(status="finished" if finished else "failed"))
    )
    for epoch, ap in enumerate([0.4, 0.2]):
        index = run / "artifacts" / f"val-events-epoch{epoch:04d}" / "index.json"
        index.parent.mkdir(parents=True)
        index.write_text(
            json.dumps(
                dict(
                    prediction_unit="observed_frame",
                    coverage=dict(recording=cover),
                    by_recording=dict(recording=dict(classification_frame_ap=ap)),
                    metrics=dict(
                        classification_frame_ap=ap,
                        classification_frames=cover,
                        classification_positives=1,
                        classification_negatives=cover - 1,
                    ),
                )
            )
        )
    return dict(name=name, config=str(path), job_id=name)


@pytest.mark.parametrize("mismatch", [False, True])
def test_report_requires_matched_coverage_and_selects_one_epoch_for_all_groups(
    tmp_path, mismatch
):
    jobs = [
        write_arm(tmp_path, "joint"),
        write_arm(tmp_path, "classification", cover=11 if mismatch else 10),
    ]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(dict(root=str(tmp_path), training_jobs=jobs)))
    report = suite_report(path)
    assert report["status"] == ("incomplete" if mismatch else "complete")
    assert report["arms"][0]["best"]["epoch"] == 0
    assert report["arms"][0]["final"]["epoch"] == 1
    assert report["arms"][1]["payload_trained"] is False
    assert bool(report["errors"]) == mismatch


def test_failed_fit_with_all_epoch_indices_is_incomplete(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            dict(
                root=str(tmp_path),
                training_jobs=[write_arm(tmp_path, "joint", finished=False)],
            )
        )
    )
    assert suite_report(path)["status"] == "incomplete"


def test_missing_epoch_cannot_be_reported_as_complete(tmp_path):
    job = write_arm(tmp_path, "joint")
    (
        tmp_path / "runs/fixture/joint/joint/artifacts/val-events-epoch0000/index.json"
    ).unlink()
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(dict(root=str(tmp_path), training_jobs=[job])))
    assert suite_report(path)["status"] == "incomplete"
