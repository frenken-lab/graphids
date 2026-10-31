import json

import yaml

from scripts.event_suite_report import suite_report


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
                    by_recording=dict(
                        recording=dict(
                            classification_frame_ap=ap,
                            classification_frames=cover,
                            classification_positives=1,
                            classification_negatives=cover - 1,
                            payload_frame_ap=0.8 if epoch == 0 else 0.9,
                            payload_frames=cover - 1,
                        )
                    ),
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


def test_report_requires_matched_coverage_and_selects_one_epoch_for_all_groups(
    tmp_path,
):
    jobs = [
        write_arm(tmp_path, "joint"),
        write_arm(tmp_path, "classification"),
    ]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(dict(root=str(tmp_path), training_jobs=jobs)))
    report = suite_report(path)
    assert report["status"] == "complete"
    assert report["arms"][0]["best"]["epoch"] == 0
    assert report["arms"][0]["final"]["epoch"] == 1
    assert report["arms"][1]["payload_trained"] is False
    assert not report["errors"]

    # One arm's otherwise valid run cannot be compared against a different cohort.
    for index in (tmp_path / "runs/fixture/classification").rglob("index.json"):
        payload = json.loads(index.read_text())
        payload["coverage"]["recording"] = 11
        payload["metrics"].update(classification_frames=11, classification_negatives=10)
        index.write_text(json.dumps(payload))
    mismatched = suite_report(path)
    assert mismatched["status"] == "incomplete"
    assert mismatched["errors"]


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


def test_macro_ap_excludes_normal_recordings_and_keeps_aggregate_epoch_selection(
    tmp_path,
):
    job = write_arm(tmp_path, "joint")
    indices = sorted((tmp_path / "runs").rglob("index.json"))
    for epoch, index in enumerate(indices):
        value = json.loads(index.read_text())
        value["coverage"] = dict(strong=4, weak=4, normal=2)
        value["metrics"].update(classification_positives=2, classification_negatives=8)
        value["by_recording"] = {
            "strong": dict(
                classification_positives=1,
                classification_negatives=3,
                classification_frame_ap=0.9,
            ),
            "weak": dict(
                classification_positives=1,
                classification_negatives=3,
                classification_frame_ap=0.1 if epoch == 0 else 0.8,
            ),
            "normal": dict(classification_positives=0, classification_negatives=2),
        }
        index.write_text(json.dumps(value))
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(dict(root=str(tmp_path), training_jobs=[job])))
    report = suite_report(path)
    assert report["status"] == "complete"
    arm = report["arms"][0]
    assert arm["best"]["epoch"] == 0  # Macro AP must not change checkpoint selection.
    assert arm["best"]["recording_macro_ap"] == 0.5
    assert arm["final"]["recording_macro_ap"] > 0.8
    assert arm["best"]["recordings_by_ap"] == ["weak", "strong"]
    assert "normal" in arm["best"]["by_recording"]
    # A missing score for an eligible group must not silently improve the mean.
    value = json.loads(indices[0].read_text())
    value["by_recording"]["weak"].pop("classification_frame_ap")
    indices[0].write_text(json.dumps(value))
    assert suite_report(path)["arms"][0]["status"] == "invalid"


def test_cli_compares_payload_at_the_same_epoch_and_excludes_untrained_scores(
    tmp_path, monkeypatch, capsys
):
    import sys

    from scripts.event_suite_report import main

    jobs = [write_arm(tmp_path, "joint"), write_arm(tmp_path, "classification")]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(dict(root=str(tmp_path), training_jobs=jobs)))
    monkeypatch.setattr(sys, "argv", ["event_suite_report", str(path)])
    main()
    output = capsys.readouterr().out
    assert "classification AP=0.400000, payload AP=0.800000 (9/10 known rows)" in output
    assert "classification AP=0.400000, payload AP=untrained" in output
    assert "payload AP=0.900000" not in output  # No payload-specific epoch selection.


def test_phase_reports_reuse_baseline_and_strict_cli_blocks_incomplete_controls(
    tmp_path,
):
    import subprocess
    import sys

    baseline = dict(write_arm(tmp_path, "baseline"), phases=["controls", "context"])
    control = dict(write_arm(tmp_path, "control", finished=False), phases=["controls"])
    context = dict(write_arm(tmp_path, "context"), phases=["context"])
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(dict(root=str(tmp_path), training_jobs=[baseline, control, context]))
    )
    assert suite_report(path, phase="controls")["status"] == "incomplete"
    assert suite_report(path, phase="context")["status"] == "complete"
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/event_suite_report.py",
            str(path),
            "--phase",
            "controls",
            "--require-complete",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "incomplete"
