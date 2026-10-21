"""Test provenance."""

from pathlib import Path

from tests.intelligence.vision.training.tracking._support import tracker_stub, training_summary


def test_training_tracking_preserves_frozen_provenance_and_validation_classes(tmp_path):
    tracker, params, tags, metrics, artifacts = tracker_stub()
    checkpoint = tmp_path / "train/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    results_csv = checkpoint.parent.parent / "results.csv"
    results_csv.write_text("epoch,loss\n1,0.1\n")
    summary_path = tmp_path / "training_summary.json"
    summary = training_summary()
    tracker.log_training(summary, summary_path, checkpoint)
    assert params["dataset.source_repository"] == "harpreetsahota/CarDD"
    assert params["dataset.split_version"] == 2
    assert params["dataset.actual_sample_count"] == 2816
    assert params["dataset.train.sha256"] == "d" * 64
    assert params["dataset.train.glass shatter.source_images"] == 328
    assert params["training.best_epoch"] == 74
    assert tags["checkpoint_reference"] == "train/weights/best.pt"
    assert tags["registry_status"] == "deferred"
    assert metrics["validation.class.glass_shatter.mask.map50_95"] == 0.2
    assert "validation.class.dent.mask.map50_95" not in metrics
    assert metrics["training.epochs_completed"] == 94
    assert [Path(path).suffix for path, _ in artifacts] == [".json", ".csv"]
