"""Test metrics."""

from tests.intelligence.vision.training.tracking._support import metric_fixture, tracker_stub


def test_test_tracking_logs_real_per_class_values_without_missing_class_imputation(tmp_path):
    tracker, _, tags, metrics, artifacts = tracker_stub()
    tracker.log_evaluation(metric_fixture(), [tmp_path / "test_metrics.json", tmp_path / "per_class_metrics.csv"])
    assert metrics["test.mask.map50_95"] == 0.3
    assert metrics["test.class.glass_shatter.mask.map50_95"] == 0.2
    assert "test.class.dent.mask.map50_95" not in metrics
    assert tags["final_evaluation_split"] == "test"
    assert len(artifacts) == 2
