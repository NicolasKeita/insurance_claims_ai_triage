"""Fixed-threshold counting must preserve empty images and class-aware matches."""

from types import SimpleNamespace

import numpy as np
import pytest

from vision.damage.validation._operating import OperatingPointCollector


def _validator(**changes):
    stats = {
        "tp": [np.zeros((0, 1))],
        "tp_m": [np.zeros((0, 1))],
        "pred_cls": [np.array([])],
        "target_cls": [np.array([0])],
        "conf": [np.array([])],
    }
    stats.update(changes)
    return SimpleNamespace(iouv=np.array([0.5]), metrics=SimpleNamespace(stats=stats))


def test_missed_ground_truth_has_zero_recall_and_undefined_precision():
    collector = OperatingPointCollector([{"yolo_class_id": 0, "name": "crack"}])
    validator = _validator()
    collector(validator)
    collector(validator)
    report = collector.report(1)
    assert report["mask"] == {"tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0.0}


def test_missing_statistics_fail_instead_of_reporting_synthetic_counts():
    collector = OperatingPointCollector([])
    with pytest.raises(RuntimeError, match="statistics unavailable"):
        collector(SimpleNamespace())


def test_wrong_matching_iou_is_refused():
    collector = OperatingPointCollector([])
    validator = _validator()
    validator.iouv = np.array([0.75])
    with pytest.raises(RuntimeError, match="IoU 0.50"):
        collector(validator)


def test_incomplete_validation_coverage_is_refused():
    collector = OperatingPointCollector([{"yolo_class_id": 0, "name": "crack"}])
    collector(_validator())
    with pytest.raises(RuntimeError, match="coverage differs"):
        collector.report(2)


def test_unknown_prediction_classes_are_refused():
    collector = OperatingPointCollector([{"yolo_class_id": 0, "name": "crack"}])
    validator = _validator(pred_cls=[np.array([3])], tp=[np.array([[True]])], tp_m=[np.array([[True]])])
    with pytest.raises(RuntimeError, match="unknown class"):
        collector(validator)
