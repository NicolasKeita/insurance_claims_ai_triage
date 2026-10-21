"""Test diagnostics."""

import pytest

from vision.error_analysis import AnalysisConfig, ErrorInstance, analyze_instances


def test_wrong_part_and_incomplete_masks_are_honest_separate_diagnostics():
    full = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    half = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
    wrong = analyze_instances(
        [ErrorInstance(1, (0, 0, 1, 1), 0.9, full)],
        [ErrorInstance(0, (0, 0, 1, 1), polygon=full)],
        config=AnalysisConfig(confidence_threshold=0.25, image_size=(100, 100), include_part_errors=True),
    )
    assert len(wrong["wrong_parts"]) == 1 and wrong["matches"] == []
    incomplete = analyze_instances(
        [ErrorInstance(0, (0, 0, 1, 1), 0.9, half)],
        [ErrorInstance(0, (0, 0, 1, 1), polygon=full)],
        config=AnalysisConfig(confidence_threshold=0.25, image_size=(100, 100), include_part_errors=True),
    )
    assert len(incomplete["incomplete_masks"]) == 1
    assert incomplete["incomplete_masks"][0]["truth_mask_coverage"] == pytest.approx(0.51)
