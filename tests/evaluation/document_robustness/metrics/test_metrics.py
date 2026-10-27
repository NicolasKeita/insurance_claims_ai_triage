"""Small deterministic checks of OCR and business metric denominators."""

import pytest

from evaluation.deterministic.document_robustness.metrics import (
    compare_baseline,
    edit_distance,
    normalize_text,
    text_metrics,
)


@pytest.mark.parametrize(
    "reference,actual,expected", [("kitten", "sitting", 3), ("", "", 0), ("", "abc", 3), ("abc", "", 3)]
)
def test_character_edit_distance(reference, actual, expected):
    assert edit_distance(reference, actual) == expected


def test_text_rates_preserve_ocr_sensitive_values():
    assert normalize_text("  1 200.00\r\n2026-09-20\tO I ") == "1 200.00 2026-09-20 O I"
    result = text_metrics("one two three", "one four")
    assert result["word_errors"] == 2 and result["reference_words"] == 3
    assert result["wer"] == 2 / 3
    assert text_metrics("AB", "Ab")["cer"] == 0.5
    assert text_metrics("", "abc")["cer"] is None
    assert text_metrics("", "")["wer"] == 0


def test_baseline_reports_actual_declines_and_new_document_failures():
    baseline = {
        "benchmark_version": "document_ocr_v1",
        "metrics": {
            "critical_field_accuracy": 1.0,
            "parse_failure_count": 0,
            "consistency_preservation_rate": 1.0,
        },
        "documents": [{"document_id": "d1", "variant": "scan_clean", "parse_success": True}],
    }
    current = {
        "benchmark_version": "document_ocr_v1",
        "metrics": {
            "critical_field_accuracy": 0.9,
            "parse_failure_count": 1,
            "consistency_preservation_rate": 0.8,
        },
        "documents": [{"document_id": "d1", "variant": "scan_clean", "parse_success": False}],
    }
    assert len(compare_baseline(baseline, current)) == 4
    assert compare_baseline(baseline, baseline) == []
    with pytest.raises(ValueError, match="version mismatch"):
        compare_baseline(baseline, {"benchmark_version": "future"})
