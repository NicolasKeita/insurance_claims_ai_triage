"""Aggregation retains failed attempts and exposes conditional consistency coverage."""

import pytest

from evaluation.deterministic.document_robustness.comparison import compare_fields
from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.metrics import text_metrics
from evaluation.deterministic.document_robustness.reporting.aggregation import aggregate, critical_fields
from evaluation.deterministic.document_robustness.runtime.consistency import consistency_results


def _record(reference, actual, status="executed"):
    return {
        "scenario_id": reference.document_id.removesuffix("_" + reference.document_type),
        "document_id": reference.document_id,
        "document_type": reference.document_type,
        "variant": "scan_clean",
        "status": status,
        "extraction_method": "OCR",
        "structured": actual.model_dump(mode="json") if actual else None,
        "structured_metrics": compare_fields(reference.expected, actual),
        "text_metrics": None,
        "native_duration_seconds": 1,
        "ocr_duration_seconds": 2,
        "structured_duration_seconds": 3,
        "duration_seconds": 6,
    }


def _mixed_records():
    form, _, quote = build_corpus()[0].documents
    wrong_id = form.expected.model_copy(update={"claim_id": "DEMO-OCR-999"})
    first, second = _record(form, wrong_id), _record(quote, quote.expected)
    first["text_metrics"] = text_metrics("a", "b")
    second["text_metrics"] = text_metrics("123456789", "123456789")
    second["extraction_method"] = "NATIVE"
    return [first, second, _record(form, None, "failed"), _record(form, None, "not_executed")]


def test_aggregation_weights_counts_and_includes_attempted_parse_failures():
    metrics = aggregate(_mixed_records())
    assert metrics["evaluated_count"] == 3
    assert metrics["not_executed_count"] == metrics["parse_failure_count"] == 1
    assert metrics["document_parse_success_rate"] == 2 / 3
    assert metrics["field_count"] == 25
    assert metrics["field_exact_match_rate"] == metrics["required_field_recall"] == 15 / 25
    assert metrics["required_field_presence_rate"] == 16 / 25
    assert metrics["critical_field_accuracy"] == 5 / 9
    assert metrics["unexpected_value_count"] == 1
    assert metrics["cer"] == 1 / 10
    assert metrics["ocr_cer"] == 1
    assert metrics["text_evaluated_count"] == 2
    assert metrics["line_items"]["expected_count"] == metrics["line_items"]["matching_items"] == 2


def test_critical_categories_count_missing_values_without_counting_unexecuted_cases():
    fields = critical_fields(_mixed_records())
    assert fields["claim_id"] == {"matched": 1, "expected": 3, "accuracy": 1 / 3}
    assert fields["incident_date"]["accuracy"] == fields["damages"]["accuracy"] == 0.5
    assert fields["total.amount"]["accuracy"] == fields["total.currency"]["accuracy"] == 1


@pytest.mark.parametrize("index", [0, 2, 3])
def test_consistency_preserves_coherent_extra_hood_and_controlled_date_mismatch(index):
    scenario = build_corpus()[index]
    records = [_record(doc, doc.expected) for doc in scenario.documents]
    result = consistency_results((scenario,), records)[0]
    assert result["evaluable"] and result["preserved"]
    assert result["actual"] == scenario.expected_consistency


def test_consistency_excludes_incomplete_and_incorrectly_extracted_dossiers():
    scenario = build_corpus()[0]
    first = _record(scenario.documents[0], scenario.documents[0].expected)
    result = consistency_results((scenario,), [first])[0]
    assert result["evaluable"] is False and result["preserved"] is None
    records = [_record(doc, doc.expected) for doc in scenario.documents]
    records[1]["structured_metrics"]["exact_match_count"] -= 1
    result = consistency_results((scenario,), records)[0]
    assert result["evaluable"] is False and result["actual"] is None


def test_empty_aggregate_does_not_invent_success_rates():
    metrics = aggregate([])
    assert metrics["document_parse_success_rate"] is None
    assert metrics["field_exact_match_rate"] is metrics["critical_field_accuracy"] is None
    assert metrics["cer"] is metrics["ocr_cer"] is None


def test_unmeasured_failure_durations_do_not_become_zero_time_observations():
    records = _mixed_records()
    for key in ("native_duration_seconds", "ocr_duration_seconds", "structured_duration_seconds"):
        records[2][key] = None
    records[1]["ocr_duration_seconds"] = 0
    metrics = aggregate(records)
    assert metrics["mean_native_duration_seconds"] == 1
    assert metrics["mean_ocr_duration_seconds"] == 1
    assert metrics["mean_structured_duration_seconds"] == 3
    assert metrics["mean_duration_seconds"] == 6


def test_hybrid_document_text_does_not_dilute_pure_ocr_rates():
    records = _mixed_records()[:2]
    records[1]["extraction_method"] = "HYBRID"
    metrics = aggregate(records)
    assert metrics["cer"] == 0.1
    assert metrics["ocr_cer"] == metrics["ocr_text_evaluated_count"] == 1
    assert metrics["hybrid_cer"] == 0
    assert metrics["hybrid_text_evaluated_count"] == 1
