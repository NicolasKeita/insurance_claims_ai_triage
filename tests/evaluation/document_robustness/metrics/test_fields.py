"""Strict business field comparisons use the product's validated schemas."""

from datetime import date
from decimal import Decimal

from claims.document_models import ClaimFormExtraction, DocumentSource, GarageQuoteExtraction
from claims.enums import DocumentType
from evaluation.deterministic.document_robustness.comparison import compare_fields, line_item_metrics


def _form():
    return ClaimFormExtraction(
        claim_id="DEMO-32",
        incident_date=date(2026, 9, 20),
        location="Demo Lane",
        vehicle_make="Demo",
        vehicle_model="Car",
        vehicle_year=2024,
        collision_type="FRONT_COLLISION",
        declared_damage=("FRONT_BUMPER", "HOOD"),
        injuries_declared=False,
        source=DocumentSource(document_type=DocumentType.CLAIM_FORM, filename="demo.pdf", page=1),
    )


def test_field_comparison_uses_damage_sets_and_excludes_source():
    expected = _form()
    actual = expected.model_copy(update={"declared_damage": tuple(reversed(expected.declared_damage))})
    result = compare_fields(expected, actual)
    assert result["exact_match_count"] == result["field_count"] == 9
    assert result["critical_match_count"] == result["critical_field_count"] == 3
    assert result["required_present_count"] == 9
    assert all(field["field"] != "source" for field in result["fields"])


def test_changed_valid_date_and_id_are_visible_critical_errors():
    expected = _form()
    actual = expected.model_copy(update={"claim_id": "DEMO-38", "incident_date": date(2026, 9, 28)})
    result = compare_fields(expected, actual)
    assert result["critical_match_count"] == 1
    assert result["required_match_count"] == 7
    assert result["required_present_count"] == 9
    assert result["unexpected_value_count"] == 2


def test_parse_failure_counts_all_required_and_critical_fields_missing():
    result = compare_fields(_form(), None)
    assert result["parse_success"] is False
    assert result["exact_match_count"] == result["critical_match_count"] == result["required_present_count"] == 0
    assert result["unexpected_value_count"] == 0


def _item(description, amount="1200.00", currency="EUR"):
    return {"description": description, "price": {"amount": Decimal(amount), "currency": currency}}


def test_line_matching_is_exact_and_preserves_duplicates():
    expected = [_item("hood"), _item("hood"), _item("front bumper")]
    actual = [_item("hood", "1200"), _item("Hood"), _item("front bumper", "1200.01")]
    assert line_item_metrics(expected, actual) == {
        "expected_count": 3,
        "extracted_count": 3,
        "matching_items": 1,
        "missing_items": 2,
        "unexpected_items": 2,
    }


def test_decimal_numeric_values_match_without_string_rounding():
    quote = GarageQuoteExtraction(
        garage="Demo",
        claim_id="DEMO-32",
        parts=(_item("hood"),),
        labor={"amount": "0", "currency": "EUR"},
        total={"amount": "1200.00", "currency": "EUR"},
        source=DocumentSource(document_type=DocumentType.REPAIR_QUOTE, filename="quote.pdf", page=1),
    )
    actual = quote.model_copy(update={"total": quote.total.model_copy(update={"amount": Decimal("1200")})})
    result = compare_fields(quote, actual)
    assert result["exact_match_count"] == result["field_count"]
    assert result["line_items"]["matching_items"] == 1
