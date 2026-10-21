"""Test consistency."""

import json

import pytest

from claims.enums import DamageType
from investigation import (
    SignalCode,
)
from investigation.signals import (
    signals_from_consistency,
)
from tests.investigation.policy._support import make_consistency


def test_perfect_consistency_has_no_signals():
    assert signals_from_consistency(make_consistency()) == ()


@pytest.mark.parametrize(
    "check, field",
    [
        ("claim_id_match", "claim_id"),
        ("incident_date_match", "incident_date"),
        ("location_match", "location"),
        ("vehicle_match", "vehicle"),
        ("collision_type_match", "collision_type"),
        ("injuries_match", "injuries_declared"),
        ("declared_damage_match", "declared_damage"),
    ],
)
def test_each_consistency_check_has_an_explicit_field_signal(check, field):
    signals = signals_from_consistency(make_consistency(**{check: False}))
    assert len(signals) == 1
    assert signals[0].code == SignalCode.DOCUMENT_FIELD_MISMATCH
    assert signals[0].evidence["field"] == field


def test_each_document_mismatch_keeps_its_field_in_evidence():
    signals = signals_from_consistency(make_consistency(incident_date_match=False, vehicle_match=False))
    mismatches = [s for s in signals if s.code == SignalCode.DOCUMENT_FIELD_MISMATCH]
    assert {s.evidence["field"] for s in mismatches} == {
        "incident_date",
        "vehicle",
    }
    assert all(s.evidence["matched"] is False for s in mismatches)
    assert all(s.explanation for s in mismatches)


def test_quote_damage_and_unmapped_descriptions_are_json_evidence():
    signals = signals_from_consistency(
        make_consistency(
            additional_quote_damage=(DamageType.HOOD,),
            unmapped_quote_items=("Paint labor", "Diagnostic scan"),
        )
    )
    by_code = {s.code: s for s in signals}
    assert by_code[SignalCode.ADDITIONAL_QUOTE_DAMAGE].evidence["damage_types"] == ["HOOD"]
    assert set(by_code[SignalCode.UNMAPPED_QUOTE_ITEM].evidence["descriptions"]) == {
        "Paint labor",
        "Diagnostic scan",
    }
    json.dumps([s.model_dump(mode="json") for s in signals])


def test_repeated_unmapped_quote_items_keep_each_observed_occurrence():
    signals = signals_from_consistency(make_consistency(unmapped_quote_items=("Paint labor", "Paint labor")))
    assert len(signals) == 1
    assert signals[0].code == SignalCode.UNMAPPED_QUOTE_ITEM
    assert signals[0].evidence["descriptions"] == ["Paint labor", "Paint labor"]
    assert signals[0].evidence["count"] == 2
