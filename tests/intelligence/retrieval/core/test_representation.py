"""Test representation."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from retrieval.representation import (
    build_claim_retrieval_text,
    build_retrieval_document,
)
from tests.infrastructure.support.retrieval_fakes import claim
from tests.intelligence.retrieval.core._support import (
    assert_representation_allowlist,
    canonical_text,
)
from tests.intelligence.retrieval.core._support import components as components


def test_canonical_representation_is_stable_and_allowlisted():
    """
    The input type cannot carry identifiers or model outputs at all.
    """
    current = claim()
    expected = canonical_text()
    assert build_claim_retrieval_text(current.facts) == expected
    reordered = replace(
        current.facts,
        declared_damage=tuple(reversed(current.facts.declared_damage)),
        repair_amount=Decimal("3160.000000"),
        location=" Bordeaux\n ",
    )
    assert build_claim_retrieval_text(reordered) == expected
    changed_identity = replace(
        current,
        claim_id="PRIVATE-ID",
        point_id=claim(9).point_id,
        incident_date=current.incident_date - timedelta(days=10),
    )
    assert build_retrieval_document(changed_identity).canonical_text == expected
    document = build_retrieval_document(current)
    assert document.representation_version == "claim_retrieval_v1"
    assert document.point_id == current.point_id
    assert_representation_allowlist(current, expected)
