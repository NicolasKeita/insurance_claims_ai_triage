"""Test claims."""

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from claims.loader import load_claim_case
from persistence.mappers import (
    claim_to_row,
    row_to_claim,
)
from persistence.models import CustomerRow, PolicyRow
from tests.domain.persistence.mapping._support import reference_claim
from tests.domain.persistence.mapping._support import reference_claim as reference_claim


def test_claim_mapping_round_trip_preserves_domain(reference_claim):
    row = claim_to_row(
        reference_claim,
        CustomerRow(customer_id=reference_claim.customer.customer_id),
        PolicyRow(policy_id=reference_claim.policy.policy_id, product=reference_claim.policy.product),
    )
    assert row_to_claim(row) == reference_claim
    assert isinstance(row.repair_estimate_amount, Decimal)
    assert len(row.documents) == 4
    assert len(row.images) == 2
    assert len(row.damages) == 2
    assert row.documents[-1].filename is None
    assert "id" not in type(row_to_claim(row)).model_fields


def test_loader_preserves_decimal_json_precision(tmp_path, reference_claim):
    payload = reference_claim.model_dump(mode="json")
    payload.update(documents=[], images=[])
    text = json.dumps(payload).replace('"3160"', "9007199254740993.12345678")
    (tmp_path / "claim.json").write_text(text, encoding="utf-8")
    assert load_claim_case(tmp_path).claim.repair_estimate.amount == Decimal("9007199254740993.12345678")


def test_duplicate_damage_rejected_before_writes(reference_claim):
    claim = reference_claim.model_copy(update={"declared_damage": reference_claim.declared_damage * 2})
    with pytest.raises(ValueError, match="only once"):
        claim_to_row(claim, CustomerRow(customer_id="c"), PolicyRow(policy_id="p", product="a"))


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity"])
def test_invalid_financial_values_rejected(reference_claim, amount):
    payload = reference_claim.model_dump()
    payload["repair_estimate"]["amount"] = amount
    with pytest.raises(ValidationError):
        type(reference_claim).model_validate(payload)
