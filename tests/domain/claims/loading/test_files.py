"""Test files."""

from pathlib import Path

import pytest

from claims.loader import load_claim_case
from claims.validation import ClaimFileValidationError


def test_missing_document_file_is_rejected():
    claim_dir = Path("tests/data/missing_document")

    with pytest.raises(ClaimFileValidationError):
        load_claim_case(claim_dir)


def test_missing_image_file_is_rejected():
    claim_dir = Path("tests/data/missing_image")

    with pytest.raises(ClaimFileValidationError):
        load_claim_case(claim_dir)


def test_claim_with_all_required_files_is_valid():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)
    claim = case.claim

    assert claim.claim_id == "CLAIM-2026-00001"
