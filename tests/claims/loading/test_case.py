"""Test case."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from claims.loader import load_claim_case


def test_loaded_claim_returns_case():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assert case.root_dir == claim_dir
    assert case.claim.claim_id == "CLAIM-2026-00001"


def test_claim_case_documents_directory():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assert case.documents_dir == (claim_dir / "documents")


def test_claim_case_image_path():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    path = case.image_path("car_front.jpg")

    assert path == (claim_dir / "images" / "car_front.jpg")

    assert path.is_file()


def test_claim_case_document_path():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    path = case.document_path("garage_quote.pdf")

    assert path == (claim_dir / "documents" / "garage_quote.pdf")

    assert path.is_file()


def test_claim_case_root_directory_is_immutable():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    with pytest.raises(FrozenInstanceError):
        case.root_dir = Path("/tmp/other")
