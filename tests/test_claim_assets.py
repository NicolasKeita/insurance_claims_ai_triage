from pathlib import Path

from claims.assets import resolve_document_assets
from claims.loader import load_claim_case
from claims.enums import DocumentType
from claims.assets import (
    resolve_document_assets,
    resolve_image_assets,
)


def test_resolve_document_assets():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    assert len(assets) == 3

    filenames = [
        asset.filename
        for asset in assets
    ]

    assert filenames == [
        "claim_form.pdf",
        "accident_report.pdf",
        "garage_quote.pdf",
    ]
    assert "police_report.pdf" not in filenames


def test_repair_quote_asset():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    repair_quote = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    assert repair_quote.filename == (
        "garage_quote.pdf"
    )

    assert repair_quote.path == (
        claim_dir
        / "documents"
        / "garage_quote.pdf"
    )

    assert repair_quote.extension == ".pdf"
    assert repair_quote.media_type == "application/pdf"
    assert repair_quote.path.is_file()
    assert repair_quote.size_bytes >= 0

def test_resolve_image_assets():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_image_assets(case)

    assert len(assets) == 2

    filenames = [
        asset.filename
        for asset in assets
    ]

    assert filenames == [
        "car_front.jpg",
        "car_left.jpg",
    ]

def test_front_image_asset():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_image_assets(case)

    image = next(
        asset
        for asset in assets
        if asset.filename == "car_front.jpg"
    )

    assert image.path == (
        claim_dir
        / "images"
        / "car_front.jpg"
    )

    assert image.extension == ".jpg"
    assert image.media_type == "image/jpeg"
    assert image.path.is_file()