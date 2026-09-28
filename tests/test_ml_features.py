import pytest

from claims.case import ClaimCase
from ml.features import (
    FeatureEngineeringError,
    calculate_vehicle_age,
)
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.consistency import (
    compare_claim_documents,
)
from claims.document_extraction import (
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import (
    extract_pdf_text,
)
from claims.enums import DocumentType
from claims.loader import load_claim_case

from ml.features import (
    build_triage_features,
)

def test_calculate_vehicle_age():
    assert calculate_vehicle_age(
        vehicle_year=2021,
        incident_year=2026,
    ) == 5

def test_vehicle_after_incident_is_rejected():
    with pytest.raises(
        FeatureEngineeringError
    ):
        calculate_vehicle_age(
            vehicle_year=2028,
            incident_year=2026,
        )

def test_build_triage_features():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    def get_asset(document_type):
        return next(
            asset
            for asset in assets
            if asset.type == document_type
        )
    claim_form = extract_claim_form(
        extract_pdf_text(
            get_asset(
                DocumentType.CLAIM_FORM
            )
        )
    )
    accident_report = (
        extract_accident_report(
            extract_pdf_text(
                get_asset(
                    DocumentType.ACCIDENT_REPORT
                )
            )
        )
    )
    garage_quote = extract_garage_quote(
        extract_pdf_text(
            get_asset(
                DocumentType.REPAIR_QUOTE
            )
        )
    )
    consistency = compare_claim_documents(
        claim_form,
        accident_report,
        garage_quote,
    )
    features = build_triage_features(
        case=case,
        claim_form=claim_form,
        garage_quote=garage_quote,
        consistency=consistency,
    )
    assert features.vehicle_age == 5

    assert features.repair_amount == 3160

    assert (
        features.declared_damage_count
        == 2
    )

    assert (
        features.additional_damage_count
        == 1
    )
    assert features.document_count == 4

    assert (
        features.missing_document_count
        == 1
    )

    assert (
        features.document_completeness_ratio
        == 0.75
    )

    assert not features.injuries_declared

    assert (
        features.consistency_issue_count
        == 1
    )