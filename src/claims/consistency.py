from pydantic import BaseModel

from claims.document_models import (
    AccidentReportExtraction,
    ClaimFormExtraction,
    GarageQuoteExtraction,
)
from claims.enums import DamageType

class ClaimConsistencyReport(BaseModel):
    claim_id_match: bool
    incident_date_match: bool
    location_match: bool
    vehicle_match: bool
    collision_type_match: bool
    injuries_match: bool

    declared_damage_match: bool

    additional_quote_damage: tuple[
        DamageType,
        ...
    ]

    unmapped_quote_items: tuple[str, ...]

    @property
    def is_consistent(self) -> bool:
        return (
            self.claim_id_match
            and self.incident_date_match
            and self.location_match
            and self.vehicle_match
            and self.collision_type_match
            and self.injuries_match
            and self.declared_damage_match
            and not self.additional_quote_damage
            and not self.unmapped_quote_items
        )

QUOTE_DAMAGE_MAP = {
    "front bumper": DamageType.FRONT_BUMPER,
    "left headlight": DamageType.LEFT_HEADLIGHT,
    "hood": DamageType.HOOD,
}

def compare_claim_documents(
    form: ClaimFormExtraction,
    report: AccidentReportExtraction,
    quote: GarageQuoteExtraction,
) -> ClaimConsistencyReport:
    claim_id_match = (
        form.claim_id
        == report.claim_id
        == quote.claim_id
    )

    incident_date_match = (
        form.incident_date
        == report.incident_date
    )

    location_match = (
        form.location.casefold()
        == report.location.casefold()
    )

    vehicle_match = (
        form.vehicle_make.casefold()
        == report.vehicle_make.casefold()
        and
        form.vehicle_model.casefold()
        == report.vehicle_model.casefold()
    )

    collision_type_match = (
        form.collision_type
        == report.collision_type
    )

    injuries_match = (
        form.injuries_declared
        == report.injuries_declared
    )
    declared_damage_match = (
        set(form.declared_damage)
        == set(report.reported_damage)
    )
    quote_damage = []
    unmapped_quote_items = []

    for item in quote.parts:
        normalized = item.description.casefold()

        damage = QUOTE_DAMAGE_MAP.get(
            normalized
        )

        if damage is None:
            unmapped_quote_items.append(
                item.description
            )
            continue

        quote_damage.append(damage)
        declared = set(
        form.declared_damage
    )

    additional_quote_damage = tuple(
        damage
        for damage in quote_damage
        if damage not in declared
    )
    return ClaimConsistencyReport(
        claim_id_match=claim_id_match,
        incident_date_match=incident_date_match,
        location_match=location_match,
        vehicle_match=vehicle_match,
        collision_type_match=collision_type_match,
        injuries_match=injuries_match,
        declared_damage_match=declared_damage_match,
        additional_quote_damage=additional_quote_damage,
        unmapped_quote_items=tuple(
            unmapped_quote_items
        ),
    )