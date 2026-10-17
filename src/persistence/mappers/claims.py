"""Claim <-> relational row conversion with re-validation."""

from claims.models import Claim
from persistence.models import ClaimRow, CustomerRow, DeclaredDamageRow, DocumentRow, ImageRow, PolicyRow


def claim_to_row(claim: Claim, customer: CustomerRow, policy: PolicyRow) -> ClaimRow:
    """
    Validate again: model_copy/model_construct can bypass Pydantic validation.
    """
    claim = Claim.model_validate(claim.model_dump())
    if len(set(claim.declared_damage)) != len(claim.declared_damage):
        raise ValueError("A declared damage type may occur only once per claim")
    return ClaimRow(
        claim_id=claim.claim_id,
        customer=customer,
        policy=policy,
        claim_type=claim.claim_type,
        status=claim.status,
        incident_date=claim.incident.date,
        incident_location=claim.incident.location,
        collision_type=claim.incident.collision_type,
        injuries_declared=claim.incident.injuries_declared,
        vehicle_make=claim.vehicle.make,
        vehicle_model=claim.vehicle.model,
        vehicle_year=claim.vehicle.year,
        repair_estimate_amount=claim.repair_estimate.amount,
        repair_estimate_currency=claim.repair_estimate.currency,
        damages=[DeclaredDamageRow(damage_type=d, position=i) for i, d in enumerate(claim.declared_damage)],
        documents=[
            DocumentRow(document_type=d.type, filename=d.filename, available=d.available, position=i)
            for i, d in enumerate(claim.documents)
        ],
        images=[ImageRow(filename=f, position=i) for i, f in enumerate(claim.images)],
    )


def row_to_claim(row: ClaimRow) -> Claim:
    return Claim.model_validate(
        {
            "claim_id": row.claim_id,
            "claim_type": row.claim_type,
            "status": row.status,
            "customer": {"customer_id": row.customer.customer_id},
            "policy": {"policy_id": row.policy.policy_id, "product": row.policy.product},
            "incident": {
                "date": row.incident_date,
                "location": row.incident_location,
                "collision_type": row.collision_type,
                "injuries_declared": row.injuries_declared,
            },
            "vehicle": {"make": row.vehicle_make, "model": row.vehicle_model, "year": row.vehicle_year},
            "repair_estimate": {"amount": row.repair_estimate_amount, "currency": row.repair_estimate_currency},
            "declared_damage": [d.damage_type for d in row.damages],
            "documents": [
                {"type": d.document_type, "filename": d.filename, "available": d.available} for d in row.documents
            ],
            "images": [image.filename for image in row.images],
        }
    )
