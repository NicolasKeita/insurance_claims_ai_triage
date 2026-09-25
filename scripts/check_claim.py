from pathlib import Path

from claims.loader import load_claim


claim_path = Path("data/CLAIM-2026-00001/claim.json")

claim = load_claim(claim_path)

print(f"Claim: {claim.claim_id}")
print(f"Type: {claim.claim_type}")
print(f"Vehicle: {claim.vehicle.make} {claim.vehicle.model}")
print(f"Incident date: {claim.incident.date}")
print(
    f"Repair estimate: "
    f"{claim.repair_estimate.amount} "
    f"{claim.repair_estimate.currency}"
)

print("\nDeclared damage:")

for damage in claim.declared_damage:
    print(f"- {damage}")

print("\nMissing documents:")

for document in claim.documents:
    if not document.available:
        print(f"- {document.type}")