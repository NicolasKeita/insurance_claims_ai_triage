import json
from pathlib import Path


claim_path = Path("data/CLAIM-2026-00001/claim.json")

with claim_path.open(encoding="utf-8") as file:
    claim = json.load(file)

print(f"Claim: {claim['claim_id']}")
print(f"Type: {claim['claim_type']}")
print(f"Vehicle: {claim['vehicle']['make']} {claim['vehicle']['model']}")
print(f"Repair estimate: {claim['repair_estimate']['amount']} EUR")

print("\nDeclared damage:")
for damage in claim["declared_damage"]:
    print(f"- {damage}")

print("\nMissing documents:")
for document in claim["documents"]:
    if not document["available"]:
        print(f"- {document['type']}")