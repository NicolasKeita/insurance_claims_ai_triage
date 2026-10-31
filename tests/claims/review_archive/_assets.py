"""Original byte fixtures and a claim inventory, without a model or database."""

import base64

from claims.models import Claim

IMAGE = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1sAAAAASUVORK5CYII=")
PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"


def claim_inventory():
    return Claim.model_validate(
        {
            "claim_id": "CLAIM-0001",
            "claim_type": "AUTO_COLLISION",
            "status": "NEW",
            "incident": {
                "date": "2026-01-20",
                "location": "Lyon",
                "collision_type": "FRONT_COLLISION",
                "injuries_declared": False,
            },
            "vehicle": {"make": "Renault", "model": "Clio", "year": 2021},
            "customer": {"customer_id": "CUSTOMER-001"},
            "policy": {"policy_id": "POLICY-001", "product": "AUTO"},
            "repair_estimate": {"amount": "1500", "currency": "EUR"},
            "declared_damage": ["FRONT_BUMPER"],
            "documents": [
                {"type": "CLAIM_FORM", "filename": "declaration finale.pdf", "available": True},
                {"type": "POLICE_REPORT", "filename": "police.pdf", "available": False},
            ],
            "images": ["photo originale.png"],
        }
    )


def source_assets(root):
    root.mkdir(parents=True)
    document, image = root / "declaration finale.pdf", root / "photo originale.png"
    document.write_bytes(PDF)
    image.write_bytes(IMAGE)
    return {document.name: document}, {image.name: image}
