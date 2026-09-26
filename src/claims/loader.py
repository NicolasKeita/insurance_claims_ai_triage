import json
from pathlib import Path

from claims.models import Claim


def load_claim(claim_dir: Path) -> Claim:
    claim_path = claim_dir / "claim.json"

    with claim_path.open(encoding="utf-8") as file:
        raw_claim = json.load(file)

    return Claim.model_validate(raw_claim)