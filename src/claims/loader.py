import json
from pathlib import Path

from claims.models import Claim


def load_claim(claim_path: Path) -> Claim:
    with claim_path.open(encoding="utf-8") as file:
        raw_claim = json.load(file)

    return Claim.model_validate(raw_claim)