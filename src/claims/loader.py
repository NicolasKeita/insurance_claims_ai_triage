import json
from pathlib import Path

from claims.models import Claim
from claims.validation import validate_claim_files


def load_claim(claim_dir: Path) -> Claim:
    claim_path = claim_dir / "claim.json"

    with claim_path.open(encoding="utf-8") as file:
        raw_claim = json.load(file)

    claim = Claim.model_validate(raw_claim)

    validate_claim_files(
        claim=claim,
        claim_dir=claim_dir,
    )

    return claim