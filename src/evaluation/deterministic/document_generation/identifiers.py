"""Stable synthetic customer and policy references without scenario names."""

import hashlib


def fake_identifiers(seed: str) -> tuple[str, str]:
    """Return separate numeric references, stable across processes and runs."""
    values = []
    for namespace, prefix in (("customer", "CLI"), ("policy", "AUTO")):
        digest = hashlib.sha256(f"claims-demo:{namespace}:{seed}".encode()).digest()
        number = int.from_bytes(digest[:8], "big") % 10_000_000_000
        values.append(f"{prefix}-{number:010d}")
    return values[0], values[1]


def display_identifiers(review):
    """Alias legacy scenario-labelled references only in demonstration views."""
    if review.mode != "demo":
        return review
    claim = review.claim.model_copy(deep=True)
    customer = claim.customer.customer_id
    policy = claim.policy.policy_id
    if customer.startswith("DEMO-E2E-CUSTOMER-"):
        seed = customer.removeprefix("DEMO-E2E-CUSTOMER-")
        claim.customer.customer_id = fake_identifiers(seed)[0]
    if policy.startswith("DEMO-E2E-POLICY-"):
        seed = policy.removeprefix("DEMO-E2E-POLICY-")
        claim.policy.policy_id = fake_identifiers(seed)[1]
    return review.model_copy(update={"claim": claim})
