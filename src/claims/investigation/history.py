"""Pure historical facts; incident dates define the business chronology."""

from datetime import date, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def history_window_start(current_incident_date: date, days: int) -> date:
    """Inclusive lower boundary; the upper boundary is always exclusive."""
    if days <= 0:
        raise ValueError("Window length must be positive")
    return current_incident_date - timedelta(days=days)


class HistoricalClaimProfile(BaseModel):
    """Counts cover all currencies; amounts cover only repair_amount_currency.

    No compatible amounts: total=0, average/max=None. No previous claims:
    counts=0 and days_since_previous_claim=None. Anomaly flags are the latest
    currently stored model outputs, not incident-time ground truth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    previous_claim_count: int = Field(default=0, ge=0)
    claims_last_30_days: int = Field(default=0, ge=0)
    claims_last_90_days: int = Field(default=0, ge=0)
    claims_last_365_days: int = Field(default=0, ge=0)
    days_since_previous_claim: int | None = Field(default=None, ge=1)
    repair_amount_currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    previous_claims_in_amount_currency: int = Field(default=0, ge=0)
    total_previous_repair_amount: Decimal = Field(default=Decimal("0"), ge=0)
    average_previous_repair_amount: Decimal | None = Field(default=None, ge=0)
    max_previous_repair_amount: Decimal | None = Field(default=None, ge=0)
    previous_anomalous_claim_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_facts(self):
        _validate_count_nesting(self)
        _validate_interval(self)
        _validate_amounts(self)
        return self


def _validate_count_nesting(profile):
    if not (
        profile.claims_last_30_days
        <= profile.claims_last_90_days
        <= profile.claims_last_365_days
        <= profile.previous_claim_count
    ):
        raise ValueError("Historical window counts must be nested")
    if profile.previous_anomalous_claim_count > profile.previous_claim_count:
        raise ValueError("Anomalous count exceeds previous claim count")
    if profile.previous_claims_in_amount_currency > profile.previous_claim_count:
        raise ValueError("Currency count exceeds previous claim count")


def _validate_interval(profile):
    if (profile.previous_claim_count == 0) != (profile.days_since_previous_claim is None):
        raise ValueError("Previous claim interval must match history availability")
    if profile.days_since_previous_claim is None:
        return
    for days, count in (
        (30, profile.claims_last_30_days),
        (90, profile.claims_last_90_days),
        (365, profile.claims_last_365_days),
    ):
        if (count > 0) != (profile.days_since_previous_claim <= days):
            raise ValueError("Window count disagrees with most recent incident")


def _validate_amounts(profile):
    if profile.previous_claims_in_amount_currency == 0:
        if (
            profile.total_previous_repair_amount != 0
            or profile.average_previous_repair_amount is not None
            or profile.max_previous_repair_amount is not None
        ):
            raise ValueError("No compatible amounts requires zero total and null average/max")
        return
    if (
        profile.repair_amount_currency is None
        or profile.average_previous_repair_amount is None
        or profile.max_previous_repair_amount is None
    ):
        raise ValueError("Compatible amounts require currency, average and maximum")
    if not (
        profile.average_previous_repair_amount
        <= profile.max_previous_repair_amount
        <= profile.total_previous_repair_amount
    ):
        raise ValueError("Inconsistent amount aggregates")
