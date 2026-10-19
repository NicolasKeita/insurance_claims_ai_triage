"""Versioned rules for review prioritization and signal thresholds."""

from investigation.models import SignalCode
from investigation.policy_model import (
    HISTORICAL_CODES,
    MULTIMODAL_CODES,
    V1_CODES,
    InvestigationPolicy,
)

__all__ = [
    "DEFAULT_INVESTIGATION_POLICY",
    "HISTORICAL_CODES",
    "INVESTIGATION_POLICY_V1",
    "INVESTIGATION_POLICY_V2",
    "INVESTIGATION_POLICY_V3",
    "InvestigationPolicy",
    "MULTIMODAL_CODES",
    "V1_CODES",
]


INVESTIGATION_POLICY_V1 = InvestigationPolicy()
INVESTIGATION_POLICY_V2 = InvestigationPolicy(
    version="investigation_policy_v2",
    signal_weights={
        **INVESTIGATION_POLICY_V1.signal_weights,
        SignalCode.RECENT_CLAIM_FREQUENCY: 15,
        SignalCode.PREVIOUS_ANOMALOUS_CLAIMS: 18,
    },
    recent_claims_30_days_threshold=3,
    recent_claims_365_days_threshold=5,
    previous_anomalous_claims_threshold=2,
)
INVESTIGATION_POLICY_V3 = InvestigationPolicy(
    **{
        **INVESTIGATION_POLICY_V2.model_dump(),
        "version": "investigation_policy_v3",
        "signal_weights": {
            **INVESTIGATION_POLICY_V2.signal_weights,
            SignalCode.VISUAL_ADDITIONAL_DAMAGE: 12,
        },
        "multimodal_enabled": True,
    }
)
DEFAULT_INVESTIGATION_POLICY = INVESTIGATION_POLICY_V2
