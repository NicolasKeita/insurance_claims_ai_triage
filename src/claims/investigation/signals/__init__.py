"""Independent fact-to-signal translators. Scoring belongs to the policy."""

from claims.investigation.signals.anomaly import signals_from_anomaly
from claims.investigation.signals.completeness import signals_from_completeness
from claims.investigation.signals.consistency import signals_from_consistency
from claims.investigation.signals.history import signals_from_history
from claims.investigation.signals.multimodal import signals_from_multimodal

__all__ = [
    "signals_from_anomaly",
    "signals_from_completeness",
    "signals_from_consistency",
    "signals_from_history",
    "signals_from_multimodal",
]
