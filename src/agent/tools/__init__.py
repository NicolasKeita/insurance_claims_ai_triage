"""Claim-scoped read-only tools; repositories are injectable at this boundary."""

from persistence.history import HistoricalClaimRepository as HistoricalClaimRepository
from persistence.repositories import AssessmentRepository as AssessmentRepository
from persistence.repositories import ClaimRepository as ClaimRepository
from persistence.vision_assessments import (
    VisionAssessmentRepository as VisionAssessmentRepository,
)

from ._adapter import ReadOnlyAgentTools as ReadOnlyAgentTools
from ._data import MAX_KNOWLEDGE_EXCERPT_CHARS as MAX_KNOWLEDGE_EXCERPT_CHARS
from ._data import MAX_SIGNALS as MAX_SIGNALS
from ._data import MAX_VALUE_CHARS as MAX_VALUE_CHARS
from ._data import _bounded as _bounded
from ._data import _claim_data as _claim_data
from ._protocol import AgentTools as AgentTools
