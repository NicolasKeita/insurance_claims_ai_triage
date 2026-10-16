"""Explicit test boundaries; no fake is imported by the real-stack backend.

These adapters replace external inference and persistence only. Domain parsing,
retrieval filtering, fusion, scoring, planning and grounding remain production code.
"""

from evaluation.fakes.embeddings import FakeEmbeddings
from evaluation.fakes.history import FakeHistoricalRepository, historical_claims
from evaluation.fakes.llm import FakeStructuredLlm
from evaluation.fakes.predictors import FakeAnomalyPredictor, FakeTriagePredictor
from evaluation.fakes.retrieval import InMemoryRetrievalRepository
from evaluation.fakes.tools import SnapshotAgentTools

__all__ = [
    "FakeAnomalyPredictor",
    "FakeEmbeddings",
    "FakeHistoricalRepository",
    "FakeStructuredLlm",
    "FakeTriagePredictor",
    "InMemoryRetrievalRepository",
    "SnapshotAgentTools",
    "historical_claims",
]