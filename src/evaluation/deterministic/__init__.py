"""Offline deterministic evaluation backend for the controlled benchmark."""

from evaluation.deterministic.analysis import investigate, review_agent, score_models
from evaluation.deterministic.documents import extract_case_documents
from evaluation.deterministic.fixtures import ROOT, SYNTHETIC_PROVENANCE, materialize_scenario_case
from evaluation.deterministic.recording import TraceRecorder
from evaluation.deterministic.retrieval import retrieve_knowledge, retrieve_similar
from evaluation.deterministic.stages import (
    DeterministicRun,
    assess_history,
    build_features,
    compare_documents,
    extract_documents,
    load_claim,
)
from evaluation.deterministic.vision import assess_vision, fake_vision_service

__all__ = [
    "ROOT",
    "SYNTHETIC_PROVENANCE",
    "DeterministicRun",
    "TraceRecorder",
    "assess_history",
    "assess_vision",
    "build_features",
    "compare_documents",
    "extract_case_documents",
    "extract_documents",
    "fake_vision_service",
    "investigate",
    "load_claim",
    "materialize_scenario_case",
    "retrieve_knowledge",
    "retrieve_similar",
    "review_agent",
    "score_models",
]
