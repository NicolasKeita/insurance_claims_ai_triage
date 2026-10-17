"""Structured expectations and provenance checks, independent of LLM phrasing."""

from evaluation.assertions.agent import check_agent_expectations
from evaluation.assertions.facts import check_expected_facts, check_expected_signals, check_forbidden_signals
from evaluation.assertions.matching import FORBIDDEN_ACTIONS, fact_at, forbidden_decisions, matches_fact
from evaluation.assertions.observation import (
    check_affirmative_prose,
    check_provenance,
    check_structured_coherence,
)
from evaluation.assertions.recording import AssertionRecorder
from evaluation.assertions.retrieval import check_rag_expectations
from evaluation.assertions.schemas import check_component_schemas, check_document_schemas

__all__ = ["FORBIDDEN_ACTIONS", "evaluate_assertions", "fact_at", "forbidden_decisions", "matches_fact"]


def evaluate_assertions(scenario, observation, *, mode="deterministic"):
    """Evaluate every structured expectation for one scenario observation."""
    recorder = AssertionRecorder(scenario, observation, mode)
    check_expected_facts(recorder)
    check_expected_signals(recorder)
    check_forbidden_signals(recorder)
    check_document_schemas(recorder)
    check_component_schemas(recorder)
    check_structured_coherence(recorder)
    check_affirmative_prose(recorder)
    check_agent_expectations(recorder)
    check_rag_expectations(recorder)
    check_provenance(recorder)
    return recorder.checks
