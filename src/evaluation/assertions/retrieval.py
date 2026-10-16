"""Grounded-answer schema, citation and knowledge-source checks."""

from pydantic import ValidationError

from evaluation.assertions.recording import Check
from knowledge.models import GroundedAnswer

CITED_FIELDS = ("chunk_id", "source_id", "source_version", "title", "section")


def _retrieved_chunks(rag):
    return rag.get("results", rag.get("retrieved_chunks", rag.get("retrieved", [])))


def _retrieved_ids(rag, results):
    if results:
        return {item.get("chunk_id") for item in results}
    return set(rag.get("retrieved_ids", rag.get("retrieved_chunk_ids", [])))


def _citations_match_chunks(citations, results):
    if not results:
        return not citations
    return all(
        any(all(citation.get(key) == result.get(key) for key in CITED_FIELDS) for result in results)
        for citation in citations
    )


def check_rag_answer(recorder, results, ids):
    """Validate the grounded answer schema and its citations."""
    rag = recorder.observation.rag
    answer = rag.get("answer", rag.get("grounded_answer"))
    try:
        parsed = GroundedAnswer.model_validate(answer)
        valid = True
    except (ValidationError, ValueError):
        parsed, valid = None, False
    recorder.record(Check("rag:answer_schema", "retrieval", "GroundedAnswer", answer, valid))
    citations = answer.get("citations", [])
    unknown = sorted({item.get("chunk_id") for item in citations} - ids)
    recorder.record(
        Check(
            "rag:citation_validity",
            "retrieval",
            "cited IDs and metadata from retrieved chunks",
            unknown,
            not unknown and _citations_match_chunks(citations, results),
        )
    )
    if parsed is not None and not parsed.insufficient_evidence:
        recorder.record(
            Check(
                "rag:grounded_answer_cited",
                "retrieval",
                "at least one valid citation",
                len(citations),
                bool(citations) and not unknown,
            )
        )


def check_rag_sources(recorder, results):
    """The expected knowledge sources and section must be retrieved."""
    scenario = recorder.scenario
    sources = {item.get("source_id") for item in results}
    recorder.record(
        Check(
            "rag:expected_sources",
            "retrieval",
            scenario.expected_knowledge_sources,
            sorted(sources),
            set(scenario.expected_knowledge_sources).issubset(sources),
        )
    )
    section = scenario.inputs.get("rag_section")
    if section:
        found = any(
            item.get("source_id") in scenario.expected_knowledge_sources and item.get("section") == section
            for item in results
        )
        recorder.record(
            Check("rag:expected_section", "retrieval", section, [item.get("section") for item in results], found)
        )


def check_rag_expectations(recorder):
    """Run the retrieval checks when a retrieval observation exists."""
    rag = recorder.observation.rag
    if not rag:
        return
    results = _retrieved_chunks(rag)
    ids = _retrieved_ids(rag, results)
    if rag.get("answer", rag.get("grounded_answer")) is not None:
        check_rag_answer(recorder, results, ids)
    if recorder.scenario.expected_knowledge_sources:
        check_rag_sources(recorder, results)