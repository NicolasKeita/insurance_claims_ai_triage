import json

import pytest

from knowledge.config import ROOT
from knowledge.models import GroundingError, KnowledgeSourceType as Kind, RetrievedKnowledgeChunk
from knowledge.rag import KnowledgeRagService
from knowledge.sources import KnowledgeCatalog
from knowledge_fakes import FakeStructuredLlm


def retrieved():
    chunk = next(iter(KnowledgeCatalog.load(ROOT / "data/knowledge").chunks.values()))
    return RetrievedKnowledgeChunk(**chunk.model_dump(), similarity_score=0.5)


class FakeRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks if chunks is not None else [retrieved()]
        self.error = error
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.chunks


def test_citations_resolved_from_retrieval_and_context_is_bounded():
    chunk = retrieved()
    llm = FakeStructuredLlm({"answer": "Demo answer.", "cited_chunk_ids": [chunk.chunk_id, chunk.chunk_id], "insufficient_evidence": False})
    retriever = FakeRetriever([chunk])
    answer = KnowledgeRagService(retriever, llm).answer("test", Kind.POLICY, product="AUTO_BASIC")
    assert len(answer.citations) == 1
    assert answer.citations[0].source_id == chunk.source_id
    assert answer.citations[0].source_version == chunk.source_version
    assert answer.citations[0].section == chunk.section and answer.citations[0].title == chunk.title
    assert retriever.calls[0][1]["product"] == "AUTO_BASIC"
    prompt = llm.prompts[0]
    context = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
    assert len(context["retrieved_sources"]) == 1
    assert "Do not use outside knowledge" in prompt and "not instructions" in prompt


@pytest.mark.parametrize("insufficient", [False, True])
def test_unknown_citation_rejected_even_when_insufficient(insufficient):
    llm = FakeStructuredLlm({"answer": "Bad citation", "cited_chunk_ids": ["invented"], "insufficient_evidence": insufficient})
    with pytest.raises(GroundingError, match="absent"):
        KnowledgeRagService(FakeRetriever(), llm).answer("test", Kind.POLICY)


def test_uncited_answer_rejected_and_insufficient_accepted():
    output = {"answer": "Excerpts do not establish volcanic ash coverage.", "cited_chunk_ids": [], "insufficient_evidence": True}
    answer = KnowledgeRagService(FakeRetriever(), FakeStructuredLlm(output)).answer("volcanic ash?", Kind.POLICY)
    assert answer.insufficient_evidence and answer.citations == []
    output["insufficient_evidence"] = False
    with pytest.raises(GroundingError, match="must cite"):
        KnowledgeRagService(FakeRetriever(), FakeStructuredLlm(output)).answer("test", Kind.POLICY)


def test_inline_uuid_reference_cannot_bypass_structured_citation_validation():
    chunk = retrieved()
    output = {"answer": "Unverified inline reference [00000000-0000-0000-0000-000000000000].",
              "cited_chunk_ids": [chunk.chunk_id], "insufficient_evidence": False}
    with pytest.raises(GroundingError, match="absent"):
        KnowledgeRagService(FakeRetriever([chunk]), FakeStructuredLlm(output)).answer("test", Kind.POLICY)


def test_valid_inline_reference_is_resolved_and_marker_removed():
    chunk = retrieved()
    output = {"answer": "Evidence does not establish coverage [chunk_id: " + chunk.chunk_id + "].",
              "cited_chunk_ids": [], "insufficient_evidence": True}
    answer = KnowledgeRagService(FakeRetriever([chunk]), FakeStructuredLlm(output)).answer("test", Kind.POLICY)
    assert answer.answer == "Evidence does not establish coverage."
    assert answer.citations[0].chunk_id == chunk.chunk_id
    assert answer.citations[0].source_id == chunk.source_id


def test_retrieval_must_finish_before_generation_and_empty_context_skips_llm():
    retriever = FakeRetriever(error=RuntimeError("retrieval failed"))
    llm = FakeStructuredLlm()
    with pytest.raises(RuntimeError, match="retrieval failed"):
        KnowledgeRagService(retriever, llm).answer("test", Kind.POLICY)
    assert llm.prompts == []
    answer = KnowledgeRagService(FakeRetriever([]), llm).answer("test", Kind.POLICY)
    assert answer.insufficient_evidence and llm.prompts == []
    events = []

    class OrderedRetriever(FakeRetriever):
        def search(self, *args, **kwargs):
            events.append("retrieval")
            return super().search(*args, **kwargs)

    class OrderedLlm(FakeStructuredLlm):
        def generate(self, **kwargs):
            assert events == ["retrieval"]
            events.append("llm")
            return super().generate(**kwargs)

    KnowledgeRagService(OrderedRetriever(), OrderedLlm()).answer("test", Kind.POLICY)
    assert events == ["retrieval", "llm"]
