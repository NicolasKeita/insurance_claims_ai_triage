"""Similar-claims and grounded-answer stages over real in-memory stores."""

from evaluation.deterministic.fixtures import ROOT
from evaluation.fakes import FakeEmbeddings, FakeStructuredLlm, InMemoryRetrievalRepository


def retrieve_similar(run, client):
    """Index controlled claims in the real in-memory Qdrant store and search."""
    from retrieval.qdrant_store import QdrantClaimStore
    from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService

    run.embeddings = FakeEmbeddings()
    repository = InMemoryRetrievalRepository([run.case.claim, *(claim for claim, _ in run.history_inputs)])
    store = QdrantClaimStore(client, "e2e_claims", run.embeddings)
    SimilarClaimsIndexer(repository, run.embeddings, store).index_all()
    run.similar = SimilarClaimsService(repository, run.embeddings, store)
    retrieved = run.recorder.record("similar_claims", lambda: run.similar.search(run.case.claim.claim_id))
    run.output.facts["similar_claims"] = retrieved.model_dump(mode="json")


def index_knowledge(run, client):
    """Index the versioned knowledge catalog in the in-memory store."""
    from retrieval.knowledge.config import KnowledgeConfig
    from retrieval.knowledge.models import KnowledgeSourceType
    from retrieval.knowledge.qdrant_store import QdrantKnowledgeStore
    from retrieval.knowledge.retrieval import KnowledgeRetriever
    from retrieval.knowledge.sources import KnowledgeCatalog

    config = KnowledgeConfig(knowledge_dir=ROOT / "data/knowledge")
    store = QdrantKnowledgeStore(client, run.embeddings, config)
    catalog = KnowledgeCatalog.load(config.knowledge_dir, config.max_chunk_chars)
    for kind in KnowledgeSourceType:
        store.ensure_collection(kind)
        chunks = [chunk for chunk in catalog.chunks.values() if chunk.source_type == kind]
        store.upsert(kind, chunks, run.embeddings.embed_documents([chunk.text for chunk in chunks]))
    return KnowledgeRetriever(store)


def search_knowledge(run, question):
    """Search the real knowledge collections and record the retrieved chunks."""
    from retrieval.knowledge.models import KnowledgeFilter, KnowledgeSourceType

    return run.recorder.record(
        "retrieval",
        lambda: {
            "results": [
                chunk.model_dump(mode="json")
                for chunk in run.retriever.search(
                    question,
                    KnowledgeSourceType.POLICY,
                    filters=KnowledgeFilter(
                        product=run.case.claim.policy.product,
                        source_id=run.scenario.inputs.get("rag_source_id"),
                        as_of_date=run.case.claim.incident.date,
                    ),
                    limit=5,
                )
            ]
        },
    )


def retrieve_knowledge(run, client):
    """Index knowledge collections, retrieve chunks, then generate a grounded answer."""
    from retrieval.knowledge.models import RetrievedKnowledgeChunk
    from retrieval.knowledge.rag import KnowledgeRagService

    run.llm = FakeStructuredLlm(
        vision=run.scenario.vision_required, rag_supported=run.scenario.inputs.get("rag_supported", True)
    )
    run.retriever = index_knowledge(run, client)
    question = run.scenario.inputs.get("rag_question", "What is the demo collision deductible?")
    retrieved = search_knowledge(run, question)

    def answer_question():
        chunks = [RetrievedKnowledgeChunk.model_validate(item) for item in retrieved["results"]]
        return KnowledgeRagService(run.retriever, run.llm).answer_from_chunks(question, chunks)

    answer = run.recorder.record("rag_generation", answer_question)
    run.output.rag = {
        "answer": answer.model_dump(mode="json"),
        "retrieved_chunks": retrieved["results"],
        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved["results"]],
    }
    add_section_metrics(run, retrieved)
    run.output.facts["rag"] = run.output.rag


def add_section_metrics(run, retrieved):
    """Record separate retrieval metrics only when the scenario defines them."""
    scenario = run.scenario
    if not (scenario.expected_knowledge_sources and scenario.inputs.get("rag_section")):
        return
    from retrieval.knowledge.evaluation import section_metrics
    from retrieval.knowledge.models import RetrievedKnowledgeChunk

    run.output.rag["retrieval_metrics"] = section_metrics(
        [RetrievedKnowledgeChunk.model_validate(chunk) for chunk in retrieved["results"]],
        scenario.expected_knowledge_sources[0],
        [scenario.inputs["rag_section"]],
    )
