"""Retrieval and similar-claims stages over the configured real stores."""

from evaluation.real.context import payload


class RetrievalSteps:
    """Search real knowledge and similar-claim collections without fake backends."""

    def retrieve(self, run):
        """Retrieve knowledge chunks and generate a grounded answer when possible."""
        observed = run.observed
        if not self.config.include_rag:
            self._skip(observed, "retrieval", "Disabled by configuration")
            self._skip(observed, "rag_generation", "Disabled by configuration")
            return
        if not self._available(observed, "retrieval", ["qdrant", "knowledge_collections", "embeddings"]):
            self._skip(observed, "rag_generation", "Retrieval unavailable")
            return
        from knowledge.evaluation import section_metrics
        from knowledge.models import KnowledgeFilter, KnowledgeSourceType
        from knowledge.qdrant_store import QdrantKnowledgeStore
        from knowledge.retrieval import KnowledgeRetriever

        retriever = KnowledgeRetriever(
            QdrantKnowledgeStore(
                self.resources["qdrant_client"], self.resources["embeddings"], self.resources["knowledge_config"]
            )
        )
        scenario = run.scenario
        question = scenario.inputs.get(
            "rag_question", "What documents are needed for human review of the demonstration claim?"
        )
        kind = KnowledgeSourceType(scenario.inputs.get("rag_source_type", "POLICY"))
        results = self._record(
            observed,
            "retrieval",
            lambda: retriever.search(
                question,
                kind,
                filters=KnowledgeFilter(product=scenario.inputs.get("rag_product"), language="en"),
                limit=5,
            ),
            inputs={"question": question, "source_type": kind.value},
        )
        if results is None:
            self._skip(observed, "rag_generation", "Retrieval failed")
            return
        observed.rag = {"results": [payload(item) for item in results], "question": question, "answer": None}
        if scenario.inputs.get("rag_supported") and scenario.inputs.get("rag_section"):
            observed.rag["retrieval_metrics"] = section_metrics(
                results, scenario.inputs.get("rag_source_id"), [scenario.inputs["rag_section"]]
            )
        observed.provenance["rag"] = {"retrieved_chunk_ids": [item.chunk_id for item in results]}
        self.generate_answer(run, retriever, results, question)

    def generate_answer(self, run, retriever, results, question):
        """Generate and record a grounded answer with the configured local LLM."""
        observed = run.observed
        if not self._available(observed, "rag_generation", ["ollama"]):
            return
        import os

        from claims.llm import OllamaStructuredLlm
        from knowledge.rag import RAG_SYSTEM_PROMPT, KnowledgeRagService

        llm = OllamaStructuredLlm(
            model=self.resources["llm_model"],
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        answer = self._record(
            observed,
            "rag_generation",
            lambda: KnowledgeRagService(retriever, llm).answer_from_chunks(question, results),
            inputs={"question": question, "retrieved_chunk_ids": [item.chunk_id for item in results]},
            sources=[item.chunk_id for item in results],
        )
        if answer is not None:
            observed.rag["answer"] = answer.model_dump(mode="json")
        observed.facts["rag"] = observed.rag
