"""Real agent review stage and its lazy service loaders."""

import os

from evaluation.real.agent_environment import llm_environment
from evaluation.tool_outages import history_outage_session, tool_outage_reason


class AgentSteps:
    """Run the real agent service against the existing application tools."""

    def assess_agent(self, run):
        """
        The normal application factory supplies lazy real services. Closing here only closes those
        owned by this service, not preflight clients.
        """
        observed, scenario = run.observed, run.scenario
        if self._skip_agent(observed, scenario):
            return
        self._run_agent(run, observed)

    def _skip_agent(self, observed, scenario):
        """Report every precondition that prevents the real agent stage from running."""
        if not self.config.include_agent:
            self._skip(observed, "agent", "Disabled by configuration")
            return True
        if not self._available(observed, "agent", ["postgresql", "ollama"]):
            return True
        outage_reason = self._agent_tool_outage_reason(scenario)
        if outage_reason:
            self._skip(observed, "agent", outage_reason)
            return True
        vision = observed.components.get("vision")
        if scenario.vision_required and vision and vision.status == "SKIP":
            self._skip(
                observed,
                "agent",
                "This golden case requires unavailable real visual ground truth; its document subset remains observed",
            )
            return True
        history = observed.components.get("history")
        if history and history.status != "PASS":
            self._skip(
                observed, "agent", "Controlled claim has no matching stored source; no fake agent tool backend used"
            )
            return True
        return False

    def _agent_tool_outage_reason(self, scenario):
        """Require both declared opt-in and the active disposable schema at execution."""
        reason = tool_outage_reason(self.config, scenario)
        if reason is None and scenario.inputs.get("unavailable_tools") and not self.resources.get("isolated_schema"):
            return "Controlled agent tool outages require an active isolated schema"
        return reason

    def _run_agent(self, run, observed):
        """Run the real application service and record its review result."""
        from agent.factory import create_claims_agent_service

        unavailable = set(run.scenario.inputs.get("unavailable_tools", []))
        options = {}
        inputs = {"claim_id": run.case.claim.claim_id, "source": "existing PostgreSQL claim"}
        if unavailable:
            reason = self._agent_tool_outage_reason(run.scenario)
            if reason:
                raise ValueError(reason)
            inputs["controlled_unavailable_tools"] = sorted(unavailable)
            observed.provenance["agent_tool_outages"] = {
                "controlled": True,
                "scope": "agent_only",
                "requested_tools": sorted(unavailable),
                "isolated_schema": self.resources["isolated_schema"],
                "methods": {
                    tool: "injected_history_connection_failure" if tool == "GET_HISTORY" else "omitted_isolated_assessment"
                    for tool in sorted(unavailable)
                },
            }
        if "GET_HISTORY" in unavailable:
            options.update(history_session_factory=history_outage_session, require_history=True)
        with llm_environment(self.resources["llm_model"]):
            service = create_claims_agent_service(
                self.resources["session_factory"],
                similar_service_loader=self.similar_service,
                knowledge_retriever_loader=self.knowledge_retriever,
                **options,
            )
            try:
                result = self._record(
                    observed,
                    "agent",
                    lambda: self.review(service, run),
                    inputs=inputs,
                    error_from_result=self._agent_error,
                )
            finally:
                service.tools.close()
        if result is not None:
            self._record_result(observed, result)

    def _record_result(self, observed, result):
        """Preserve the terminal review payload after its status has been recorded."""
        observed.agent = result.model_dump(mode="json")
        observed.facts["agent"] = observed.agent
        observed.model_versions["agent_llm"] = self.resources["llm_model"]

    @staticmethod
    def _agent_error(result):
        """Classify a persisted failed run before publishing stage completion."""
        return None if result.status == "COMPLETED" else result.error or "Real agent did not complete"

    def review(self, service, run):
        """Review the stored claim, tolerating a completed run recorded as an error."""
        from agent.service import AgentRunError
        from persistence.agent_runs import AgentRunStore

        try:
            return service.review_claim(
                run.case.claim.claim_id,
                objective="Review the controlled demonstration claim and recommend the next human review step.",
            )
        except AgentRunError as error:
            result = AgentRunStore(self.resources["session_factory"]).get_by_run_id(error.run_id)
            if result is None:
                raise
            return result

    def knowledge_retriever(self):
        """Load the real knowledge retriever, or fail the agent tool explicitly."""
        from retrieval.models import RetrievalUnavailableError

        available = ("qdrant", "knowledge_collections", "embeddings")
        if not self.config.include_rag or not all(self.preflight.available(name) for name in available):
            raise RetrievalUnavailableError("Knowledge retrieval unavailable or disabled by evaluation configuration")
        from qdrant_client import QdrantClient

        from retrieval.knowledge.qdrant_store import QdrantKnowledgeStore
        from retrieval.knowledge.retrieval import KnowledgeRetriever

        config = self.resources["knowledge_config"]
        client = QdrantClient(
            url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
        )
        return KnowledgeRetriever(QdrantKnowledgeStore(client, self.resources["embeddings"], config))
