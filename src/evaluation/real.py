"""Real application adapters behind the opt-in evaluation pipeline.

Unavailable components remain explicit. Independent document parsing, model
inference and local multimodal diagnostics do not require a running database.
"""

import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from evaluation.models import ComponentResult, EvaluationScenario, EvaluationStatus, PipelineObservation, TraceEvent
from evaluation.preflight import ROOT, PreflightReport, RealEvaluationConfig, run_preflight


def _payload(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class RealClaimsEvaluationPipeline:
    def __init__(self, config: RealEvaluationConfig, preflight: PreflightReport | None = None):
        self.config = config
        self.preflight = preflight if preflight is not None else run_preflight(config)
        self.resources = self.preflight.resources

    def close(self):
        self.preflight.close()

    def _record(self, observation, component, operation: Callable, *, inputs=None, sources=None):
        started = perf_counter()
        try:
            result = operation()
            payload = _payload(result)
            outputs = payload if isinstance(payload, dict) else {"result": payload}
            status, reason = "PASS", None
        except Exception as error:
            result, outputs = None, {}
            status, reason = "FAIL", f"{type(error).__name__}: {error}"
            observation.errors.append(f"{component}: {reason}")
        duration = perf_counter() - started
        observation.components[component] = ComponentResult(status=status, duration_seconds=duration, reason=reason)
        observation.trace.append(
            TraceEvent(
                component=component,
                status=status,
                inputs=inputs or {},
                outputs=outputs,
                duration_seconds=duration,
                source_references=sources or [],
            )
        )
        return result

    def _skip(self, observation, component, reason, *, dependency=False):
        status = self.config.dependency_unavailable if dependency else "SKIP"
        observation.components[component] = ComponentResult(status=status, reason=reason)
        observation.trace.append(TraceEvent(component=component, status=status, outputs={"reason": reason}))
        if status == "FAIL":
            observation.errors.append(f"{component}: {reason}")

    def _available(self, observation, component, dependencies):
        missing = [name for name in dependencies if not self.preflight.available(name)]
        if missing:
            self._skip(
                observation,
                component,
                "; ".join(f"{name}: {self.preflight.reason(name)}" for name in missing),
                dependency=True,
            )
            return False
        return True

    def _version(self, observation, kind):
        row = next((row for row in self.preflight.rows if row["component"] == kind), {})
        if row.get("version"):
            observation.model_versions[kind] = str(row["version"])
        observation.provenance[kind] = {key: value for key, value in row.items() if key != "duration_seconds"}

    def run(self, scenario: EvaluationScenario) -> PipelineObservation:
        """
        The shared helper uses real PDF ingestion and production extraction functions.

        Evaluation fixtures are never silently imported into the application DB. History for
        controlled claims requires an explicit prepared/seeded source.

        The observed subset remains inspectable, but cannot validate a scenario whose required signal
        depends on unavailable history.

        Real stored absence represents the unavailable-tool case; the read-only application tools see
        genuine missing assessment rows.
        """
        started = perf_counter()
        observed = PipelineObservation()
        observed.provenance["mode"] = "real"
        observed.provenance["scenario_source"] = scenario.claim_source
        if scenario.inputs.get("external_vision_diagnostic"):
            self._diagnostic_vision(scenario, observed)
            self._finish(observed, started)
            return observed

        from claims.loader import load_claim_case
        from evaluation.pipeline import extract_case_documents, materialize_scenario_case

        use_reference = scenario.inputs.get("use_reference_case", False)
        if use_reference:
            case = self._record(
                observed,
                "claim",
                lambda: load_claim_case(ROOT / scenario.claim_source),
                sources=[str(ROOT / scenario.claim_source / "claim.json")],
            )
        else:
            self.config.fixture_dir.mkdir(parents=True, exist_ok=True)
            fixture_root = Path(mkdtemp(prefix=scenario.scenario_id + "-", dir=self.config.fixture_dir))
            case = self._record(
                observed,
                "claim",
                lambda: materialize_scenario_case(scenario, fixture_root),
                inputs={"controlled_synthetic_inputs": scenario.inputs},
                sources=[scenario.claim_source],
            )
        if case is None:
            for name in (
                "document",
                "consistency",
                "triage",
                "anomaly",
                "history",
                "investigation",
                "vision",
                "retrieval",
                "agent",
            ):
                self._skip(observed, name, "Claim loading failed")
            self._finish(observed, started)
            return observed
        observed.facts["claim"] = case.claim.model_dump(mode="json")
        observed.provenance["controlled_synthetic_claim"] = not use_reference
        observed.provenance["claim_id"] = case.claim.claim_id
        if self.config.isolated_demo_stack:
            from evaluation.isolated_stack import seed_case, seed_history

            seeded = self._record(
                observed,
                "demo_seed",
                lambda: seed_case(self.resources["session_factory"], case),
                inputs={"claim_id": case.claim.claim_id, "schema": self.resources["isolated_schema"]},
            )
            if scenario.inputs.get("history_claims") and seeded is not None:
                self._record(
                    observed,
                    "history_seed",
                    lambda: seed_history(
                        self.resources["session_factory"], case.claim, scenario.inputs["history_claims"]
                    ),
                    inputs={
                        "claim_id": case.claim.claim_id,
                        "controlled_previous_claims": scenario.inputs["history_claims"],
                    },
                )

        docs = self._record(
            observed,
            "document",
            lambda: extract_case_documents(case),
            inputs={"claim_id": case.claim.claim_id},
            sources=[str(case.document_path(d.filename)) for d in case.claim.documents if d.available],
        )
        extractions = docs[0] if docs is not None else {}
        from claims.consistency import compare_claim_documents
        from claims.enums import DocumentType
        from ml.features import build_triage_features, calculate_document_completeness

        form, report, quote = (
            extractions.get(kind)
            for kind in (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)
        )
        count, missing, ratio = calculate_document_completeness(case)
        observed.facts["extraction"] = {
            "success": bool(extractions),
            "document_count": len(extractions),
            "form": _payload(form),
            "report": _payload(report),
            "quote": _payload(quote),
        }
        observed.provenance["documents"] = [item.source.model_dump(mode="json") for item in extractions.values()]
        for source in observed.provenance["documents"]:
            source["asset_reference"] = str(case.document_path(source["filename"]))
        if form is not None and report is not None and quote is not None:
            consistency = self._record(
                observed,
                "consistency",
                lambda: compare_claim_documents(form, report, quote),
                sources=[source["asset_reference"] for source in observed.provenance["documents"]],
            )
        else:
            consistency = None
            self._skip(observed, "consistency", "At least one comparison document is unavailable; no matches invented")
        if consistency is not None:
            observed.facts["consistency"] = {
                **consistency.model_dump(mode="json"),
                "is_consistent": consistency.is_consistent,
            }
            features = self._record(observed, "features", lambda: build_triage_features(case, form, quote, consistency))
        else:
            features = None
            self._skip(observed, "features", "Document comparison is unavailable")
        observed.facts["features"] = (
            features.model_dump(mode="json")
            if features is not None
            else {
                "document_count": count,
                "missing_document_count": missing,
                "document_completeness_ratio": ratio,
            }
        )
        observed.facts["extraction"].update(missing_document_count=missing, document_completeness_ratio=ratio)

        triage, anomaly, registered_anomaly_result = None, None, None
        for kind in ("triage", "anomaly"):
            if features is None:
                self._skip(observed, kind, "Production feature inputs are unavailable")
                continue
            if not self._available(observed, kind, [kind]):
                continue
            self._version(observed, kind)
            predictor = self.resources[kind]
            if kind == "triage":
                triage = self._record(
                    observed,
                    kind,
                    lambda predictor=predictor: predictor.predict(features),
                    inputs=features.model_dump(mode="json"),
                    sources=[str(observed.provenance[kind].get("checkpoint", "mlflow_candidate"))],
                )
                if triage is not None:
                    observed.facts[kind] = triage.model_dump(mode="json")
            else:
                result = self._record(
                    observed,
                    kind,
                    lambda predictor=predictor: predictor.assess(features),
                    inputs=features.model_dump(mode="json"),
                    sources=[str(observed.provenance[kind].get("checkpoint", "mlflow_candidate"))],
                )
                if result is not None:
                    if hasattr(result, "assessment"):
                        registered_anomaly_result = result
                    anomaly = result.assessment if hasattr(result, "assessment") else result
                    observed.facts[kind] = anomaly.model_dump(mode="json")
                    observed.facts[kind]["interpretation"] = "Statistical atypicality; not a fraud determination"

        history = None
        if self._available(observed, "history", ["postgresql"]):
            from persistence.history import HistoricalClaimRepository
            from persistence.repositories import ClaimRepository

            factory = self.resources["session_factory"]
            try:
                with factory() as session:
                    stored = ClaimRepository(session).get_by_claim_id(case.claim.claim_id)
                if stored is None:
                    self._skip(
                        observed, "history", "Controlled claim is not seeded in the configured PostgreSQL database"
                    )
                elif not use_reference and stored.model_dump(mode="json") != case.claim.model_dump(mode="json"):
                    self._skip(
                        observed, "history", "Stored claim differs from controlled fixture; no unrelated history reused"
                    )
                else:

                    def load_history():
                        with factory() as session:
                            return HistoricalClaimRepository(session).get_profile(case.claim.claim_id)

                    history = self._record(
                        observed, "history", load_history, sources=["postgresql:claims:" + case.claim.claim_id]
                    )
            except Exception as error:
                self._skip(
                    observed,
                    "history",
                    f"Configured PostgreSQL schema unavailable: {type(error).__name__}: {error}",
                    dependency=True,
                )
            if history is not None:
                observed.facts["history"] = history.model_dump(mode="json")

        vision = None
        if not self.config.include_vision:
            self._skip(observed, "vision", "Disabled by configuration")
        elif scenario.vision_required and not scenario.external_assets:
            self._skip(
                observed,
                "vision",
                "Golden vision uses deterministic detectors; no matching real image ground truth. "
                "Real external CarDD diagnostic is reported separately",
            )
        elif self._available(observed, "vision", ["damage_vision", "part_vision"]):
            from claims.assets import resolve_image_assets
            from claims.consistency import map_quote_damage_types
            from vision.images import load_validated_image
            from vision.multimodal_service import ClaimMultimodalVisionService

            reasons = []
            assets = resolve_image_assets(case)
            if not assets:
                reasons.append("Claim has no real image assets")
            for asset in assets:
                try:
                    image = load_validated_image(asset.path)
                    image.close()
                except Exception as error:
                    reasons.append(f"{asset.filename}: {type(error).__name__}: {error}")
            if reasons:
                self._skip(observed, "vision", "; ".join(reasons))
            else:
                service = ClaimMultimodalVisionService(
                    self.resources["damage_detector"], self.resources["part_detector"]
                )
                quoted = map_quote_damage_types(quote)[0] if quote is not None else ()
                vision = self._record(
                    observed,
                    "vision",
                    lambda: service.assess_claim(case, quoted_damage_types=quoted),
                    inputs={"declared_damage": [item.value for item in case.claim.declared_damage]},
                    sources=[str(asset.path) for asset in assets],
                )
                self._capture_vision(observed, vision)

        self._investigation(
            observed, consistency=consistency, features=features, anomaly=anomaly, history=history, vision=vision
        )
        if scenario.inputs.get("history_claims") and history is None:
            observed.components["investigation"].status = EvaluationStatus.SKIP
            observed.components[
                "investigation"
            ].reason = "Required historical context unavailable; observed partial signal assessment retained"
        if self.config.isolated_demo_stack:
            from evaluation.isolated_stack import seed_case
            from ml.anomaly_inference import RegisteredAnomalyAssessment

            registered_anomaly = registered_anomaly_result or (
                RegisteredAnomalyAssessment(
                    assessment=anomaly,
                    registered_model="local:anomaly_detector.joblib",
                    model_version=observed.model_versions.get("anomaly", "unknown"),
                    model_alias="local-artifact",
                )
                if anomaly is not None
                else None
            )
            from investigation.models import InvestigationAssessment

            assessment = (
                InvestigationAssessment.model_validate(observed.investigation) if observed.investigation else None
            )
            unavailable = set(scenario.inputs.get("unavailable_tools", []))
            self._record(
                observed,
                "assessment_persistence",
                lambda: seed_case(
                    self.resources["session_factory"],
                    case,
                    triage=None if "GET_TRIAGE" in unavailable else triage,
                    anomaly=None if "GET_ANOMALY" in unavailable else registered_anomaly,
                    investigation=None if "GET_INVESTIGATION" in unavailable else assessment,
                    vision=None if "GET_VISUAL_ASSESSMENT" in unavailable else vision,
                ),
                inputs={"claim_id": case.claim.claim_id, "unavailable_assessments": sorted(unavailable)},
            )
        self._retrieval(scenario, observed)
        self._similar(case, observed)
        self._agent(scenario, case, observed, use_reference=use_reference)
        self._finish(observed, started)
        return observed

    def _capture_vision(self, observed, vision):
        if vision is None:
            return
        observed.vision = vision.model_dump(mode="json")
        observed.facts["vision"] = observed.vision
        observed.model_versions.update(
            damage_vision=vision.damage_model_metadata.version, part_vision=vision.part_model_metadata.version
        )
        observed.policy_versions["fusion"] = vision.fusion_policy.version
        observed.provenance["vision"] = {
            "images": [image.filename for image in vision.images],
            "damage_model": vision.damage_model_metadata.model_dump(mode="json"),
            "part_model": vision.part_model_metadata.model_dump(mode="json"),
            "geometry": "Production schema validates normalized geometry and source identity",
        }

    def _investigation(self, observed, *, consistency=None, features=None, anomaly=None, history=None, vision=None):
        from investigation.models import InvestigationAssessment
        from investigation.policy import INVESTIGATION_POLICY_V2, INVESTIGATION_POLICY_V3
        from investigation.signals import (
            signals_from_anomaly,
            signals_from_completeness,
            signals_from_consistency,
            signals_from_history,
            signals_from_multimodal,
        )

        policy = INVESTIGATION_POLICY_V3 if vision is not None else INVESTIGATION_POLICY_V2
        sources = {
            "consistency": consistency,
            "features": features,
            "anomaly": anomaly,
            "history": history,
            "vision": vision,
        }
        omitted = [key for key, value in sources.items() if value is None]
        if all(value is None for value in sources.values()):
            self._skip(observed, "investigation", "No observed signal inputs are available")
            return

        def assess_available_inputs():
            """
            Reuse the application's independent translators and policy without inventing a
            negative anomaly assessment or an empty history.
            """
            if all(value is not None for value in (consistency, features, anomaly, history)):
                from investigation.engine import assess_investigation

                return assess_investigation(
                    consistency=consistency,
                    anomaly=anomaly,
                    features=features,
                    history=history,
                    policy=policy,
                    visual_assessment=vision,
                )
            signals = (
                *(signals_from_consistency(consistency) if consistency is not None else ()),
                *(signals_from_anomaly(anomaly, policy) if anomaly is not None else ()),
                *(signals_from_completeness(features, policy) if features is not None else ()),
                *(signals_from_history(history, policy) if history is not None else ()),
                *(signals_from_multimodal(vision, policy) if vision is not None else ()),
            )
            score = policy.score_signals(signals)
            return InvestigationAssessment(
                review_score=score,
                review_priority=policy.priority_for_score(score),
                review_recommended=policy.review_recommended_for_score(score),
                policy_version=policy.version,
                signals=signals,
            )

        assessment = self._record(
            observed,
            "investigation",
            assess_available_inputs,
            inputs={
                "available_inputs": [key for key, value in sources.items() if value is not None],
                "unavailable_inputs": omitted,
            },
        )
        observed.components["investigation"].metadata.update(partial=bool(omitted), unavailable_inputs=omitted)
        if assessment is not None:
            observed.investigation = assessment.model_dump(mode="json")
            observed.facts["investigation"] = observed.investigation
            observed.signals = [item.code.value for item in assessment.signals]
            observed.policy_versions["investigation"] = assessment.policy_version

    def _retrieval(self, scenario, observed):
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
        observed.rag = {"results": [_payload(item) for item in results], "question": question, "answer": None}
        if scenario.inputs.get("rag_supported") and scenario.inputs.get("rag_section"):
            observed.rag["retrieval_metrics"] = section_metrics(
                results, scenario.inputs.get("rag_source_id"), [scenario.inputs["rag_section"]]
            )
        observed.provenance["rag"] = {"retrieved_chunk_ids": [item.chunk_id for item in results]}
        if self._available(observed, "rag_generation", ["ollama"]):
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

    def _similar(self, case, observed):
        if not self.config.include_similar:
            self._skip(observed, "similar_claims", "Disabled by configuration")
            return
        if not self._available(
            observed, "similar_claims", ["postgresql", "qdrant", "similar_claims_collection", "embeddings"]
        ):
            return
        if observed.components.get("history") and observed.components["history"].status != "PASS":
            self._skip(observed, "similar_claims", "Controlled claim has no matching stored source")
            return
        from persistence.retrieval import RetrievalClaimRepository
        from retrieval.embeddings import SentenceTransformerEmbeddingProvider
        from retrieval.qdrant_store import QdrantClaimStore
        from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService

        config = self.resources["retrieval_config"]
        embeddings = self.resources.get("similar_embeddings")
        if embeddings is None:
            embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, local_files_only=True)
            self.resources["similar_embeddings"] = embeddings
        repository = RetrievalClaimRepository(self.resources["session_factory"])
        store = QdrantClaimStore(self.resources["qdrant_client"], config.collection, embeddings)
        service = SimilarClaimsService(repository, embeddings, store, same_claim_type=config.same_claim_type)
        if self.config.isolated_demo_stack:
            self._record(
                observed,
                "similar_indexing",
                lambda: SimilarClaimsIndexer(repository, embeddings, store).index_all(),
                inputs={"collection": config.collection, "scope": "Controlled claims in this run's isolated schema"},
            )
        result = self._record(
            observed,
            "similar_claims",
            lambda: service.search(case.claim.claim_id),
            inputs={"claim_id": case.claim.claim_id},
        )
        if result is not None:
            observed.facts["similar_claims"] = result.model_dump(mode="json")

    def _agent(self, scenario, case, observed, *, use_reference):
        """
        The normal application factory supplies lazy real services. Closing here only closes those
        owned by this service, not preflight clients.
        """
        if not self.config.include_agent:
            self._skip(observed, "agent", "Disabled by configuration")
            return
        if not self._available(observed, "agent", ["postgresql", "ollama"]):
            return
        unavailable = set(scenario.inputs.get("unavailable_tools", []))
        missing_assessment_tools = {"GET_TRIAGE", "GET_ANOMALY", "GET_INVESTIGATION", "GET_VISUAL_ASSESSMENT"}
        if unavailable and (not self.config.isolated_demo_stack or unavailable - missing_assessment_tools):
            self._skip(
                observed,
                "agent",
                "Requested deterministic tool outage cannot be safely imposed on the real application service; "
                "no fake tools used",
            )
            return
        if (
            scenario.vision_required
            and observed.components.get("vision")
            and observed.components["vision"].status == "SKIP"
        ):
            self._skip(
                observed,
                "agent",
                "This golden case requires unavailable real visual ground truth; its document subset remains observed",
            )
            return
        if observed.components.get("history") and observed.components["history"].status != "PASS":
            self._skip(
                observed, "agent", "Controlled claim has no matching stored source; no fake agent tool backend used"
            )
            return
        from agent.factory import create_claims_agent_service

        with _llm_environment(self.resources["llm_model"]):
            service = create_claims_agent_service(
                self.resources["session_factory"],
                similar_service_loader=self._agent_similar_service,
                knowledge_retriever_loader=self._agent_knowledge_retriever,
            )

            def review():
                from agent.service import AgentRunError
                from persistence.agent_runs import AgentRunStore

                try:
                    return service.review_claim(
                        case.claim.claim_id,
                        objective="Review the controlled demonstration claim and recommend the next human review step.",
                    )
                except AgentRunError as error:
                    result = AgentRunStore(self.resources["session_factory"]).get_by_run_id(error.run_id)
                    if result is None:
                        raise
                    return result

            result = self._record(
                observed,
                "agent",
                review,
                inputs={"claim_id": case.claim.claim_id, "source": "existing PostgreSQL claim"},
            )
            service.tools.close()
        if result is not None:
            observed.agent = result.model_dump(mode="json")
            observed.facts["agent"] = observed.agent
            observed.model_versions["agent_llm"] = self.resources["llm_model"]
            if result.status != "COMPLETED":
                observed.components["agent"].status = EvaluationStatus.FAIL
                observed.components["agent"].reason = result.error or "Real agent did not complete"
                observed.errors.append("agent: " + observed.components["agent"].reason)

    def _agent_knowledge_retriever(self):
        from retrieval.models import RetrievalUnavailableError

        if not self.config.include_rag or not all(
            self.preflight.available(name) for name in ("qdrant", "knowledge_collections", "embeddings")
        ):
            raise RetrievalUnavailableError("Knowledge retrieval unavailable or disabled by evaluation configuration")
        from qdrant_client import QdrantClient

        from knowledge.qdrant_store import QdrantKnowledgeStore
        from knowledge.retrieval import KnowledgeRetriever

        config = self.resources["knowledge_config"]
        client = QdrantClient(
            url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
        )
        return KnowledgeRetriever(QdrantKnowledgeStore(client, self.resources["embeddings"], config))

    def _agent_similar_service(self):
        from retrieval.models import RetrievalUnavailableError

        if not self.config.include_similar or not all(
            self.preflight.available(name) for name in ("qdrant", "similar_claims_collection", "embeddings")
        ):
            raise RetrievalUnavailableError("Similar claims unavailable or disabled by evaluation configuration")
        from qdrant_client import QdrantClient

        from persistence.retrieval import RetrievalClaimRepository
        from retrieval.embeddings import SentenceTransformerEmbeddingProvider
        from retrieval.qdrant_store import QdrantClaimStore
        from retrieval.similar_claims import SimilarClaimsService

        config = self.resources["retrieval_config"]
        embeddings = self.resources.get("similar_embeddings")
        if embeddings is None:
            embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, local_files_only=True)
            self.resources["similar_embeddings"] = embeddings
        client = QdrantClient(
            url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
        )
        return SimilarClaimsService(
            RetrievalClaimRepository(self.resources["session_factory"]),
            embeddings,
            QdrantClaimStore(client, config.collection, embeddings),
            same_claim_type=config.same_claim_type,
        )

    def _diagnostic_vision(self, scenario, observed):
        for component in (
            "claim",
            "document",
            "consistency",
            "features",
            "triage",
            "anomaly",
            "history",
            "retrieval",
            "rag_generation",
            "similar_claims",
            "agent",
        ):
            self._skip(
                observed, component, "Standalone local external-image diagnostic; no document/claim/LLM ground truth"
            )
        if not self.config.include_vision:
            self._skip(observed, "vision", "Disabled by configuration")
            self._skip(observed, "investigation", "Vision disabled")
            return
        if not self._available(observed, "vision", ["damage_vision", "part_vision"]):
            self._skip(observed, "investigation", "Vision dependencies unavailable")
            return
        from claims.enums import DamageType
        from scripts.check_multimodal_integration_smoke import diagnostic_case, select_sources
        from vision.fusion_config import DEFAULT_FUSION_POLICY
        from vision.models import ClaimMultimodalVisionAssessment
        from vision.multimodal_service import ClaimMultimodalVisionService, assess_multimodal_consistency

        def infer():
            dataset_root, sources = select_sources(
                self.config.source_summary, self.config.training_summary, max_images=self.config.diagnostic_max_images
            )
            declared = [DamageType(item) for item in self.config.synthetic_declared_damage]
            quoted = [DamageType(item) for item in self.config.synthetic_quoted_damage]
            service = ClaimMultimodalVisionService(self.resources["damage_detector"], self.resources["part_detector"])
            images = []
            for index, source in enumerate(sources, 1):
                case = diagnostic_case(dataset_root, source, index, declared)
                result = service.assess_claim(case, quoted_damage_types=quoted)
                images.extend(result.images)
            observed.provenance["external_assets"] = [
                {
                    "path": str(source.path),
                    "source_sample_id": source.source_sample_id,
                    "source_summary": str(self.config.source_summary),
                    "redistributed": False,
                }
                for source in sources
            ]
            observed.provenance["vision_context"] = {
                "declaration_is_synthetic": True,
                "quoted_damage_is_synthetic": True,
                "declared_damage": list(self.config.synthetic_declared_damage),
                "ground_truth_available": False,
                "detection_truth_labels_loaded": False,
                "meaning": "Conditional integration diagnostic; does not establish facts about an insurance claim",
                "existing_visualizations": "artifacts/environment_merge/gpu-smoke/"
                "run_20261015T135812_022547Z/summary.json",
            }
            return ClaimMultimodalVisionAssessment(
                claim_id="E2E-DIAGNOSTIC-CARDD",
                images=images,
                damage_model_metadata=service.damage_detector.model_metadata,
                part_model_metadata=service.part_detector.model_metadata,
                fusion_policy=DEFAULT_FUSION_POLICY,
                consistency=assess_multimodal_consistency(
                    images, declared_damage_types=declared, quoted_damage_types=quoted
                ),
            )

        vision = self._record(
            observed,
            "vision",
            infer,
            inputs={
                "synthetic_declared_damage": list(self.config.synthetic_declared_damage),
                "ground_truth_available": False,
            },
            sources=[str(self.config.source_summary), str(self.config.training_summary)],
        )
        self._capture_vision(observed, vision)
        self._investigation(observed, vision=vision)

    @staticmethod
    def _finish(observed, started):
        observed.components["total"] = ComponentResult(
            status="PASS",
            duration_seconds=perf_counter() - started,
            metadata={"meaning": "wall clock completion; inspect individual components for actual availability"},
        )


@contextmanager
def _llm_environment(model):
    previous = os.environ.get("CLAIMS_LLM_MODEL")
    os.environ["CLAIMS_LLM_MODEL"] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CLAIMS_LLM_MODEL", None)
        else:
            os.environ["CLAIMS_LLM_MODEL"] = previous


def real_diagnostic_scenarios() -> list[EvaluationScenario]:
    """Diagnostics supplement the frozen golden manifest without invented truth."""
    path = ROOT / "data/evaluation/e2e_v1/diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["benchmark_version"] != "e2e_claims_v1":
        raise ValueError("Unsupported diagnostic manifest version")
    return [EvaluationScenario.model_validate(item) for item in payload["diagnostics"]]


@contextmanager
def isolated_demo_stack(config, preflight, output_dir):
    """Prepare explicitly requested ephemeral real services using existing indexers.

    Only a safe dedicated test database and new randomly named collections are
    mutated. Existing app data and existing collections are never imported or
    overwritten. All created resources are removed on context exit.
    """
    from uuid import uuid4

    from evaluation.isolated_stack import isolated_postgres

    if not config.database_url or not config.qdrant_url:
        raise ValueError("--isolated-demo-stack requires explicit --database-url and --qdrant-url")
    if not preflight.available("postgresql") or not preflight.available("qdrant"):
        raise RuntimeError("Isolated demo setup requires both explicitly selected services to pass preflight")
    if not preflight.available("embeddings"):
        raise RuntimeError("Isolated real indexing requires a locally loadable embedding model")
    suffix = uuid4().hex
    names = [f"evaluation_e2e_{kind}_{suffix}" for kind in ("policy", "procedure", "claims")]
    output_dir = Path(output_dir)
    client = preflight.resources["qdrant_client"]
    audit = {
        "scope": "Explicit isolated demo evaluation; no existing application rows or collections modified",
        "collections": names,
        "cleaned_up": False,
    }

    def write_audit():
        (output_dir / "isolated_setup.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    with isolated_postgres(config.database_url) as stack:
        original_engine = preflight.resources.get("database_engine")
        if original_engine is not None:
            original_engine.dispose()
        preflight.resources.update(
            database_engine=stack.engine, session_factory=stack.factory, isolated_schema=stack.schema
        )
        audit["schema"] = stack.schema
        knowledge_config = replace(
            preflight.resources["knowledge_config"], policy_collection=names[0], procedure_collection=names[1]
        )
        retrieval_config = replace(preflight.resources["retrieval_config"], collection=names[2])
        preflight.resources.update(knowledge_config=knowledge_config, retrieval_config=retrieval_config)
        try:
            from knowledge.indexing import KnowledgeIndexer
            from knowledge.qdrant_store import QdrantKnowledgeStore
            from retrieval.embeddings import SentenceTransformerEmbeddingProvider
            from retrieval.qdrant_store import QdrantClaimStore

            if config.include_rag:
                manifest = KnowledgeIndexer(
                    QdrantKnowledgeStore(client, preflight.resources["embeddings"], knowledge_config)
                ).index_all(output_dir / "isolated_knowledge_index_manifest.json")
                audit["knowledge_index"] = manifest
            if config.include_similar:
                embeddings = SentenceTransformerEmbeddingProvider(
                    retrieval_config.embedding_model, local_files_only=True
                )
                preflight.resources["similar_embeddings"] = embeddings
                QdrantClaimStore(client, names[2], embeddings).ensure_collection()
            for row in preflight.rows:
                if row["component"] == "postgresql":
                    row.update(schema_available=True, isolated_schema=stack.schema)
                elif row["component"] == "knowledge_collections" and config.include_rag:
                    row.update(
                        status="PASS",
                        reason=None,
                        collections=names[:2],
                        setup="New owned isolated demo collections indexed using real embeddings",
                    )
                elif row["component"] == "similar_claims_collection" and config.include_similar:
                    row.update(
                        status="PASS",
                        reason=None,
                        collections=names[2:],
                        setup="New owned isolated demo collection; "
                        "indexed from run's real PostgreSQL rows per scenario",
                    )
            write_audit()
            yield
        finally:
            cleanup_errors = []
            for name in names:
                try:
                    if client.collection_exists(name):
                        client.delete_collection(name)
                except Exception as error:
                    cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
            audit.update(
                cleaned_up=not cleanup_errors,
                collection_cleanup_errors=cleanup_errors,
                schema_cleanup="Owned schema is dropped by the surrounding PostgreSQL context",
            )
            write_audit()
