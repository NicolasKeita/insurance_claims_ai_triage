"""Thin benchmark orchestration over existing application/domain services.

Only external boundaries are synthetic in ``ClaimsEvaluationPipeline``. Fixture
materialization is shared with the real backend and never rewrites source assets.
"""

import json
from contextlib import closing
from datetime import date as document_date
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter

import pymupdf

from claims.assets import resolve_document_assets
from claims.case import ClaimCase
from claims.consistency import compare_claim_documents, map_quote_damage_types
from claims.document_extraction import extract_accident_report, extract_claim_form, extract_garage_quote
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from evaluation.models import ComponentResult, PipelineObservation, TraceEvent
from ml.features import build_triage_features

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_PROVENANCE = "Generated controlled synthetic E2E fixture; no third-party imagery or real personal data."


def _pdf(path: Path, text: str):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((45, 45), text, fontsize=10)
    document.save(path)
    document.close()


def materialize_scenario_case(scenario, root_dir=None) -> ClaimCase:
    """Create actual demo PDFs and optional generated pixels in an isolated dir.

    The caller owns cleanup of ``root_dir``. If omitted, a fresh temporary dir is
    returned through ``ClaimCase.root_dir``. No inference or fake model is used.

    General overrides are fixture inputs, never inferred from expectations.

    Amount arithmetic remains exact and the real quote parser verifies it.
    """
    root = Path(root_dir) if root_dir is not None else Path(mkdtemp(prefix="claims-e2e-"))
    root.mkdir(parents=True, exist_ok=True)
    source = Path(scenario.claim_source)
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir():
        source /= "claim.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    inputs = scenario.inputs
    payload["claim_id"] = inputs.get("claim_id", f"DEMO-E2E-{scenario.scenario_id.upper()}")
    payload["customer"]["customer_id"] = f"DEMO-E2E-CUSTOMER-{scenario.scenario_id}"
    payload["policy"]["policy_id"] = f"DEMO-E2E-POLICY-{scenario.scenario_id}"
    payload["declared_damage"] = inputs.get("declared_damage", ["FRONT_BUMPER"])
    quoted = inputs.get("quote_damage", payload["declared_damage"])
    missing = set(inputs.get("missing_documents", []))
    filenames = {
        "CLAIM_FORM": "claim_form.pdf",
        "ACCIDENT_REPORT": "accident_report.pdf",
        "REPAIR_QUOTE": "garage_quote.pdf",
        "POLICE_REPORT": "police_report.pdf",
    }
    payload["documents"] = [
        {"type": kind, "filename": name if kind not in missing else None, "available": kind not in missing}
        for kind, name in filenames.items()
    ]
    amount = Decimal(str(inputs.get("repair_amount", 1500)))
    payload["repair_estimate"]["amount"] = str(amount)
    payload["images"] = ["synthetic_geometry.png"] if scenario.vision_required else []
    for key, value in inputs.get("claim_overrides", {}).items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    docs = root / "documents"
    docs.mkdir(exist_ok=True)
    date = document_date.fromisoformat(payload["incident"]["date"]).strftime("%d/%m/%Y")
    damage = "\n".join(item.lower().replace("_", " ") for item in payload["declared_damage"])
    vehicle = f"{payload['vehicle']['make']} {payload['vehicle']['model']}"
    injuries = "Yes" if payload["incident"]["injuries_declared"] else "No"
    texts = {
        "CLAIM_FORM": f"DEMO SYNTHETIC CLAIM FORM\nClaim ID: {payload['claim_id']}\nIncident date: {date}\n"
        f"Location: {payload['incident']['location']}\nVehicle: {vehicle}\n"
        f"Vehicle year: {payload['vehicle']['year']}\nCollision type: Front collision\n"
        f"Declared damage:\n{damage}\nInjuries declared: {injuries}",
        "ACCIDENT_REPORT": f"DEMO SYNTHETIC ACCIDENT REPORT\nClaim ID: {payload['claim_id']}\nDate: {date}\n"
        f"Location: {payload['incident']['location']}\nVehicle:\n{vehicle}\n"
        f"Collision type: Front collision\nReported damage:\n{damage}\nPassengers: 1\nInjuries: {injuries}",
        "POLICE_REPORT": f"DEMO SYNTHETIC SUPPORTING RECORD\nClaim ID: {payload['claim_id']}\n"
        "No inference of misconduct is made.",
    }
    part_amount = (amount / Decimal(max(1, len(quoted) + 1))).quantize(Decimal("0.01"))
    labor = amount - part_amount * len(quoted)
    parts = "\n".join(f"{item.lower().replace('_', ' ')}: {part_amount} EUR" for item in quoted)
    texts["REPAIR_QUOTE"] = (
        f"DEMO SYNTHETIC GARAGE QUOTE\nGarage: Demo Garage\nClaim ID: {payload['claim_id']}\n{parts}\n"
        f"Labor: {labor} EUR\nTotal: {amount} EUR"
    )
    for kind, overrides in inputs.get("document_overrides", {}).items():
        if isinstance(overrides, str):
            texts[kind] = overrides
        else:
            for old, new in overrides.items():
                texts[kind] = texts[kind].replace(old, str(new))
    for kind, filename in filenames.items():
        if kind not in missing:
            _pdf(docs / filename, texts[kind])
    if payload["images"]:
        from PIL import Image, ImageDraw

        images = root / "images"
        images.mkdir(exist_ok=True)
        with Image.new("RGB", (128, 128), "#d8dee3") as image:
            drawing = ImageDraw.Draw(image)
            drawing.rectangle((12, 12, 116, 116), outline="#304050", width=2)
            drawing.line((44, 44, 64, 62), fill="#8b3344", width=3)
            image.save(images / payload["images"][0])
    (root / "claim.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "fixture_provenance.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "source_claim": str(source),
                "provenance": SYNTHETIC_PROVENANCE,
                "vision_semantics": "Generated pixels exercise geometry only; "
                "fake detections are explicit test inputs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return load_claim_case(root)


def extract_case_documents(case):
    """Real native PDF ingestion plus existing structured extractors."""
    ingested, extracted = {}, {}
    extractors = {
        DocumentType.CLAIM_FORM: extract_claim_form,
        DocumentType.ACCIDENT_REPORT: extract_accident_report,
        DocumentType.REPAIR_QUOTE: extract_garage_quote,
    }
    for asset in resolve_document_assets(case):
        text = extract_pdf_text(asset)
        ingested[asset.filename] = text
        if asset.type in extractors:
            extracted[asset.type] = extractors[asset.type](text)
    return extracted, ingested


def fake_vision_service(case, fixture):
    """Construct only explicit test observations; the fusion service is real."""
    from vision.detector import FakeDamageDetector
    from vision.models import NormalizedBoundingBox, NormalizedPolygon, VehiclePartDetection, VisualDamageDetection
    from vision.multimodal_service import ClaimMultimodalVisionService
    from vision.parts_detector import FakeVehiclePartDetector

    detections, parts = [], []
    observations = fixture.get("detections", [fixture] if fixture.get("damage_type") else [])
    for item in observations:
        box = item.get("damage_box", [0.3, 0.3, 0.5, 0.5])
        part_box = item.get("part_box", [0.1, 0.1, 0.9, 0.9])

        def geometry(values):
            x1, y1, x2, y2 = values
            return {
                "bounding_box": NormalizedBoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                "polygon": NormalizedPolygon(
                    points=[{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]
                ),
            }

        detections.append(
            VisualDamageDetection(
                damage_type=item.get("damage_type", "SCRATCH").lower(),
                confidence=item.get("damage_confidence", 0.95),
                model_identifier="fake-damage-detector",
                model_version="test-only",
                **geometry(box),
            )
        )
        parts.append(
            VehiclePartDetection(
                part_type=item.get("vehicle_part", "HOOD").lower(),
                confidence=item.get("part_confidence", 0.96),
                **geometry(part_box),
            )
        )
    name = case.claim.images[0] if case.claim.images else "synthetic_geometry.png"
    return ClaimMultimodalVisionService(FakeDamageDetector({name: detections}), FakeVehiclePartDetector({name: parts}))


class ClaimsEvaluationPipeline:
    """Deterministic E2E adapter. Real evaluation uses RealClaimsEvaluationPipeline."""

    def __init__(self, *, triage_predictor=None, anomaly_predictor=None):
        self.triage_predictor, self.anomaly_predictor = triage_predictor, anomaly_predictor

    def run(self, scenario):
        """
        Use the real Qdrant stores/index/query orchestration in local memory.

        Preserve partial observations for useful expectation diagnostics.
        """
        from qdrant_client import QdrantClient

        from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
        from agent.service import ClaimsAgentService
        from evaluation.fakes import (
            FakeAnomalyPredictor,
            FakeEmbeddings,
            FakeHistoricalRepository,
            FakeStructuredLlm,
            FakeTriagePredictor,
            InMemoryRetrievalRepository,
            SnapshotAgentTools,
            historical_claims,
        )
        from investigation.engine import assess_investigation
        from investigation.policy import INVESTIGATION_POLICY_V2, INVESTIGATION_POLICY_V3
        from knowledge.config import KnowledgeConfig
        from knowledge.models import KnowledgeSourceType
        from knowledge.qdrant_store import QdrantKnowledgeStore
        from knowledge.rag import KnowledgeRagService
        from knowledge.retrieval import KnowledgeRetriever
        from knowledge.sources import KnowledgeCatalog
        from retrieval.qdrant_store import QdrantClaimStore
        from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService

        output = PipelineObservation()

        def record(name, call, *, inputs=None, sources=None):
            started = perf_counter()
            try:
                result = call()
            except Exception as error:
                duration = perf_counter() - started
                output.components[name] = ComponentResult(
                    status="FAIL", duration_seconds=duration, reason=f"{type(error).__name__}: {error}"
                )
                output.errors.append(f"{name}: {type(error).__name__}: {error}")
                output.trace.append(
                    TraceEvent(
                        component=name,
                        status="FAIL",
                        duration_seconds=duration,
                        inputs=inputs or {},
                        outputs={"error_type": type(error).__name__},
                    )
                )
                raise
            duration = perf_counter() - started
            if isinstance(result, ClaimCase):
                snapshot = {"claim": result.claim.model_dump(mode="json"), "fixture_provenance": SYNTHETIC_PROVENANCE}
            elif name == "document":
                extracted_models, pdf_texts = result
                snapshot = {
                    "structured_extractions": {
                        kind.value: model.model_dump(mode="json") for kind, model in extracted_models.items()
                    },
                    "documents": [
                        {
                            "filename": filename,
                            "document_type": text.asset.type.value,
                            "page_count": text.page_count,
                            "method": text.method.value,
                            "pages": list(text.pages),
                            "text_sha256": sha256(text.text.encode("utf-8")).hexdigest(),
                            "asset_sha256": sha256(text.asset.path.read_bytes()).hexdigest(),
                        }
                        for filename, text in pdf_texts.items()
                    ],
                }
            else:
                snapshot = (
                    result.model_dump(mode="json")
                    if hasattr(result, "model_dump")
                    else (result if isinstance(result, dict) else {"completed": True})
                )
            output.components[name] = ComponentResult(
                status="PASS", duration_seconds=duration, metadata={"backend": "deterministic"}
            )
            output.trace.append(
                TraceEvent(
                    component=name,
                    status="PASS",
                    inputs=inputs or {},
                    outputs=snapshot,
                    duration_seconds=duration,
                    source_references=sources or [],
                )
            )
            return result

        with (
            TemporaryDirectory(prefix="claims-e2e-", dir=ROOT) as workspace,
            closing(QdrantClient(":memory:")) as client,
        ):
            try:
                case = record(
                    "claim",
                    lambda: materialize_scenario_case(scenario, workspace),
                    inputs={"claim_source": scenario.claim_source},
                )
                output.facts["claim"] = case.claim.model_dump(mode="json")
                extracted, ingested = record(
                    "document",
                    lambda: extract_case_documents(case),
                    inputs={
                        "claim_id": case.claim.claim_id,
                        "fixture_source": scenario.claim_source,
                        "document_entries": [d.model_dump(mode="json") for d in case.claim.documents],
                    },
                    sources=[d.filename for d in case.claim.documents if d.available],
                )
                form, report, quote = (
                    extracted[kind]
                    for kind in (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)
                )
                output.facts["extraction"] = {
                    "success": True,
                    "form": form.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                    "quote": quote.model_dump(mode="json"),
                }
                output.provenance["documents"] = [
                    {
                        **item.source.model_dump(mode="json"),
                        "asset_sha256": sha256(ingested[item.source.filename].asset.path.read_bytes()).hexdigest(),
                        "text_sha256": sha256(ingested[item.source.filename].text.encode("utf-8")).hexdigest(),
                        "fixture_source": scenario.claim_source,
                        "synthetic": True,
                    }
                    for item in (form, report, quote)
                ]
                output.provenance["fixture"] = {"kind": "controlled-synthetic", "description": SYNTHETIC_PROVENANCE}
                consistency = record("consistency", lambda: compare_claim_documents(form, report, quote))
                output.facts["consistency"] = {
                    **consistency.model_dump(mode="json"),
                    "is_consistent": consistency.is_consistent,
                }
                features = record("features", lambda: build_triage_features(case, form, quote, consistency))
                output.facts["features"] = features.model_dump(mode="json")
                triage = record("triage", lambda: (self.triage_predictor or FakeTriagePredictor()).predict(features))
                anomaly = record(
                    "anomaly",
                    lambda: (
                        self.anomaly_predictor or FakeAnomalyPredictor(scenario.inputs.get("anomaly_fixture"))
                    ).assess(features),
                )
                output.facts["triage"] = triage.model_dump(mode="json")
                output.facts["anomaly"] = {
                    **anomaly.assessment.model_dump(mode="json"),
                    **anomaly.model_dump(mode="json", exclude={"assessment"}),
                }
                output.model_versions.update(triage=triage.model_version, anomaly=anomaly.model_version)
                history_inputs = historical_claims(case.claim, scenario.inputs.get("history_claims", []))
                history = record(
                    "history",
                    lambda: FakeHistoricalRepository(case.claim, history_inputs).get_profile(case.claim.claim_id),
                )
                output.facts["history"] = history.model_dump(mode="json")
                visual = None
                if scenario.vision_required:
                    visual = record(
                        "vision",
                        lambda: fake_vision_service(case, scenario.inputs.get("vision_fixture", {})).assess_claim(
                            case, quoted_damage_types=map_quote_damage_types(quote)[0]
                        ),
                    )
                    output.vision = {
                        **visual.model_dump(mode="json"),
                        "observations": [item.model_dump(mode="json") for item in visual.observations],
                    }
                    output.facts["vision"] = output.vision
                    output.model_versions.update(
                        damage=visual.damage_model_metadata.version or "unknown",
                        parts=visual.part_model_metadata.version or "unknown",
                    )
                    output.policy_versions["fusion"] = visual.fusion_policy.version
                    output.provenance["images"] = [
                        {
                            "filename": filename,
                            "synthetic": True,
                            "asset_sha256": sha256(case.image_path(filename).read_bytes()).hexdigest(),
                            "description": "Generated geometry-only pixels; "
                            "detector observations supplied explicitly by the scenario.",
                        }
                        for filename in case.claim.images
                    ]
                else:
                    output.components["vision"] = ComponentResult(
                        status="SKIP",
                        reason="Document-centric scenario; no vision fixture requested.",
                        metadata={"applicable": False},
                    )
                    output.trace.append(
                        TraceEvent(
                            component="vision", status="SKIP", outputs={"reason": output.components["vision"].reason}
                        )
                    )
                policy = INVESTIGATION_POLICY_V3 if visual is not None else INVESTIGATION_POLICY_V2
                investigation = record(
                    "investigation",
                    lambda: assess_investigation(
                        consistency=consistency,
                        anomaly=anomaly.assessment,
                        features=features,
                        history=history,
                        policy=policy,
                        visual_assessment=visual,
                    ),
                )
                output.investigation = investigation.model_dump(mode="json")
                output.facts["investigation"] = output.investigation
                output.signals = [signal.code.value for signal in investigation.signals]
                output.policy_versions["investigation"] = investigation.policy_version
                embeddings = FakeEmbeddings()
                repository = InMemoryRetrievalRepository([case.claim, *(claim for claim, _ in history_inputs)])
                claim_store = QdrantClaimStore(client, "e2e_claims", embeddings)
                SimilarClaimsIndexer(repository, embeddings, claim_store).index_all()
                similar = SimilarClaimsService(repository, embeddings, claim_store)
                retrieved = record("similar_claims", lambda: similar.search(case.claim.claim_id))
                output.facts["similar_claims"] = retrieved.model_dump(mode="json")
                config = KnowledgeConfig(knowledge_dir=ROOT / "data/knowledge")
                knowledge_store = QdrantKnowledgeStore(client, embeddings, config)
                catalog = KnowledgeCatalog.load(config.knowledge_dir, config.max_chunk_chars)
                for kind in KnowledgeSourceType:
                    knowledge_store.ensure_collection(kind)
                    chunks = [chunk for chunk in catalog.chunks.values() if chunk.source_type == kind]
                    knowledge_store.upsert(kind, chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
                retriever = KnowledgeRetriever(knowledge_store)
                llm = FakeStructuredLlm(
                    vision=scenario.vision_required, rag_supported=scenario.inputs.get("rag_supported", True)
                )
                question = scenario.inputs.get("rag_question", "What is the demo collision deductible?")
                chunks = record(
                    "retrieval",
                    lambda: {
                        "results": [
                            chunk.model_dump(mode="json")
                            for chunk in retriever.search(
                                question,
                                KnowledgeSourceType.POLICY,
                                product=case.claim.policy.product,
                                source_id=scenario.inputs.get("rag_source_id"),
                                as_of_date=case.claim.incident.date,
                                limit=5,
                            )
                        ]
                    },
                )
                from knowledge.models import RetrievedKnowledgeChunk

                answer = record(
                    "rag_generation",
                    lambda: KnowledgeRagService(retriever, llm).answer_from_chunks(
                        question, [RetrievedKnowledgeChunk.model_validate(chunk) for chunk in chunks["results"]]
                    ),
                )
                output.rag = {
                    "answer": answer.model_dump(mode="json"),
                    "retrieved_chunks": chunks["results"],
                    "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks["results"]],
                }
                if scenario.expected_knowledge_sources and scenario.inputs.get("rag_section"):
                    from knowledge.evaluation import section_metrics

                    output.rag["retrieval_metrics"] = section_metrics(
                        [RetrievedKnowledgeChunk.model_validate(chunk) for chunk in chunks["results"]],
                        scenario.expected_knowledge_sources[0],
                        [scenario.inputs["rag_section"]],
                    )
                output.facts["rag"] = output.rag
                snapshots = {
                    "triage": output.facts["triage"],
                    "anomaly": anomaly.model_dump(mode="json"),
                    "history": output.facts["history"],
                    "investigation": output.investigation,
                    "vision": output.vision,
                }
                tools = SnapshotAgentTools(
                    case.claim,
                    snapshots,
                    similar_service=similar,
                    knowledge_retriever=retriever,
                    unavailable=scenario.inputs.get("unavailable_tools", []),
                )
                agent = record(
                    "agent",
                    lambda: ClaimsAgentService(
                        tools,
                        StructuredAgentPlanner(llm),
                        StructuredAgentFinalizer(llm),
                        max_iterations=8,
                        llm_model="fake-structured-e2e-v1",
                    ).review_claim(case.claim.claim_id),
                )
                output.agent = agent.model_dump(mode="json")
                output.facts["agent"] = output.agent
                output.model_versions.update(embeddings=embeddings.model_name, llm="fake-structured-e2e-v1")
            except Exception as error:
                if not output.errors:
                    output.errors.append(f"orchestration: {type(error).__name__}: {error}")
        return output
