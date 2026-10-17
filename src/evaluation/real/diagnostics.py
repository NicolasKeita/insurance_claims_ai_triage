"""Standalone external-image diagnostic steps and manifest."""

import json

from evaluation.models import EvaluationScenario
from evaluation.preflight import ROOT

DIAGNOSTIC_COMPONENTS = (
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
)


def real_diagnostic_scenarios() -> list[EvaluationScenario]:
    """Diagnostics supplement the frozen golden manifest without invented truth."""
    path = ROOT / "data/evaluation/e2e_v1/diagnostics.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["benchmark_version"] != "e2e_claims_v1":
        raise ValueError("Unsupported diagnostic manifest version")
    return [EvaluationScenario.model_validate(item) for item in manifest["diagnostics"]]


def diagnostic_context(declared_damage):
    return {
        "declaration_is_synthetic": True,
        "quoted_damage_is_synthetic": True,
        "declared_damage": list(declared_damage),
        "ground_truth_available": False,
        "detection_truth_labels_loaded": False,
        "meaning": "Conditional integration diagnostic; does not establish facts about an insurance claim",
        "existing_visualizations": "artifacts/environment_merge/gpu-smoke/run_20261015T135812_022547Z/summary.json",
    }


class DiagnosticSteps:
    """Run the optional external-image diagnostic with explicit provenance."""

    def diagnostic_vision(self, run):
        """Infer on real external images without claiming claim-level ground truth."""
        observed = run.observed
        for component in DIAGNOSTIC_COMPONENTS:
            self._skip(
                observed,
                component,
                "Standalone local external-image diagnostic; no document/claim/LLM ground truth",
            )
        if not self.config.include_vision:
            self._skip(observed, "vision", "Disabled by configuration")
            self._skip(observed, "investigation", "Vision disabled")
            return
        if not self._available(observed, "vision", ["damage_vision", "part_vision"]):
            self._skip(observed, "investigation", "Vision dependencies unavailable")
            return
        run.vision = self._record(
            observed,
            "vision",
            lambda: self.infer_diagnostic(run),
            inputs={
                "synthetic_declared_damage": list(self.config.synthetic_declared_damage),
                "ground_truth_available": False,
            },
            sources=[str(self.config.source_summary), str(self.config.training_summary)],
        )
        self._capture_vision(observed, run.vision)
        self.investigate(run)

    def infer_diagnostic(self, run):
        """Assess the selected real image sources with the production fusion service."""
        from claims.enums import DamageType
        from scripts.check_multimodal_integration_smoke import diagnostic_case, select_sources
        from vision.fusion_config import DEFAULT_FUSION_POLICY
        from vision.models import ClaimMultimodalVisionAssessment
        from vision.multimodal_service import ClaimMultimodalVisionService, assess_multimodal_consistency

        observed = run.observed
        dataset_root, sources = select_sources(
            self.config.source_summary,
            self.config.training_summary,
            max_images=self.config.diagnostic_max_images,
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
        observed.provenance["vision_context"] = diagnostic_context(self.config.synthetic_declared_damage)
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
