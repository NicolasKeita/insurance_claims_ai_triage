"""Real application adapters behind the opt-in evaluation pipeline.

Unavailable components remain explicit. Independent document parsing, model
inference and local multimodal diagnostics do not require a running database.
"""

from evaluation.models import EvaluationScenario, PipelineObservation
from evaluation.preflight import PreflightReport, RealEvaluationConfig, run_preflight
from evaluation.real.context import ComponentRecorder, RealRun
from evaluation.real.similar_claims import SimilarClaimsSteps
from evaluation.real.stages.agent import AgentSteps
from evaluation.real.stages.claim import ClaimSteps
from evaluation.real.stages.features import FeatureSteps
from evaluation.real.stages.investigation import InvestigationSteps
from evaluation.real.stages.retrieval import RetrievalSteps
from evaluation.real.stages.scoring import ScoringSteps
from evaluation.real.stages.vision import VisionSteps


class RealClaimsEvaluationPipeline(
    ComponentRecorder,
    ClaimSteps,
    FeatureSteps,
    ScoringSteps,
    VisionSteps,
    InvestigationSteps,
    RetrievalSteps,
    SimilarClaimsSteps,
    AgentSteps,
):
    """Evaluate golden scenarios against the real application services."""

    def __init__(self, config: RealEvaluationConfig, preflight: PreflightReport | None = None):
        self.config = config
        self.preflight = preflight if preflight is not None else run_preflight(config)
        self.resources = self.preflight.resources

    def close(self):
        self.preflight.close()

    def _skip_prerequisites(self, run):
        missing = self.preflight.blocked_scenarios.get(run.scenario.scenario_id)
        if missing:
            self._skip(run.observed, "prerequisites", "; ".join(missing), dependency=True)
        return bool(missing)

    def run(self, scenario: EvaluationScenario) -> PipelineObservation:
        """
        The shared helper uses real PDF ingestion and production extraction functions.

        Evaluation fixtures are never silently imported into the application DB. History for
        controlled claims requires an explicit prepared/seeded source.

        Real stored absence represents the unavailable-tool case; the read-only application tools see
        genuine missing assessment rows.
        """
        run = RealRun(scenario=scenario)
        observed = run.observed
        observed.provenance.update(mode="real", scenario_source=scenario.claim_source)
        if self._skip_prerequisites(run):
            return observed
        if scenario.inputs.get("external_vision_diagnostic"):
            self.diagnostic_vision(run)
            self._finish(observed, run.started)
            return observed
        run.use_reference = scenario.inputs.get("use_reference_case", False)
        self.load_case(run)
        if run.case is None:
            self.skip_claim_components(run)
            self._finish(observed, run.started)
            return observed
        self.extract_documents(run)
        self.assess_ml_models(run)
        self.assess_history(run)
        self.assess_vision(run)
        self.investigate(run)
        self.persist_demo_assessments(run)
        self.retrieve(run)
        self.similar(run)
        self.assess_agent(run)
        self._finish(observed, run.started)
        return observed

    def persist_demo_assessments(self, run):
        """Persist actually computed assessments into the isolated demo schema."""
        if not self.config.isolated_demo_stack:
            return
        from claims.investigation.models import InvestigationAssessment
        from evaluation.isolated_stack import DemoAssessments, seed_case
        from ml.anomaly_inference import RegisteredAnomalyAssessment

        observed = run.observed
        registered_anomaly = run.registered_anomaly or (
            RegisteredAnomalyAssessment(
                assessment=run.anomaly,
                registered_model="local:anomaly_detector.joblib",
                model_version=observed.model_versions.get("anomaly", "unknown"),
                model_alias="local-artifact",
            )
            if run.anomaly is not None
            else None
        )
        assessment = InvestigationAssessment.model_validate(observed.investigation) if observed.investigation else None
        unavailable = set(run.scenario.inputs.get("unavailable_tools", []))
        self._record(
            observed,
            "assessment_persistence",
            lambda: seed_case(
                self.resources["session_factory"],
                run.case,
                DemoAssessments(
                    triage=None if "GET_TRIAGE" in unavailable else run.triage,
                    anomaly=None if "GET_ANOMALY" in unavailable else registered_anomaly,
                    investigation=None if "GET_INVESTIGATION" in unavailable else assessment,
                    vision=None if "GET_VISUAL_ASSESSMENT" in unavailable else run.vision,
                ),
            ),
            inputs={"claim_id": run.case.claim.claim_id, "unavailable_assessments": sorted(unavailable)},
        )
