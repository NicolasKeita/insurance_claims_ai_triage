"""Thin benchmark orchestration over existing application/domain services.

Only external boundaries are synthetic in ``ClaimsEvaluationPipeline``. Fixture
materialization is shared with the real backend and never rewrites source assets.
"""

from contextlib import closing
from tempfile import TemporaryDirectory

from evaluation.deterministic import (
    ROOT,
    SYNTHETIC_PROVENANCE,
    DeterministicRun,
    TraceRecorder,
    assess_history,
    assess_vision,
    build_features,
    compare_documents,
    extract_case_documents,
    extract_documents,
    fake_vision_service,
    investigate,
    load_claim,
    materialize_scenario_case,
    retrieve_knowledge,
    retrieve_similar,
    review_agent,
    score_models,
)
from evaluation.models import PipelineObservation

__all__ = [
    "ROOT",
    "SYNTHETIC_PROVENANCE",
    "ClaimsEvaluationPipeline",
    "extract_case_documents",
    "fake_vision_service",
    "materialize_scenario_case",
]


def execute(run):
    """Run every deterministic stage and retain partial observations on failure."""
    from qdrant_client import QdrantClient

    with (
        TemporaryDirectory(prefix="claims-e2e-", dir=ROOT) as workspace,
        closing(QdrantClient(":memory:")) as client,
    ):
        try:
            load_claim(run, workspace)
            extract_documents(run)
            compare_documents(run)
            build_features(run)
            score_models(run)
            assess_history(run)
            assess_vision(run)
            investigate(run)
            retrieve_similar(run, client)
            retrieve_knowledge(run, client)
            review_agent(run)
        except Exception as error:
            if not run.output.errors:
                run.output.errors.append(f"orchestration: {type(error).__name__}: {error}")
    return run.output


class ClaimsEvaluationPipeline:
    """Deterministic E2E adapter. Real evaluation uses RealClaimsEvaluationPipeline."""

    def __init__(self, *, triage_predictor=None, anomaly_predictor=None):
        self.triage_predictor, self.anomaly_predictor = triage_predictor, anomaly_predictor

    def run(self, scenario):
        """
        Use the real Qdrant stores/index/query orchestration in local memory.

        Preserve partial observations for useful expectation diagnostics.
        """
        output = PipelineObservation()
        return execute(
            DeterministicRun(
                scenario=scenario,
                output=output,
                recorder=TraceRecorder(output),
                triage_predictor=self.triage_predictor,
                anomaly_predictor=self.anomaly_predictor,
            )
        )
