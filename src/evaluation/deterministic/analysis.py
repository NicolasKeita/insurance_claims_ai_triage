"""Model scoring, investigation and agent review stages."""

from evaluation.fakes import FakeAnomalyPredictor, FakeTriagePredictor


def score_models(run):
    """Score the production features with the configured or fixture predictors."""
    run.triage = run.recorder.record(
        "triage", lambda: (run.triage_predictor or FakeTriagePredictor()).predict(run.features)
    )
    run.anomaly = run.recorder.record(
        "anomaly",
        lambda: (run.anomaly_predictor or FakeAnomalyPredictor(run.scenario.inputs.get("anomaly_fixture"))).assess(
            run.features
        ),
    )
    run.output.facts["triage"] = run.triage.model_dump(mode="json")
    run.output.facts["anomaly"] = {
        **run.anomaly.assessment.model_dump(mode="json"),
        **run.anomaly.model_dump(mode="json", exclude={"assessment"}),
    }
    run.output.model_versions.update(triage=run.triage.model_version, anomaly=run.anomaly.model_version)


def investigate(run):
    """Assess investigation signals with the production engine and policy."""
    from investigation.engine import assess_investigation
    from investigation.policy import INVESTIGATION_POLICY_V2, INVESTIGATION_POLICY_V3

    policy = INVESTIGATION_POLICY_V3 if run.visual is not None else INVESTIGATION_POLICY_V2
    investigation = run.recorder.record(
        "investigation",
        lambda: assess_investigation(
            consistency=run.consistency,
            anomaly=run.anomaly.assessment,
            features=run.features,
            history=run.history,
            policy=policy,
            visual_assessment=run.visual,
        ),
    )
    run.output.investigation = investigation.model_dump(mode="json")
    run.output.facts["investigation"] = run.output.investigation
    run.output.signals = [signal.code.value for signal in investigation.signals]
    run.output.policy_versions["investigation"] = investigation.policy_version


def review_agent(run):
    """Run the production agent service against controlled snapshot tools."""
    from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
    from agent.service import AgentServiceConfig, ClaimsAgentService
    from evaluation.fakes import SnapshotAgentTools

    output = run.output
    snapshots = {
        "triage": output.facts["triage"],
        "anomaly": run.anomaly.model_dump(mode="json"),
        "history": output.facts["history"],
        "investigation": output.investigation,
        "vision": output.vision,
    }
    tools = SnapshotAgentTools(
        run.case.claim,
        snapshots,
        similar_service=run.similar,
        knowledge_retriever=run.retriever,
        unavailable=run.scenario.inputs.get("unavailable_tools", []),
    )
    agent = run.recorder.record(
        "agent",
        lambda: ClaimsAgentService(
            AgentServiceConfig(
                tools,
                StructuredAgentPlanner(run.llm),
                StructuredAgentFinalizer(run.llm),
                max_iterations=8,
                llm_model="fake-structured-e2e-v1",
            )
        ).review_claim(run.case.claim.claim_id),
    )
    output.agent = agent.model_dump(mode="json")
    output.facts["agent"] = output.agent
    output.model_versions.update(embeddings=run.embeddings.model_name, llm="fake-structured-e2e-v1")
