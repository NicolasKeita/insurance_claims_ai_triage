"""Model scoring, vision observation and investigation stages."""

from hashlib import sha256

from evaluation.fakes import FakeAnomalyPredictor, FakeTriagePredictor
from evaluation.models import ComponentResult, TraceEvent


def score_models(run):
    """Score the production features with the configured or fixture predictors."""
    run.triage = run.recorder.record(
        "triage", lambda: (run.triage_predictor or FakeTriagePredictor()).predict(run.features)
    )
    run.anomaly = run.recorder.record(
        "anomaly",
        lambda: (
            run.anomaly_predictor or FakeAnomalyPredictor(run.scenario.inputs.get("anomaly_fixture"))
        ).assess(run.features),
    )
    run.output.facts["triage"] = run.triage.model_dump(mode="json")
    run.output.facts["anomaly"] = {
        **run.anomaly.assessment.model_dump(mode="json"),
        **run.anomaly.model_dump(mode="json", exclude={"assessment"}),
    }
    run.output.model_versions.update(triage=run.triage.model_version, anomaly=run.anomaly.model_version)


def _skip_vision(run, reason):
    run.output.components["vision"] = ComponentResult(status="SKIP", reason=reason, metadata={"applicable": False})
    run.output.trace.append(TraceEvent(component="vision", status="SKIP", outputs={"reason": reason}))


def _image_provenance(run):
    return [
        {
            "filename": filename,
            "synthetic": True,
            "asset_sha256": sha256(run.case.image_path(filename).read_bytes()).hexdigest(),
            "description": "Generated geometry-only pixels; "
            "detector observations supplied explicitly by the scenario.",
        }
        for filename in run.case.claim.images
    ]


def assess_vision(run):
    """Record explicit vision observations, or an explicit non-applicable skip."""
    if not run.scenario.vision_required:
        _skip_vision(run, "Document-centric scenario; no vision fixture requested.")
        return
    from claims.consistency import map_quote_damage_types

    from evaluation.deterministic.vision import fake_vision_service

    output = run.output
    run.visual = run.recorder.record(
        "vision",
        lambda: fake_vision_service(run.case, run.scenario.inputs.get("vision_fixture", {})).assess_claim(
            run.case, quoted_damage_types=map_quote_damage_types(run.quote)[0]
        ),
    )
    output.vision = {
        **run.visual.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in run.visual.observations],
    }
    output.facts["vision"] = output.vision
    output.model_versions.update(
        damage=run.visual.damage_model_metadata.version or "unknown",
        parts=run.visual.part_model_metadata.version or "unknown",
    )
    output.policy_versions["fusion"] = run.visual.fusion_policy.version
    output.provenance["images"] = _image_provenance(run)


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