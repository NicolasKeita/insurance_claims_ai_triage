"""Investigation signal stages built from observed inputs."""


class InvestigationSteps:
    """Derive investigation signals without inventing missing inputs."""

    def investigate(self, run):
        """Assess investigation signals from every actually observed input."""
        observed = run.observed
        from claims.investigation.policy import INVESTIGATION_POLICY_V2, INVESTIGATION_POLICY_V3

        policy = INVESTIGATION_POLICY_V3 if run.vision is not None else INVESTIGATION_POLICY_V2
        sources = {
            "consistency": run.consistency,
            "features": run.features,
            "anomaly": run.anomaly,
            "history": run.history,
            "vision": run.vision,
        }
        omitted = [key for key, value in sources.items() if value is None]
        if all(value is None for value in sources.values()):
            self._skip(observed, "investigation", "No observed signal inputs are available")
            return
        assessment = self._record(
            observed,
            "investigation",
            lambda: self.assess_signals(policy, sources),
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
        if run.scenario.inputs.get("history_claims") and run.history is None:
            from evaluation.models import EvaluationStatus

            observed.components["investigation"].status = EvaluationStatus.SKIP
            observed.components[
                "investigation"
            ].reason = "Required historical context unavailable; observed partial signal assessment retained"

    @staticmethod
    def assess_signals(policy, sources):
        """Reuse the application's independent translators and policy.

        No negative anomaly assessment or empty history is invented.
        """
        consistency, features = sources["consistency"], sources["features"]
        anomaly, history, vision = sources["anomaly"], sources["history"], sources["vision"]
        if all(value is not None for value in (consistency, features, anomaly, history)):
            from claims.investigation.engine import InvestigationContext, assess_investigation

            return assess_investigation(
                consistency=consistency,
                anomaly=anomaly,
                features=features,
                policy=policy,
                context=InvestigationContext(history=history, visual_assessment=vision),
            )
        from claims.investigation.models import InvestigationAssessment

        signals = InvestigationSteps._partial_signals(policy, sources)
        score = policy.score_signals(signals)
        return InvestigationAssessment(
            review_score=score,
            review_priority=policy.priority_for_score(score),
            review_recommended=policy.review_recommended_for_score(score),
            policy_version=policy.version,
            signals=signals,
        )

    @staticmethod
    def _partial_signals(policy, sources):
        """Translate whichever partial inputs were actually observed."""
        from claims.investigation.signals import (
            signals_from_anomaly,
            signals_from_completeness,
            signals_from_consistency,
            signals_from_history,
            signals_from_multimodal,
        )

        consistency, features = sources["consistency"], sources["features"]
        anomaly, history, vision = sources["anomaly"], sources["history"], sources["vision"]
        return (
            *(signals_from_consistency(consistency) if consistency is not None else ()),
            *(signals_from_anomaly(anomaly, policy) if anomaly is not None else ()),
            *(signals_from_completeness(features, policy) if features is not None else ()),
            *(signals_from_history(history, policy) if history is not None else ()),
            *(signals_from_multimodal(vision, policy) if vision is not None else ()),
        )
