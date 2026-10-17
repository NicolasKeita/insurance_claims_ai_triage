"""Registered ML predictor and stored history stages."""


class ScoringSteps:
    """Score production features and load stored history for controlled claims."""

    def assess_ml_models(self, run):
        """Score production features with the retained registered predictors."""
        observed = run.observed
        for kind in ("triage", "anomaly"):
            if run.features is None:
                self._skip(observed, kind, "Production feature inputs are unavailable")
                continue
            if not self._available(observed, kind, [kind]):
                continue
            self._version(observed, kind)
            if kind == "triage":
                self.assess_triage(run)
            else:
                self.assess_anomaly(run)

    def assess_triage(self, run):
        observed = run.observed
        predictor = self.resources["triage"]
        run.triage = self._record(
            observed,
            "triage",
            lambda: predictor.predict(run.features),
            inputs=run.features.model_dump(mode="json"),
            sources=[str(observed.provenance["triage"].get("checkpoint", "mlflow_candidate"))],
        )
        if run.triage is not None:
            observed.facts["triage"] = run.triage.model_dump(mode="json")

    def assess_anomaly(self, run):
        observed = run.observed
        predictor = self.resources["anomaly"]
        result = self._record(
            observed,
            "anomaly",
            lambda: predictor.assess(run.features),
            inputs=run.features.model_dump(mode="json"),
            sources=[str(observed.provenance["anomaly"].get("checkpoint", "mlflow_candidate"))],
        )
        if result is None:
            return
        if hasattr(result, "assessment"):
            run.registered_anomaly = result
        run.anomaly = result.assessment if hasattr(result, "assessment") else result
        observed.facts["anomaly"] = run.anomaly.model_dump(mode="json")
        observed.facts["anomaly"]["interpretation"] = "Statistical atypicality; not a fraud determination"

    def assess_history(self, run):
        """Load stored history only for a matching, seeded controlled claim."""
        observed = run.observed
        if not self._available(observed, "history", ["postgresql"]):
            return
        from persistence.repositories import ClaimRepository

        factory = self.resources["session_factory"]
        try:
            with factory() as session:
                stored = ClaimRepository(session).get_by_claim_id(run.case.claim.claim_id)
            if stored is None:
                self._skip(observed, "history", "Controlled claim is not seeded in the configured PostgreSQL database")
            elif not run.use_reference and stored.model_dump(mode="json") != run.case.claim.model_dump(mode="json"):
                self._skip(
                    observed, "history", "Stored claim differs from controlled fixture; no unrelated history reused"
                )
            else:
                run.history = self._record(
                    observed,
                    "history",
                    lambda: self.history_profile(factory, run.case.claim.claim_id),
                    sources=["postgresql:claims:" + run.case.claim.claim_id],
                )
        except Exception as error:
            self._skip(
                observed,
                "history",
                f"Configured PostgreSQL schema unavailable: {type(error).__name__}: {error}",
                dependency=True,
            )
        if run.history is not None:
            observed.facts["history"] = run.history.model_dump(mode="json")

    @staticmethod
    def history_profile(factory, claim_id):
        from persistence.history import HistoricalClaimRepository

        with factory() as session:
            return HistoricalClaimRepository(session).get_profile(claim_id)
