"""Document comparison and production feature stages."""


class FeatureSteps:
    """Compare parsed documents and build the production triage features."""

    def compare_documents(self, run):
        """Compare documents when all three were parsed; otherwise skip explicitly."""
        from claims.consistency import compare_claim_documents

        observed = run.observed
        if run.form is None or run.report is None or run.quote is None:
            self._skip(observed, "consistency", "At least one comparison document is unavailable; no matches invented")
            return
        run.consistency = self._record(
            observed,
            "consistency",
            lambda: compare_claim_documents(run.form, run.report, run.quote),
            sources=[source["asset_reference"] for source in observed.provenance["documents"]],
        )
        if run.consistency is not None:
            observed.facts["consistency"] = {
                **run.consistency.model_dump(mode="json"),
                "is_consistent": run.consistency.is_consistent,
            }

    def build_features(self, run):
        """Build production features, or record the completeness-only fallback."""
        from ml.features import build_triage_features, calculate_document_completeness

        observed = run.observed
        count, missing, ratio = calculate_document_completeness(run.case)
        if run.consistency is not None:
            run.features = self._record(
                observed,
                "features",
                lambda: build_triage_features(run.case, run.form, run.quote, run.consistency),
            )
        else:
            self._skip(observed, "features", "Document comparison is unavailable")
        observed.facts["features"] = (
            run.features.model_dump(mode="json")
            if run.features is not None
            else {
                "document_count": count,
                "missing_document_count": missing,
                "document_completeness_ratio": ratio,
            }
        )
        observed.facts["extraction"].update(missing_document_count=missing, document_completeness_ratio=ratio)
