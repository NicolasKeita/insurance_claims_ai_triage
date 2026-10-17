"""Claim loading, document extraction and feature assessment stages."""

from pathlib import Path
from tempfile import mkdtemp

from evaluation.preflight import ROOT
from evaluation.real.context import payload

CLAIM_COMPONENTS = (
    "document",
    "consistency",
    "triage",
    "anomaly",
    "history",
    "investigation",
    "vision",
    "retrieval",
    "agent",
)


class ClaimSteps:
    """Load the controlled claim, parse its documents and score the features."""

    def skip_claim_components(self, run):
        for name in CLAIM_COMPONENTS:
            self._skip(run.observed, name, "Claim loading failed")

    def load_case(self, run):
        """Load the reference case or materialize a controlled synthetic one."""
        observed = run.observed
        if run.use_reference:
            from claims.loader import load_claim_case

            source = ROOT / run.scenario.claim_source
            run.case = self._record(
                observed, "claim", lambda: load_claim_case(source), sources=[str(source / "claim.json")]
            )
        else:
            from evaluation.pipeline import materialize_scenario_case

            self.config.fixture_dir.mkdir(parents=True, exist_ok=True)
            fixture_root = Path(mkdtemp(prefix=run.scenario.scenario_id + "-", dir=self.config.fixture_dir))
            run.case = self._record(
                observed,
                "claim",
                lambda: materialize_scenario_case(run.scenario, fixture_root),
                inputs={"controlled_synthetic_inputs": run.scenario.inputs},
                sources=[run.scenario.claim_source],
            )
        if run.case is None:
            return
        observed.facts["claim"] = run.case.claim.model_dump(mode="json")
        observed.provenance["controlled_synthetic_claim"] = not run.use_reference
        observed.provenance["claim_id"] = run.case.claim.claim_id
        self.seed_demo_claim(run)

    def seed_demo_claim(self, run):
        """Seed the controlled claim, and its history, into the isolated schema."""
        if not self.config.isolated_demo_stack:
            return
        from evaluation.isolated_stack import seed_case, seed_history

        observed = run.observed
        seeded = self._record(
            observed,
            "demo_seed",
            lambda: seed_case(self.resources["session_factory"], run.case),
            inputs={"claim_id": run.case.claim.claim_id, "schema": self.resources["isolated_schema"]},
        )
        if not (run.scenario.inputs.get("history_claims") and seeded is not None):
            return
        self._record(
            observed,
            "history_seed",
            lambda: seed_history(
                self.resources["session_factory"], run.case.claim, run.scenario.inputs["history_claims"]
            ),
            inputs={
                "claim_id": run.case.claim.claim_id,
                "controlled_previous_claims": run.scenario.inputs["history_claims"],
            },
        )

    def extract_documents(self, run):
        """Ingest real PDFs and record the structured extraction facts."""
        from claims.enums import DocumentType
        from evaluation.pipeline import extract_case_documents

        observed, case = run.observed, run.case
        docs = self._record(
            observed,
            "document",
            lambda: extract_case_documents(case),
            inputs={"claim_id": case.claim.claim_id},
            sources=[str(case.document_path(entry.filename)) for entry in case.claim.documents if entry.available],
        )
        run.extractions = docs[0] if docs is not None else {}
        run.form, run.report, run.quote = (
            run.extractions.get(kind)
            for kind in (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)
        )
        observed.facts["extraction"] = {
            "success": bool(run.extractions),
            "document_count": len(run.extractions),
            "form": payload(run.form),
            "report": payload(run.report),
            "quote": payload(run.quote),
        }
        observed.provenance["documents"] = [item.source.model_dump(mode="json") for item in run.extractions.values()]
        for source in observed.provenance["documents"]:
            source["asset_reference"] = str(case.document_path(source["filename"]))
        self.compare_documents(run)
        self.build_features(run)
