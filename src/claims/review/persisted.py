"""Read existing repositories without initializing inference or retrieval clients."""

from sqlalchemy.exc import SQLAlchemyError

from claims.review.assembly import build_review
from claims.review.media import ClaimImageStore
from claims.review.models import ReviewSection
from claims.review.stored_context import context_from_agent
from persistence.agent_runs import AgentRunRepository
from persistence.history import HistoricalClaimRepository
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository
from persistence.vision_assessments import VisionAssessmentRepository


class PersistedReviewService:
    """Independent read sessions preserve partial results if a component fails."""

    def __init__(self, factory, *, asset_root=None):
        self.factory = factory
        self.images = ClaimImageStore(asset_root)

    def get_review(self, claim_id):
        claim = self._claim(claim_id)
        sections = {
            kind: self._read(
                kind, lambda session, kind=kind: getattr(AssessmentRepository(session), f"get_latest_{kind}")(claim_id)
            )
            for kind in ("triage", "anomaly", "investigation")
        }
        sections["history"] = self._read(
            "history", lambda session: HistoricalClaimRepository(session).get_profile(claim_id), record=False
        )
        if sections["history"].status == "AVAILABLE":
            sections["history"].source = f"history:{claim_id}"
        sections["vision"] = self._read(
            "vision", lambda session: VisionAssessmentRepository(session).latest_for_claim(claim_id)
        )
        sections["agent"] = self._read(
            "agent", lambda session: AgentRunRepository(session).latest_for_claim(claim_id), record=False
        )
        self._visual_sections(claim, sections)
        sections.update(context_from_agent(claim, sections["agent"]))
        return build_review(claim, sections)

    def _claim(self, claim_id):
        with self.factory() as session:
            claim = ClaimRepository(session).get_by_claim_id(claim_id)
        if claim is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return claim

    def _read(self, kind, reader, *, record=True):
        try:
            with self.factory() as session:
                result = reader(session)
            if result is None:
                return ReviewSection(status="NOT_EXECUTED", reason=f"No stored {kind} result is available.")
            data = result.result.model_dump(mode="json") if record else result.model_dump(mode="json")
            identifier = str(result.id) if record else data.get("run_id")
            updated_at = result.created_at if hasattr(result, "created_at") else None
            status = "FAILED" if kind == "agent" and data.get("status") == "FAILED" else "AVAILABLE"
            if kind == "agent":
                data.pop("trace", None)
                if data.get("error"):
                    data["error"] = "The stored agent run failed."
            return ReviewSection(
                status=status,
                data=data,
                source=f"{kind}:{identifier}" if identifier else None,
                updated_at=updated_at,
                reason="The stored agent run failed; no recommendation is available." if status == "FAILED" else None,
            )
        except SQLAlchemyError:
            return ReviewSection(status="UNAVAILABLE", reason=f"Stored {kind} data is temporarily unavailable.")
        except (ValueError, TypeError, KeyError):
            return ReviewSection(status="FAILED", reason=f"The stored {kind} snapshot could not be read safely.")

    def _visual_sections(self, claim, sections):
        section = sections["vision"]
        if section.status == "AVAILABLE":
            consistency = section.data.get("consistency")
            sections["visual_consistency"] = ReviewSection(
                status="AVAILABLE", data=consistency, source=section.source, updated_at=section.updated_at
            )
            images = section.data.get("images", [])
            if not any(image.get("evidence") for image in images):
                section.status = "NO_EVIDENCE"
                section.reason = (
                    "The stored assessment contains no supported visual observations; "
                    "non-detection does not establish absence."
                )
        else:
            sections["visual_consistency"] = ReviewSection(status=section.status, reason=section.reason)
            images = [{"filename": filename, "evidence": []} for filename in claim.images]
            section.data = {"images": images}
        self.images.annotate(claim, images)

    def image_path(self, claim_id, filename):
        return self.images.path(self._claim(claim_id), filename)

    @staticmethod
    def list_demo_claims():
        return []
