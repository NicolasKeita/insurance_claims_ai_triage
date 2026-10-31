"""Provenance of the frozen component results, not a fabricated pipeline run."""

from datetime import datetime, timezone

import tomllib

from api.version import application_version
from claims.review.archive.storage.catalogue import ARCHIVE_FORMAT, PROJECT_ROOT
from claims.review.assembly import SECTION_NAMES


def archive_manifest(review, options, files, fingerprint):
    return {
        "format": ARCHIVE_FORMAT,
        "application_version": application_version(),
        "claim_id": review.claim.claim_id,
        "analysis_id": options.analysis_id,
        "title": options.title
        or (f"{review.claim.vehicle.make} {review.claim.vehicle.model} · {review.claim.incident.location}"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "captured_at": review.generated_at.isoformat(),
        "source": options.source,
        "scope": "Frozen stored review; component results may have different analysis dates.",
        "review_file": "resultats.json",
        "report_file": "rapport.html",
        "fingerprint": fingerprint,
        "components": {name: _component(getattr(review, name)) for name in SECTION_NAMES},
        "model_versions": review.audit.model_versions,
        "policy_versions": review.audit.policy_versions,
        "files": files,
    }


def _component(section):
    return {
        "status": section.status,
        "source": section.source,
        "updated_at": section.updated_at.isoformat() if section.updated_at else None,
    }


def translations():
    path = PROJECT_ROOT / "frontend/locales/fr.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["messages"] if path.is_file() else {}
