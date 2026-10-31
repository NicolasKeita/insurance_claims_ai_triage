"""Copy already computed claim reviews and their original assets into analysis_results/."""

import argparse
import os
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from claims.review.archive import export_review
from claims.review.archive.sources import case_sources, prepared_sources, read_review, selected_cases
from claims.review.archive.storage.metadata import translations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--demo-dir", type=Path, help="Existing prepared reviews or portable analysis collection")
    sources.add_argument("--review-file", type=Path, help="Existing ClaimReview JSON; no analysis is performed")
    parser.add_argument("--claim-id", help="Select a prepared claim, or read its stored PostgreSQL results")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument(
        "--analysis-id", help="Optional export identity for a single claim; existing exports are preserved"
    )
    parser.add_argument("--title", help="Readable report title for a single claim")
    parser.add_argument("--asset-root", type=Path, help="Original files under <root>/<claim-id>/documents and images")
    parser.add_argument(
        "--case-dir", type=Path, action="append", default=[], help="Original case directory with claim.json"
    )
    options = parser.parse_args()
    if not any((options.demo_dir, options.review_file, options.claim_id)):
        parser.error("Select --demo-dir, --review-file, or --claim-id for stored database results")
    try:
        return _export_selected(options)
    except SQLAlchemyError:
        print("Cannot read saved database results. Check DATABASE_URL and the PostgreSQL service.")
        return 1
    except (OSError, ValueError, LookupError, TypeError) as error:
        print(f"Cannot export analysis results: {error}")
        return 1


def _export_selected(options):
    cases = selected_cases(options.case_dir)
    if options.demo_dir:
        from claims.review.demo.provider import DemoReviewService

        service = DemoReviewService(options.demo_dir)
        ids = [options.claim_id] if options.claim_id else list(service.entries)
        if len(ids) > 1 and (options.analysis_id or options.title):
            raise ValueError("--analysis-id and --title apply to a single --claim-id")
        for claim_id in ids:
            review = service.get_review(claim_id)
            documents, images = prepared_sources(service, review, cases, options.asset_root)
            settings = argparse.Namespace(**vars(options))
            entry_title = service.entries[claim_id]["title"]
            settings.title = options.title or (
                f"{translations().get(entry_title, entry_title)} · "
                f"{review.claim.vehicle.make} {review.claim.vehicle.model}"
            )
            _write(settings, review, documents, images, "prepared-review")
    elif options.review_file:
        review = read_review(options.review_file)
        if options.claim_id and options.claim_id != review.claim.claim_id:
            raise ValueError("Review file identity differs from --claim-id")
        documents, images = _sources(options, review, cases)
        _write(options, review, documents, images, "review-file")
    else:
        _database(options, cases)
    print(f"Open {(options.output_dir.resolve() / 'index.html')}")
    return 0


def _sources(options, review, cases):
    case = cases.get(review.claim.claim_id)
    asset_root = options.asset_root or os.environ.get("CLAIMS_REVIEW_ASSET_ROOT")
    if case is None and asset_root:
        case = Path(asset_root) / review.claim.claim_id
    return case_sources(review, case) if case is not None else ({}, {})


def _database(options, cases):
    from claims.review.persisted import PersistedReviewService
    from persistence.database import DatabaseConfig, create_database_engine, create_session_factory

    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        service = PersistedReviewService(create_session_factory(engine), asset_root=options.asset_root)
        review = service.get_review(options.claim_id)
        documents, images = _sources(options, review, cases)
        _write(options, review, documents, images, "persisted-latest")
    finally:
        engine.dispose()


def _write(options, review, documents, images, kind):
    path = export_review(
        review,
        output_root=options.output_dir,
        analysis_id=options.analysis_id,
        title=options.title,
        document_paths=documents,
        image_paths=images,
        source={"kind": kind, "reference": review.claim.claim_id},
    )
    print(f"Exported {review.claim.claim_id}: {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
