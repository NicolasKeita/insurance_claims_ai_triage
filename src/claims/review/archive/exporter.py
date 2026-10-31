"""Freeze an existing review and copy its sources into a portable directory."""

import hashlib
import json
import tempfile
from pathlib import Path

from claims.review.archive.assets import copied_files, plain_name
from claims.review.archive.report import render_index, render_report
from claims.review.archive.storage.catalogue import COLLECTION_FORMAT, atomic_write, collection, write_json
from claims.review.archive.storage.metadata import archive_manifest, translations
from claims.review.archive.storage.options import ExportOptions
from claims.review.archive.storage.snapshot import local_review


def export_review(review, **options):
    """Export only supplied, already computed data; refuse to overwrite an analysis.

    The ID identifies this frozen set of component results and copied source bytes.
    It does not assert that independently dated results came from one pipeline run.
    """
    config = ExportOptions(**options)
    plain_name(review.claim.claim_id, "claim identifier")
    if config.analysis_id is not None:
        plain_name(config.analysis_id, "analysis identifier")
    root = Path(config.output_root).absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("Export directory symlinks are not allowed")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".export.lock"
    try:
        lock.open("x", encoding="utf-8").close()
    except FileExistsError as error:
        raise FileExistsError("Another export is using this analysis_results directory") from error
    try:
        return _export(root, review, config)
    finally:
        lock.unlink()


def _export(root, review, options):
    entries = collection(root)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=root) as temporary:
        staging = Path(temporary)
        files = copied_files(review, staging, options.document_paths or {}, options.image_paths or {})
        frozen = local_review(review, files)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"review": frozen.model_dump(mode="json"), "files": files},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        identifier = options.analysis_id or f"ANALYSIS-{fingerprint[:12]}"
        options.analysis_id = identifier
        folder = f"{review.claim.claim_id}_{identifier}"
        target = root / folder
        if target.exists():
            raise FileExistsError(f"Analysis export already exists: {target.name}; choose a new analysis identifier")
        manifest = archive_manifest(frozen, options, files, fingerprint)
        write_json(staging / "resultats.json", frozen.model_dump(mode="json"))
        manifest["review_sha256"] = hashlib.sha256((staging / "resultats.json").read_bytes()).hexdigest()
        write_json(staging / "manifest.json", manifest)
        (staging / "rapport.html").write_text(
            render_report(frozen, manifest, translations=translations()), encoding="utf-8"
        )
        staging.rename(target)
    entry = {
        "claim_id": review.claim.claim_id,
        "analysis_id": identifier,
        "title": manifest["title"],
        "path": f"{folder}/rapport.html",
        "manifest": f"{folder}/manifest.json",
        "exported_at": manifest["exported_at"],
        "captured_at": manifest["captured_at"],
    }
    entries.append(entry)
    atomic_write(root / "manifest.json", json.dumps({"format": COLLECTION_FORMAT, "analyses": entries}, indent=2))
    atomic_write(root / "index.html", render_index(entries))
    return target
