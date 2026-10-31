"""Copy registered source files; never fetch, regenerate or follow symlinks."""

import hashlib
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def plain_name(value, label="filename"):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(character in value for character in '/\\:\x00<>"|?*')
        or any(ord(character) < 32 for character in value)
        or value.endswith((" ", "."))
        or value.split(".")[0].casefold() in RESERVED
    ):
        raise ValueError(f"Unsafe {label}; use a plain filename or identifier")
    return value


def copied_files(review, root, document_paths, image_paths):
    documents = {item.filename for item in review.claim.documents if item.filename}
    images = set(review.claim.images)
    for paths, registered in ((document_paths, documents), (image_paths, images)):
        if set(paths) - registered:
            raise ValueError("An export source file is not registered with this claim")
        for filename in paths:
            plain_name(filename)
    records = []
    seen = {}
    for document in review.claim.documents:
        if document.filename:
            plain_name(document.filename)
        key = (document.filename, document.available)
        if document.filename and key in seen:
            record = dict(seen[key])
        else:
            record = _copy(
                root, "document", document.filename, document_paths.get(document.filename), provided=document.available
            )
            seen[key] = record
        record["document_type"] = document.type.value
        records.append(record)
    for filename in review.claim.images:
        plain_name(filename)
        record = _copy(root, "image", filename, image_paths.get(filename))
        _check_recorded_hash(review, record)
        records.append(record)
    return records


def _check_recorded_hash(review, record):
    images = review.vision.data.get("images", []) if isinstance(review.vision.data, dict) else []
    for image in images:
        if image.get("filename") != record["filename"]:
            continue
        recorded = (image.get("provenance") or {}).get("sha256")
        if recorded and record["status"] == "COPIED" and record["sha256"] != recorded:
            raise ValueError("Source image checksum differs from its recorded analysis provenance")


def _copy(root, kind, filename, source, *, provided=True):
    record = {
        "kind": kind,
        "filename": filename,
        "path": None,
        "status": "MISSING",
        "sha256": None,
        "size_bytes": None,
    }
    if not provided or not filename:
        return {**record, "status": "NOT_PROVIDED", "reason": "No file was supplied with this claim."}
    suffixes = {".pdf"} if kind == "document" else IMAGE_SUFFIXES
    if Path(filename).suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported {kind} file extension")
    if source is None:
        return {**record, "reason": "The original file was not supplied for export."}
    source = Path(source).absolute()
    if any(part.is_symlink() for part in (source, *source.parents)):
        raise ValueError("Export source symlinks are not allowed")
    if not source.is_file():
        return {**record, "reason": "The registered source file could not be found."}
    if source.suffix.lower() not in suffixes:
        raise ValueError("Export source file extension does not match its registered kind")
    relative = f"{'documents' if kind == 'document' else 'images'}/{filename}"
    target = root / relative
    target.parent.mkdir(exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as reader, target.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            writer.write(block)
            digest.update(block)
            count += len(block)
    return {**record, "path": relative, "status": "COPIED", "sha256": digest.hexdigest(), "size_bytes": count}
