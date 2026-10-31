"""Independent recorded review with archive-relative file references."""


def local_review(review, files):
    data = review.model_dump(mode="json")
    images = {item["filename"]: item for item in files if item["kind"] == "image"}
    documents = {item["filename"]: item for item in files if item["kind"] == "document"}
    data = _rewrite_images(data, images)
    section = data["documents"].get("data")
    if isinstance(section, dict):
        for document in section.get("items", []):
            item = documents.get(document.get("filename"), {})
            document["document_url"] = item.get("path")
            document["document_unavailable_reason"] = (
                None if item.get("path") else item.get("reason", "The original PDF is not included in this archive.")
            )
    return review.__class__.model_validate(data)


def _rewrite_images(value, images):
    """Localize images and discard server document URLs before adding copied PDFs."""
    if isinstance(value, list):
        return [_rewrite_images(item, images) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _rewrite_images(item, images) for key, item in value.items()}
    result.pop("document_url", None)
    if "image_url" in result:
        item = images.get(result.get("filename"), {})
        result["image_url"] = item.get("path")
        result["image_unavailable_reason"] = (
            None if item.get("path") else item.get("reason", "The original photograph is not included in this archive.")
        )
    return result
