"""Offline reports stay portable after moving and deleting their original sources."""

import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

from claims.review.archive import export_review
from tests.claims.review_archive._support import recorded_review, source_assets


def test_copied_assets_and_report_references_survive_moving_export_and_removing_originals(tmp_path):
    from html.parser import HTMLParser

    class References(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []

        def handle_starttag(self, tag, attrs):
            self.values.extend(value for key, value in attrs if key in {"href", "src"} and value)

    documents, images = source_assets(tmp_path / "originals")
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="PORTABLE",
        document_paths=documents,
        image_paths=images,
    )
    moved = tmp_path / "on-another-machine" / bundle.name
    moved.parent.mkdir()
    shutil.copytree(bundle, moved)
    shutil.rmtree(tmp_path / "originals")
    shutil.rmtree(bundle)
    html = (moved / "rapport.html").read_text(encoding="utf-8")
    parser = References()
    parser.feed(html)
    file_references = []
    for value in parser.values:
        reference = urlsplit(value)
        assert reference.scheme not in {"http", "https", "file"}
        if not reference.path or reference.scheme in {"data", "mailto"}:
            continue
        relative = Path(unquote(reference.path))
        assert not relative.is_absolute()
        assert (moved / relative).is_file()
        file_references.append(relative)
    assert any(reference.suffix == ".pdf" for reference in file_references)
    assert any(reference.suffix == ".png" for reference in file_references)
