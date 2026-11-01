"""Static export refuses stale checksums, real data and unsafe replacements."""

from pathlib import Path

import pytest

from evaluation.reporting.static_demo import export_static_demo
from tests.api.review.demo._static_demo import BODY, make_archive, read_json, tree_bytes


def test_corrupted_source_aborts_without_replacing_an_existing_dataset(tmp_path):
    bundle = make_archive(tmp_path)
    output = export_static_demo(tmp_path / "archives", tmp_path / "demo")
    frozen = tree_bytes(output)
    document = next((bundle / "documents").glob("*.pdf"))
    document.write_bytes(document.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        export_static_demo(tmp_path / "archives", output, replace=True)
    assert tree_bytes(output) == frozen


@pytest.mark.parametrize("mode,body", [("persisted", BODY), ("demo", BODY.replace("DEMO SYNTHETIC ", ""))])
def test_only_saved_synthetic_demo_reviews_can_be_exported(tmp_path, mode, body):
    make_archive(tmp_path, mode=mode, body=body)
    with pytest.raises(ValueError, match="synthetic|demonstration"):
        export_static_demo(tmp_path / "archives", tmp_path / "demo")
    assert not (tmp_path / "demo").exists()


def test_unknown_output_directories_are_preserved_even_with_replace(tmp_path):
    make_archive(tmp_path)
    output = tmp_path / "demo"
    output.mkdir()
    marker = output / "precious.txt"
    marker.write_text("Keep this file", encoding="utf-8")
    with pytest.raises(ValueError, match="existing files were preserved"):
        export_static_demo(tmp_path / "archives", output, replace=True)
    assert marker.read_text(encoding="utf-8") == "Keep this file"


def test_refresh_requires_replace_and_keeps_previous_archives_unchanged(tmp_path):
    make_archive(tmp_path)
    source = tmp_path / "archives"
    output = export_static_demo(source, tmp_path / "demo")
    make_archive(tmp_path, analysis_id="SECOND")
    frozen = tree_bytes(source)
    with pytest.raises(FileExistsError, match="--replace"):
        export_static_demo(source, output)
    assert len(read_json(output / "manifest.json")["claims"]) == 1
    export_static_demo(source, output, replace=True)
    assert len(read_json(output / "manifest.json")["claims"]) == 2
    assert tree_bytes(source) == frozen


def test_replacement_is_restricted_to_the_named_demo_directory(tmp_path):
    make_archive(tmp_path)
    source = tmp_path / "archives"
    output = export_static_demo(source, tmp_path / "other")
    frozen = tree_bytes(output)
    make_archive(tmp_path, analysis_id="SECOND")
    with pytest.raises(ValueError, match="restricted"):
        export_static_demo(source, output, replace=True)
    assert tree_bytes(output) == frozen


@pytest.mark.parametrize("linked", ["source", "output"])
def test_symlink_paths_are_rejected_before_any_publication(tmp_path, monkeypatch, linked):
    make_archive(tmp_path)
    output = tmp_path / "demo"
    selected = output if linked == "output" else tmp_path / "archives"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == selected or original(path))
    with pytest.raises(ValueError, match="symlinks"):
        export_static_demo(tmp_path / "archives", output)
    assert not output.exists()
