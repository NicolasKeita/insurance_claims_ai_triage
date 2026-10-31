"""Human-readable command errors and immutable exports from already computed files."""

import json

import pytest

from scripts.export_analysis_results import main
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review
from tests.claims.review_archive.cli._support import arguments, write_case, write_prepared


def test_command_requires_an_explicit_result_source(monkeypatch, capsys):
    arguments(monkeypatch)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Select --demo-dir, --review-file, or --claim-id" in capsys.readouterr().err


def test_review_file_exports_one_case_with_copies_and_replay_preserves_it(tmp_path, monkeypatch, capsys):
    review = recorded_review()
    case = write_case(tmp_path / "original", review)
    snapshot, output = tmp_path / "recorded.json", tmp_path / "analysis_results"
    snapshot.write_text(review.model_dump_json(), encoding="utf-8")
    arguments(
        monkeypatch,
        "--review-file",
        snapshot,
        "--case-dir",
        case,
        "--claim-id",
        "CLAIM-0001",
        "--analysis-id",
        "ONE-EXPORT",
        "--title",
        "Collision avant",
        "--output-dir",
        output,
    )
    assert main() == 0
    printed = capsys.readouterr().out
    assert "Exported CLAIM-0001:" in printed and "index.html" in printed
    bundles = list(output.glob("*/manifest.json"))
    assert len(bundles) == 1
    bundle = bundles[0].parent
    manifest = json.loads(bundles[0].read_text(encoding="utf-8"))
    assert manifest["title"] == "Collision avant"
    assert (bundle / "documents" / "declaration finale.pdf").read_bytes() == PDF
    assert (bundle / "images" / "photo originale.png").read_bytes() == IMAGE
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert main() == 1
    assert "Cannot export analysis results:" in capsys.readouterr().out
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert before == after


def test_review_file_claim_selector_cannot_export_a_different_identity(tmp_path, monkeypatch, capsys):
    snapshot, output = tmp_path / "recorded.json", tmp_path / "analysis_results"
    snapshot.write_text(recorded_review().model_dump_json(), encoding="utf-8")
    arguments(monkeypatch, "--review-file", snapshot, "--claim-id", "OTHER", "--output-dir", output)
    assert main() == 1
    assert "identity differs" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize("option,value", [("--title", "My report"), ("--analysis-id", "ONE-ID")])
def test_custom_title_or_identity_requires_selecting_one_prepared_claim(tmp_path, monkeypatch, capsys, option, value):
    first, second = recorded_review(), recorded_review()
    second.claim.claim_id = "CLAIM-0002"
    prepared = write_prepared(tmp_path / "prepared", [first, second])
    output = tmp_path / "analysis_results"
    arguments(monkeypatch, "--demo-dir", prepared, option, value, "--output-dir", output)
    assert main() == 1
    assert "single --claim-id" in capsys.readouterr().out
    assert not output.exists()
    arguments(monkeypatch, "--demo-dir", prepared, "--claim-id", "CLAIM-0002", option, value, "--output-dir", output)
    assert main() == 0
    catalogue = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["claim_id"] for entry in catalogue["analyses"]] == ["CLAIM-0002"]
