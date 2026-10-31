"""Selection and database failures remain actionable without exposing internals."""

from sqlalchemy.exc import SQLAlchemyError

from scripts import export_analysis_results as command
from tests.claims.review_archive._support import recorded_review
from tests.claims.review_archive.cli._support import arguments, write_prepared


def test_missing_prepared_claim_reports_a_human_error_without_partial_export(tmp_path, monkeypatch, capsys):
    root = write_prepared(tmp_path / "prepared", [recorded_review()])
    output = tmp_path / "analysis_results"
    arguments(monkeypatch, "--demo-dir", root, "--claim-id", "UNKNOWN", "--output-dir", output)
    assert command.main() == 1
    message = capsys.readouterr().out
    assert "Cannot export analysis results:" in message
    assert "UNKNOWN" in message
    assert not output.exists()


def test_database_failure_is_sanitized_without_connecting_to_a_database(tmp_path, monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise SQLAlchemyError("password=private-token host=private-db")

    monkeypatch.setattr(command, "_database", unavailable)
    output = tmp_path / "analysis_results"
    arguments(monkeypatch, "--claim-id", "CLAIM-0001", "--output-dir", output)
    assert command.main() == 1
    message = capsys.readouterr().out
    assert "DATABASE_URL" in message and "PostgreSQL" in message
    assert "private-token" not in message and "private-db" not in message
    assert not output.exists()
