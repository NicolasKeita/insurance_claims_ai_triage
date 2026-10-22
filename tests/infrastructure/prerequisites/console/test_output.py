"""Readable root causes, actionable messages and preserved technical details."""

from types import SimpleNamespace

import pytest

from evaluation.preflight import PreflightReport
from evaluation.preflight.console import blocking_groups, print_preflight_row, print_preflight_summary
from tests.infrastructure.prerequisites.console._support import missing_services


def test_missing_services_are_two_causes_instead_of_five_errors(capsys, tmp_path):
    report = missing_services()
    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)
    output = capsys.readouterr().out
    assert "[SKIP] controlled : test non exécuté" in output
    assert "Composants bloquants : 2" in output
    assert output.count("- PostgreSQL :") == output.count("- Qdrant :") == 1
    assert "--database-url" in output and "--qdrant-url" in output
    assert "Non vérifiés : collections de recherche documentaire, collection des sinistres similaires" in output
    assert "Aucun modèle chargé ; aucun scénario exécuté" in output
    assert "RuntimeError" not in output and "stored_claim:" not in output
    assert "preflight.json" in output
    assert "RuntimeError" in report.rows[0]["reason"]


def test_present_service_does_not_hide_its_missing_data(capsys, tmp_path):
    report = missing_services()
    report.rows[0].update(status="PASS", reason=None)
    report.rows[-1]["missing_components"].remove("postgresql")
    groups = blocking_groups(report, "controlled")
    assert "stored_claim:controlled" in groups
    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)
    output = capsys.readouterr().out
    assert "schéma ou données du test indisponibles" in output
    assert "--isolated-demo-stack" in output
    assert "PostgreSQL : connexion non configurée" not in output


def test_configuration_failure_and_explicit_fail_policy(capsys, tmp_path):
    report = PreflightReport(
        rows=[{"component": "configuration", "status": "FAIL", "reason": "FileNotFoundError: missing.toml"}],
        blocked_scenarios={"controlled": ["configuration: FileNotFoundError: missing.toml"]},
    )
    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)
    output = capsys.readouterr().out
    assert "[FAIL] controlled" in output and "missing.toml" in output
    assert "Composants bloquants : 1" in output


def test_mixed_selection_does_not_claim_no_models_loaded(capsys, tmp_path):
    report = missing_services()
    report.rows.append({"component": "triage", "status": "PASS"})
    scenarios = [SimpleNamespace(scenario_id=name) for name in ("controlled", "ready")]
    print_preflight_summary(report, scenarios, tmp_path)
    output = capsys.readouterr().out
    assert "[PRÊT] ready" in output and "[SKIP] controlled" in output
    assert "Aucun modèle chargé" not in output


@pytest.mark.parametrize("name", ["postgresql", "qdrant", "prerequisites:controlled", "stored_claim:controlled"])
def test_failed_progress_rows_are_not_repeated_as_raw_errors(capsys, name):
    print_preflight_row({"component": name, "status": "SKIP", "reason": "RuntimeError: unavailable"})
    assert capsys.readouterr().out == ""


def test_available_model_is_readable_without_a_hash_dump(capsys):
    print_preflight_row({"component": "ollama", "status": "PASS", "version": "claims-e2e-partiel"})
    print_preflight_row({"component": "triage_files", "status": "PASS", "version": "sha256:unhelpful"})
    output = capsys.readouterr().out
    assert "[OK] Ollama (claims-e2e-partiel)" in output
    assert "Fichier du modèle de triage" in output and "sha256:" not in output
