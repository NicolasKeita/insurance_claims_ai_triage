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
    assert "[SKIP] controlled: test not run" in output
    assert "Blocking components: 2" in output
    assert output.count("- PostgreSQL (Relation DB):") == output.count("- Qdrant (Vector DB):") == 1
    assert "Required for: Accessing claim history and assessments used by the agent." in output
    assert "Required for: Retrieving relevant policy passages and similar claims." in output
    assert "--database-url" in output and "--qdrant-url" in output
    assert "Not checked: knowledge collections, similar claims collection" in output
    assert "No models loaded; no scenarios executed" in output
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
    assert "required schema or test data unavailable" in output
    assert "--isolated-demo-stack" in output
    assert "PostgreSQL (Relation DB): connection not configured" not in output


def test_configuration_failure_and_explicit_fail_policy(capsys, tmp_path):
    report = PreflightReport(
        rows=[{"component": "configuration", "status": "FAIL", "reason": "FileNotFoundError: missing.toml"}],
        blocked_scenarios={"controlled": ["configuration: FileNotFoundError: missing.toml"]},
    )
    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)
    output = capsys.readouterr().out
    assert "[FAIL] controlled" in output and "missing.toml" in output
    assert "Blocking components: 1" in output


def test_mixed_selection_does_not_claim_no_models_loaded(capsys, tmp_path):
    report = missing_services()
    report.rows.append({"component": "triage", "status": "PASS"})
    scenarios = [SimpleNamespace(scenario_id=name) for name in ("controlled", "ready")]
    print_preflight_summary(report, scenarios, tmp_path)
    output = capsys.readouterr().out
    assert "[READY] ready" in output and "[SKIP] controlled" in output
    assert "No models loaded" not in output


@pytest.mark.parametrize("name", ["postgresql", "qdrant", "prerequisites:controlled", "stored_claim:controlled"])
def test_failed_progress_rows_are_not_repeated_as_raw_errors(capsys, name):
    print_preflight_row({"component": name, "status": "SKIP", "reason": "RuntimeError: unavailable"})
    assert capsys.readouterr().out == ""


def test_available_model_is_readable_without_a_hash_dump(capsys):
    print_preflight_row({"component": "ollama", "status": "PASS", "version": "claims-e2e-partiel"})
    print_preflight_row({"component": "triage_files", "status": "PASS", "version": "sha256:unhelpful"})
    print_preflight_row({"component": "postgresql", "status": "PASS"})
    print_preflight_row({"component": "qdrant", "status": "PASS"})
    output = capsys.readouterr().out
    assert "[OK] Ollama (claims-e2e-partiel)" in output
    assert "Triage model file" in output and "sha256:" not in output
    assert "[OK] PostgreSQL (Relation DB)" in output and "[OK] Qdrant (Vector DB)" in output
    assert "Required for: Accessing claim history" in output
    assert "Required for: Retrieving relevant policy passages" in output


@pytest.mark.parametrize(
    "reason,action",
    [
        ("Controlled agent tool outages require --include-agent", "Enable --include-agent"),
        ("Controlled agent tool outages require --isolated-demo-stack", "Use --isolated-demo-stack"),
        ("Unsupported controlled agent tool outages: GET_CLAIM", "Select a supported controlled tool outage"),
    ],
)
def test_unverifiable_agent_outage_has_a_readable_preflight_action(capsys, tmp_path, reason, action):
    name = "scenario_agent_tools:controlled"
    report = PreflightReport(
        rows=[
            {"component": name, "status": "SKIP", "reason": f"RuntimeError: {reason}"},
            {"component": "prerequisites:controlled", "status": "SKIP", "missing_components": [name]},
        ],
        blocked_scenarios={"controlled": [f"{name}: RuntimeError: {reason}"]},
    )

    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)

    output = capsys.readouterr().out
    assert "[SKIP] controlled: test not run" in output
    assert "Scenario agent tool outages" in output
    assert reason in output
    assert action in output
    assert "No models loaded; no scenarios executed" in output
    assert "RuntimeError" not in output


@pytest.mark.parametrize("compose_exists,db_blocked", [(True, True), (False, True), (True, False)])
def test_compose_hint_requires_an_existing_file_and_a_blocked_database(
    capsys, tmp_path, monkeypatch, compose_exists, db_blocked
):
    import evaluation.preflight.console as console

    monkeypatch.setattr(console, "ROOT", tmp_path)
    if compose_exists:
        (tmp_path / "compose.yaml").write_text("services: {}", encoding="utf-8")
    report = missing_services() if db_blocked else PreflightReport()
    print_preflight_summary(report, [SimpleNamespace(scenario_id="controlled")], tmp_path)
    output = capsys.readouterr().out
    command = "docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test"
    assert output.count(command) == int(compose_exists and db_blocked)
    if compose_exists and db_blocked:
        assert "1. On Windows, open Docker Desktop and wait for its engine to be ready." in output
        assert "2. From the repository root, run:" in output
        assert output.index("open Docker Desktop") < output.index(command)
        assert "--database-url" in output and "http://127.0.0.1:6335" in output
        assert "--isolated-demo-stack" in output
