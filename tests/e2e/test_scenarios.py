"""The benchmark is a reviewable, versioned fixture, rather than opaque code."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation import E2E_BENCHMARK_VERSION
from evaluation.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios

EXPECTED_IDS = {
    "simple_coherent_claim",
    "additional_quote_damage",
    "missing_documents",
    "recent_claim_history",
    "statistically_unusual_claim",
    "policy_answerable",
    "policy_insufficient_evidence",
    "agent_missing_tool_data",
    "visual_additional_damage",
    "visual_non_detection",
}


def write_manifest(tmp_path, update):
    manifest = json.loads(DEFAULT_SCENARIOS_PATH.read_text(encoding="utf-8"))
    update(manifest)
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_versioned_benchmark_has_all_golden_cases_and_existing_declared_files():
    scenarios = load_scenarios()
    assert {scenario.scenario_id for scenario in scenarios} == EXPECTED_IDS
    assert E2E_BENCHMARK_VERSION == "e2e_claims_v2"
    root = Path(__file__).resolve().parents[2]
    for scenario in scenarios:
        assert scenario.expected_facts, scenario.scenario_id
        assert scenario.acceptable_agent_actions, scenario.scenario_id
        assert scenario.real_agent_action_policy is not None, scenario.scenario_id
        assert scenario.real_agent_action_policy.otherwise_acceptable_actions, scenario.scenario_id
        assert scenario.files, scenario.scenario_id
        for file in scenario.files:
            assert (root / file).is_file(), (scenario.scenario_id, file)


@pytest.mark.parametrize("version", ["", "e2e_claims_v0", "e2e_claims_v1", "e2e_claims_v3"])
def test_unknown_version_is_rejected_before_execution(tmp_path, version):
    path = write_manifest(tmp_path, lambda data: data.update(benchmark_version=version))
    with pytest.raises(ValidationError, match="Unsupported E2E benchmark version"):
        load_scenarios(path)


def test_duplicate_ids_and_unknown_schema_fields_are_rejected(tmp_path):
    path = write_manifest(tmp_path, lambda data: data["scenarios"].append(data["scenarios"][0]))
    with pytest.raises(ValidationError, match="unique"):
        load_scenarios(path)
    path = write_manifest(tmp_path, lambda data: data["scenarios"][0].update(unknown_expectation=True))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_scenarios(path)


def test_filter_preserves_benchmark_order_and_rejects_typos():
    selected = load_scenarios(scenario_ids=["visual_non_detection", "simple_coherent_claim"])
    assert [scenario.scenario_id for scenario in selected] == ["simple_coherent_claim", "visual_non_detection"]
    with pytest.raises(ValueError, match="Unknown E2E scenario IDs: coherent_claim_typo"):
        load_scenarios(scenario_ids=["coherent_claim_typo"])
