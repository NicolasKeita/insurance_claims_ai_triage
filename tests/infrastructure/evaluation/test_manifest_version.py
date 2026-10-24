"""Read versioned configuration only; no model, service or scenario is executed."""

import json

import pytest
from pydantic import ValidationError

from evaluation import E2E_BENCHMARK_VERSION
from evaluation.models import RealAgentActionPolicy
from evaluation.preflight.config import DEFAULT_CONFIG, RealEvaluationConfig
from evaluation.real.diagnostic_manifest import real_diagnostic_scenarios
from evaluation.scenarios import DEFAULT_SCENARIOS_PATH, ROOT, load_scenarios


def test_v2_preserves_deterministic_cases_and_declares_real_action_policies():
    legacy = json.loads((ROOT / "data/evaluation/e2e_v1/scenarios.json").read_text(encoding="utf-8"))
    current = json.loads(DEFAULT_SCENARIOS_PATH.read_text(encoding="utf-8"))
    assert legacy["benchmark_version"] == "e2e_claims_v1"
    assert current["benchmark_version"] == E2E_BENCHMARK_VERSION == "e2e_claims_v2"
    for before, after in zip(legacy["scenarios"], current["scenarios"], strict=True):
        policy = after.pop("real_agent_action_policy")
        if after["scenario_id"] in {"visual_additional_damage", "visual_non_detection"}:
            assets = after.pop("external_assets")
            assert len(assets) == 1 and assets[0]["origin"] == "ai_generated"
            assert after.pop("real_expected_facts")
            after["external_assets"] = []
        assert after == before
        assert policy["otherwise_acceptable_actions"]
    assert all(item.real_agent_action_policy is not None for item in load_scenarios())


def test_defaults_and_diagnostics_use_the_same_version_without_loading_services():
    config = RealEvaluationConfig.from_toml()
    assert DEFAULT_CONFIG == ROOT / "configs/evaluation/e2e_v2.toml"
    assert config.fixture_dir == ROOT / "artifacts/evaluation/e2e_v2/fixtures"
    assert [item.scenario_id for item in real_diagnostic_scenarios()] == [
        "reference_claim_documents",
        "local_cardd_multimodal_diagnostic",
    ]


def test_legacy_manifest_and_configuration_are_rejected_instead_of_mixing_contracts():
    with pytest.raises(ValidationError, match="Unsupported E2E benchmark version"):
        load_scenarios(ROOT / "data/evaluation/e2e_v1/scenarios.json")
    with pytest.raises(ValueError, match="benchmark_version"):
        RealEvaluationConfig.from_toml(ROOT / "configs/evaluation/e2e_v1.toml")


@pytest.mark.parametrize("actions", [[], ["APPROVE_CLAIM"], ["invented_action"]])
def test_policy_rejects_empty_or_unsupported_actions(actions):
    with pytest.raises(ValidationError):
        RealAgentActionPolicy(otherwise_acceptable_actions=actions)
