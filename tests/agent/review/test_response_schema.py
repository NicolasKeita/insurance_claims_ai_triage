"""Generation references are restricted without changing stored recommendation models."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.models import AgentRecommendation
from agent.planner._responses import recommendation_constraints
from claims.llm import OllamaStructuredLlm
from tests.infrastructure.support.agent_fakes import make_evidence, make_recommendation

COLLECTED_IDS = ("claim:current", "investigation:8347-4770-97e4")


def response_model(*, repair=False):
    evidence = [make_evidence(identifier).model_dump(mode="json") for identifier in COLLECTED_IDS]
    if repair:
        evidence[-1].update(
            source_type="INVESTIGATION_ASSESSMENT",
            structured_data={"signals": [{"code": "ADDITIONAL_QUOTE_DAMAGE"}]},
        )
    return recommendation_constraints({item["evidence_id"]: item for item in evidence})[0]


def test_reference_enum_and_versions_reach_mocked_ollama_client(monkeypatch):
    calls = []
    original_schema = AgentRecommendation.model_json_schema()
    recommendation = make_recommendation(list(COLLECTED_IDS))

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=recommendation.model_dump_json()))

    monkeypatch.setattr("claims.llm.Client", RecordingClient)
    model = response_model(repair=True)
    output = OllamaStructuredLlm("unused").generate(prompt="Synthetic data", response_model=model)
    schema = calls[0]["format"]
    top_level = schema["properties"]
    rationale_ref = top_level["rationale"]["items"]["$ref"].rsplit("/", 1)[-1]
    rationale_fields = schema["$defs"][rationale_ref]["properties"]
    assert top_level["evidence_ids"]["items"]["enum"] == list(COLLECTED_IDS)
    assert rationale_fields["evidence_ids"]["items"]["enum"] == list(COLLECTED_IDS)
    assert rationale_fields["evidence_ids"]["minItems"] == 1
    assert top_level["rationale"]["minItems"] == 1
    assert top_level["agent_version"]["const"] == CLAIMS_AGENT_VERSION
    assert top_level["prompt_version"]["const"] == CLAIMS_AGENT_PROMPT_VERSION
    assert output.model_dump() == recommendation.model_dump()
    assert AgentRecommendation.model_json_schema() == original_schema


@pytest.mark.parametrize("location", ["evidence_ids", "rationale"])
def test_invented_references_are_rejected_without_rewriting(location):
    recommendation = make_recommendation(list(COLLECTED_IDS)).model_dump(mode="json")
    target = (
        recommendation["evidence_ids"] if location == "evidence_ids" else recommendation["rationale"][0]["evidence_ids"]
    )
    target.append("investigation:invented")
    with pytest.raises(ValidationError):
        response_model().model_validate(recommendation)
    assert "investigation:invented" in target


@pytest.mark.parametrize("field", ["agent_version", "prompt_version"])
def test_generated_versions_must_be_exact_current_versions(field):
    recommendation = make_recommendation(list(COLLECTED_IDS)).model_dump(mode="json")
    recommendation[field] = "wrong-version"
    with pytest.raises(ValidationError):
        response_model().model_validate(recommendation)
    assert AgentRecommendation.model_validate(recommendation)


def test_supplied_rationale_requires_reference_but_abstention_may_omit_rationale():
    recommendation = make_recommendation(list(COLLECTED_IDS)).model_dump(mode="json")
    recommendation["rationale"][0]["evidence_ids"] = []
    with pytest.raises(ValidationError):
        response_model().model_validate(recommendation)
    recommendation.update(recommended_next_action="NO_RECOMMENDATION", rationale=[], evidence_ids=[])
    assert response_model().model_validate(recommendation).rationale == []
    recommendation["recommended_next_action"] = "REFER_TO_EXPERT"
    with pytest.raises(ValidationError):
        response_model(repair=True).model_validate(recommendation)
