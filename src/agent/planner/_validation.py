"""Validate generated recommendations and permit one bounded grounding correction."""

import json

from pydantic import ValidationError

from agent.diagnostics import grounding_feedback
from agent.evidence import EvidenceValidationError, validate_recommendation, validate_recommendation_versions
from agent.models import AgentRecommendation, EvidenceItem
from agent.review_policy import follow_up_sources, unavailable_essential_tools
from claims.llm import LlmError

REPAIRABLE_CODES = {
    "UNKNOWN_EVIDENCE_ID",
    "VERSION_MISMATCH",
    "UNCITED_RATIONALE",
    "NO_SUPPORTING_RATIONALE",
    "FOLLOW_UP_ASSESSMENT_NOT_CITED",
    "INCOMPATIBLE_FOLLOW_UP",
    "INCOMPATIBLE_MISSING_DATA",
    "NO_EVIDENCE",
}


def _schema_grounding_error(error, *, action_error_code="INCOMPATIBLE_FOLLOW_UP"):
    error = error.__cause__ if isinstance(error, LlmError) else error
    if not isinstance(error, ValidationError):
        return None
    for item in error.errors(include_url=False, include_input=False, include_context=False):
        path = item["loc"]
        code = None
        if path and path[0] in {"agent_version", "prompt_version"}:
            code = "VERSION_MISMATCH"
        elif path == ("recommended_next_action",) and item["type"] == "literal_error":
            code = action_error_code
        elif "evidence_ids" in path and item["type"] in {"literal_error", "too_short"}:
            code = "UNCITED_RATIONALE" if item["type"] == "too_short" else "UNKNOWN_EVIDENCE_ID"
        elif path == ("rationale",) and item["type"] == "too_short":
            code = "NO_SUPPORTING_RATIONALE"
        if code is not None:
            return EvidenceValidationError("Generated response violates its grounding schema", code=code)
    return None


def _generate_checked(llm, prompt, response_model, registry, observations):
    output = llm.generate(prompt=prompt, response_model=response_model)
    checked = response_model.model_validate(output.model_dump())
    recommendation = AgentRecommendation.model_validate(checked.model_dump())
    validate_recommendation_versions(recommendation)
    validate_recommendation(recommendation, registry, observations=observations)
    return recommendation


def generate_validated(llm, prompt, response_model, registry, *, observations=()):
    """One original call plus at most one correction; never rewrite a model decision or citation.

    Feedback contains static application messages only, not a draft or exception.
    A failed correction retains its grounding failure instead of returning a fallback.
    """
    registry = {key: EvidenceItem.model_validate(value) for key, value in registry.items()}
    observations = tuple(observations)
    action_error_code = (
        "INCOMPATIBLE_MISSING_DATA"
        if unavailable_essential_tools(observations) and not follow_up_sources(registry)
        else "INCOMPATIBLE_FOLLOW_UP"
    )
    original_prompt, last_error = prompt, None
    for attempt in range(2):
        try:
            return _generate_checked(llm, prompt, response_model, registry, observations)
        except EvidenceValidationError as error:
            last_error = error
        except (ValidationError, LlmError) as error:
            grounding_error = _schema_grounding_error(error, action_error_code=action_error_code)
            if grounding_error is None:
                if last_error is not None:
                    raise last_error from None
                raise
            last_error = grounding_error
        except Exception:
            if last_error is None:
                raise
            raise last_error from None
        if attempt or last_error.code not in REPAIRABLE_CODES:
            raise last_error
        prompt = (
            original_prompt + "\n\nAPPLICATION VALIDATION FEEDBACK:\n" + json.dumps(grounding_feedback(last_error.code))
        )
    raise RuntimeError("Recommendation validation did not complete")
