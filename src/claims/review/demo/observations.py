"""Import existing real-run observations; never launch evaluation or inference."""

import json
from datetime import datetime
from pathlib import Path

from agent.models import AgentRunResult
from claims.consistency import ClaimConsistencyReport
from claims.investigation.models import InvestigationAssessment
from claims.models import Claim
from claims.review.assembly import build_review
from claims.review.demo.observation_fields import (
    anomaly_section,
    documents_section,
    filter_agent_similar,
    filter_similar,
    knowledge_section,
    validated_history,
)
from claims.review.demo.preparation import TITLES
from claims.review.demo.state import knowledge_state, section
from claims.review.demo.vision import import_vision
from claims.review.models import ReviewSection
from evaluation.scenarios import load_scenarios
from ml.inference import TriagePrediction
from retrieval.models import SimilarClaimsResponse


def import_observations(root, sources):
    scenarios = {scenario.scenario_id: scenario for scenario in load_scenarios()}
    entries = {}
    for source in sources:
        source = Path(source).resolve()
        paths = [source] if source.is_file() else sorted(source.rglob("observations/*.json"))
        if not paths:
            raise ValueError(f"No saved observations found under {source}")
        for path in paths:
            entry = _import_one(root, path, scenarios)
            entries[entry["claim_id"]] = entry
    return list(entries.values())


def _import_one(root, path, scenarios):
    observation = json.loads(path.read_text(encoding="utf-8"))
    summary = json.loads((path.parent.parent / "summary.json").read_text(encoding="utf-8"))
    if summary.get("mode") != "real" or observation.get("provenance", {}).get("mode") != "real":
        raise ValueError("Only saved real observations can be imported; deterministic model fakes are excluded")
    if any("fake" in str(value).lower() for value in observation.get("model_versions", {}).values()):
        raise ValueError("Saved observation contains a fake model identifier")
    scenario = scenarios[path.stem]
    claim = Claim.model_validate(observation["facts"]["claim"])
    if claim.claim_id != scenario.inputs["claim_id"] or set(claim.declared_damage) != set(
        scenario.inputs["declared_damage"]
    ):
        raise ValueError("Saved observation identity/declarations differ from the canonical scenario")
    timestamp = datetime.fromisoformat(summary["created_at"].replace("Z", "+00:00"))
    sections = _sections(observation, claim, timestamp)
    images = import_vision(root, scenario, observation, sections, timestamp)
    review = build_review(claim, sections, mode="demo")
    snapshot = f"saved-{scenario.scenario_id}.json"
    (root / snapshot).write_text(review.model_dump_json(indent=2), encoding="utf-8")
    return {
        "claim_id": claim.claim_id,
        "scenario_id": scenario.scenario_id,
        "title": TITLES.get(scenario.scenario_id, scenario.scenario_id.replace("_", " ").capitalize()),
        "description": f"Saved real results from {timestamp.date()}; controlled synthetic claim. Read-only review.",
        "snapshot": snapshot,
        "images": images,
    }


def _sections(observation, claim, timestamp):
    facts, sections = observation["facts"], {}
    for name, model in (
        ("consistency", ClaimConsistencyReport),
        ("triage", TriagePrediction),
        ("history", None),
        ("investigation", InvestigationAssessment),
        ("similar_claims", SimilarClaimsResponse),
    ):
        value = observation.get(name)
        if value is None:
            value = facts.get(name)
        try:
            if name == "history":
                value = validated_history(value)
            elif value is not None:
                value = model.model_validate(value).model_dump(mode="json")
            if name == "similar_claims":
                value = filter_similar(value, claim)
        except ValueError:
            sections[name] = ReviewSection(status="FAILED", reason="The saved assessment is incomplete or malformed.")
            continue
        sections[name] = section(name, value, observation, claim.claim_id, timestamp)
    values = {
        "anomaly": anomaly_section(observation),
        "documents": documents_section(observation, claim),
        "knowledge": knowledge_section(observation),
        "agent": observation.get("agent") or facts.get("agent"),
    }
    if values["agent"]:
        values["agent"] = AgentRunResult.model_validate(values["agent"]).model_dump(mode="json", exclude={"trace"})
        values["agent"] = filter_agent_similar(values["agent"], claim)
    for name, value in values.items():
        sections[name] = section(name, value, observation, claim.claim_id, timestamp)
    if values["agent"] and values["agent"]["status"] == "FAILED":
        sections["agent"].status = "FAILED"
        sections["agent"].reason = "The saved agent run failed; no recommendation is available."
    sections["knowledge"] = knowledge_state(sections["knowledge"], observation)
    return sections
