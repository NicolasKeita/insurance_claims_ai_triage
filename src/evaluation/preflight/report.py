"""Preflight rows, retained resources and version fingerprints."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.preflight.console import print_preflight_row as print_preflight_row

MODEL_FILES = {name: name + "_files" for name in ("triage", "anomaly", "damage_vision", "part_vision")}
FILE_CHECKS = {"embeddings": "embedding_files", **MODEL_FILES}
ISOLATED_COLLECTIONS = {"knowledge_collections", "similar_claims_collection"}


@dataclass
class PreflightReport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict, repr=False)
    blocked_scenarios: dict[str, list[str]] = field(default_factory=dict)

    def available(self, name: str) -> bool:
        return any(row["component"] == name and row["status"] == "PASS" for row in self.rows)

    def reason(self, name: str) -> str:
        return next((row.get("reason", "") for row in self.rows if row["component"] == name), "Not preflighted")

    def close(self):
        engine = self.resources.get("database_engine")
        if engine is not None:
            engine.dispose()
        client = self.resources.get("qdrant_client")
        if client is not None:
            client.close()


def checkpoint_version(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def requirements(config, scenario):
    """Combine the manifest's mandatory dependencies with the enabled stages."""
    names = set(scenario.infrastructure_requirements)
    if config.include_agent:
        names.update(("postgresql", "ollama"))
    if config.include_rag:
        names.update(("qdrant", "knowledge_collections", "embeddings", "ollama"))
    if config.include_similar:
        names.update(("postgresql", "qdrant", "similar_claims_collection", "embeddings"))
    if config.isolated_demo_stack:
        names.update(("isolated_configuration", "postgresql", "qdrant", "embeddings"))
    if config.include_vision and scenario.vision_required:
        names.update(("damage_vision", "part_vision"))
    return names


def record_eligibility(session, scenario, names, *, models_loaded=False):
    """Record explicit missing prerequisites separately from execution failures."""
    config = session.config
    checks = {"retrieval_configuration", f"scenario_files:{scenario.scenario_id}"}
    if scenario.inputs.get("unavailable_tools"):
        checks.add(f"scenario_agent_tools:{scenario.scenario_id}")
    for name in names:
        if config.isolated_demo_stack and name in ISOLATED_COLLECTIONS:
            continue
        checks.add(FILE_CHECKS.get(name, name))
    if "postgresql" in names and not config.isolated_demo_stack:
        checks.add(f"stored_claim:{scenario.scenario_id}")
    if models_loaded:
        checks.update(names & (MODEL_FILES.keys() | {"embeddings"}))
    missing = [
        f"{name}: {session.report.reason(name)}" for name in sorted(checks) if not session.report.available(name)
    ]
    if missing:
        session.report.blocked_scenarios[scenario.scenario_id] = missing
    row = {
        "component": f"prerequisites:{scenario.scenario_id}",
        "status": config.dependency_unavailable if missing else "PASS",
        "version": None,
        "reason": "; ".join(missing) if missing else None,
        "requirements": sorted(names),
        "missing_components": [name for name in sorted(checks) if not session.report.available(name)],
    }
    existing = next((item for item in session.report.rows if item["component"] == row["component"]), None)
    if existing is not None:
        existing.update(row)
    else:
        session.report.rows.append(row)
    if session.progress is not None and (not models_loaded or missing):
        session.progress(row)
