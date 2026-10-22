"""Readable prerequisite names and corrective actions for the terminal."""

LABELS = {
    "configuration": "Test configuration",
    "retrieval_configuration": "Retrieval configuration",
    "isolated_configuration": "Isolated setup configuration",
    "postgresql": "PostgreSQL",
    "qdrant": "Qdrant",
    "ollama": "Ollama",
    "knowledge_collections": "knowledge collections",
    "similar_claims_collection": "similar claims collection",
    "scenario_files": "Scenario files",
    "stored_claim": "scenario data",
    "triage_files": "Triage model file",
    "anomaly_files": "Anomaly model file",
    "embedding_files": "Embedding model files",
    "damage_vision_files": "Damage model file",
    "part_vision_files": "Vehicle parts model file",
    "triage": "Triage model loaded",
    "anomaly": "Anomaly model loaded",
    "embeddings": "Embedding model loaded",
    "damage_vision": "Damage model loaded",
    "part_vision": "Vehicle parts model loaded",
}
PARENTS = {
    "knowledge_collections": ("qdrant", "retrieval_configuration"),
    "similar_claims_collection": ("qdrant", "retrieval_configuration"),
    "qdrant": ("retrieval_configuration",),
    "stored_claim": ("postgresql",),
    "embedding_files": ("retrieval_configuration",),
    "embeddings": ("embedding_files",),
    "triage": ("triage_files",),
    "anomaly": ("anomaly_files",),
    "damage_vision": ("damage_vision_files",),
    "part_vision": ("part_vision_files",),
}


def label(component):
    return LABELS.get(component.split(":", 1)[0], component)


def explanation(component, reason):
    """Keep exception types in JSON; show causes and next actions in the terminal."""
    name = component.split(":", 1)[0]
    if reason == "Disabled by configuration":
        return "required feature disabled", "Enable the option required by this scenario in the configuration."
    if name == "postgresql":
        if "DATABASE_URL is not configured" in reason:
            return "connection not configured", "Provide --database-url or set DATABASE_URL."
        return (
            "connection failed",
            "Check that the service is running and that the address and credentials are correct.",
        )
    if name == "qdrant":
        timed_out = "timed out" in reason.lower() or "timeout" in reason.lower()
        return (
            "service unreachable (request timed out)" if timed_out else "connection failed",
            "Check the service and --qdrant-url (or QDRANT_URL). The project's test service uses port 6335.",
        )
    if name == "ollama":
        if "is not installed" in reason or "not configured" in reason:
            return (
                "requested model missing or not configured",
                "Check ollama list and select a model with --llm-model.",
            )
        return "service unreachable", "Start Ollama and check OLLAMA_HOST."
    if name == "stored_claim":
        return (
            "required schema or test data unavailable",
            "Prepare the expected data or use --isolated-demo-stack with the test services.",
        )
    if name in {"knowledge_collections", "similar_claims_collection"}:
        return (
            "required collection missing or inaccessible",
            "Prepare the expected indexes or use --isolated-demo-stack with the test services.",
        )
    detail = reason.partition(": ")[2] if ": " in reason else reason
    if name.endswith("_files") or name in {"scenario_files", "visual_golden_asset"}:
        return detail, "Check that the listed files exist and that their configured paths are correct."
    return detail, "Fix the indicated configuration; see preflight.json for details."
