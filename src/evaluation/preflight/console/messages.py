"""Readable prerequisite names and corrective actions for the terminal."""

LABELS = {
    "configuration": "Configuration du test",
    "retrieval_configuration": "Configuration de la recherche",
    "isolated_configuration": "Configuration du mode isolé",
    "postgresql": "PostgreSQL",
    "qdrant": "Qdrant",
    "ollama": "Ollama",
    "knowledge_collections": "collections de recherche documentaire",
    "similar_claims_collection": "collection des sinistres similaires",
    "scenario_files": "Fichiers du scénario",
    "stored_claim": "données du scénario",
    "triage_files": "Fichier du modèle de triage",
    "anomaly_files": "Fichier du modèle d'anomalie",
    "embedding_files": "Fichiers des embeddings",
    "damage_vision_files": "Fichier du modèle de dommages",
    "part_vision_files": "Fichier du modèle de pièces",
    "triage": "Modèle de triage chargé",
    "anomaly": "Modèle d'anomalie chargé",
    "embeddings": "Modèle d'embeddings chargé",
    "damage_vision": "Modèle de dommages chargé",
    "part_vision": "Modèle de pièces chargé",
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
        return "fonction requise désactivée", "Réactiver l'option nécessaire à ce scénario dans la configuration."
    if name == "postgresql":
        if "DATABASE_URL is not configured" in reason:
            return "connexion non configurée", "Fournir --database-url ou définir DATABASE_URL."
        return (
            "connexion impossible",
            "Vérifier que le service est démarré et que l'adresse et les identifiants sont corrects.",
        )
    if name == "qdrant":
        timed_out = "timed out" in reason.lower() or "timeout" in reason.lower()
        return (
            "service injoignable (délai dépassé)" if timed_out else "connexion impossible",
            "Vérifier le service et --qdrant-url (ou QDRANT_URL). Le service de test du projet utilise le port 6335.",
        )
    if name == "ollama":
        if "is not installed" in reason or "not configured" in reason:
            return (
                "modèle demandé absent ou non configuré",
                "Vérifier ollama list et sélectionner un modèle avec --llm-model.",
            )
        return "service injoignable", "Démarrer Ollama et vérifier OLLAMA_HOST."
    if name == "stored_claim":
        return (
            "schéma ou données du test indisponibles",
            "Préparer les données attendues ou utiliser --isolated-demo-stack avec les services de test.",
        )
    if name in {"knowledge_collections", "similar_claims_collection"}:
        return (
            "collection requise absente ou inaccessible",
            "Préparer les index attendus ou utiliser --isolated-demo-stack avec les services de test.",
        )
    detail = reason.partition(": ")[2] if ": " in reason else reason
    if name.endswith("_files") or name in {"scenario_files", "visual_golden_asset"}:
        return detail, "Vérifier la présence des fichiers indiqués et leurs chemins dans la configuration."
    return detail, "Corriger la configuration indiquée ; détails dans preflight.json."
