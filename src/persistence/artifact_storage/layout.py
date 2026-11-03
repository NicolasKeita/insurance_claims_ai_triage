"""Explicit relocation map; directory names inside historical runs stay unchanged."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY_LOCATIONS = {
    "vision": "Modeles/vision",
    "triage_v1": "Modeles/triage_v1",
    "anomaly_v1": "Modeles/anomaly_v1",
    "mlflow": "Experiences_MLflow",
    "evaluation": "Evaluations",
    "agent": "Recherche_et_agents/agent",
    "rag": "Recherche_et_agents/rag",
    "similar_claims": "Recherche_et_agents/similar_claims",
    "step25": "Recherche_et_agents/historique_des_sinistres",
}

for _name in (
    "review_demo",
    "review_demo_minimal_check",
    "review_demo_runtime_20261029",
    "review_demo_step33_minimal",
    "review_demo_step33_saved",
    "review_ui_runtime_20261031",
    "review_archive_preview_20261031",
    "static_demo",
    "pdf_review",
    "unique_pdf_preview",
):
    LEGACY_LOCATIONS[_name] = f"Demonstrations/{_name}"

for _name in (
    "step29_validation",
    "test_infrastructure",
    "test_pytest_finalization",
    "test_skip_audit",
    "validation",
):
    LEGACY_LOCATIONS[_name] = f"Diagnostics_et_validation/{_name}"

for _name in ("environment_merge", "python312_migration", "refactor", "local_cleanup_20261022"):
    LEGACY_LOCATIONS[_name] = f"Historique_technique/{_name}"

for _name in (
    "postgres_validation.log",
    "pytest-visual-fix-20261026-root.log",
    "rag_ollama_stderr.log",
    "rag_ollama_stdout.log",
    "step32_pytest_hang.log",
    "step33_api_tests.xml",
):
    LEGACY_LOCATIONS[_name] = f"Diagnostics_et_validation/Journaux_et_rapports/{_name}"


def relocate_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve only known legacy paths inside this repository, without editing provenance."""
    repository = (root or ROOT).resolve()
    candidate = Path(str(path).replace("\\", "/")).expanduser()
    candidate = (candidate if candidate.is_absolute() else repository / candidate).resolve()
    try:
        relative = candidate.relative_to(repository / "artifacts")
    except ValueError:
        return candidate
    if relative.parts and relative.parts[0] in LEGACY_LOCATIONS:
        return repository / "artifacts" / LEGACY_LOCATIONS[relative.parts[0]] / Path(*relative.parts[1:])
    return candidate


def relocate_override(path: str | Path, *, root: Path | None = None) -> Path:
    """Keep explicit overrides relative to the caller's working directory when appropriate."""
    original = Path(path).expanduser()
    absolute = original.resolve()
    relocated = relocate_path(absolute, root=root)
    if relocated == absolute:
        return original
    return relocated if original.is_absolute() else Path(os.path.relpath(relocated))
