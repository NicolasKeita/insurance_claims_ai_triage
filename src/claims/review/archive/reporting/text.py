"""Escaped source text, labels and strictly relative asset URLs."""

import json
from html import escape
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import quote

_LABELS = {
    "AVAILABLE": "Disponible",
    "UNAVAILABLE": "Indisponible",
    "NOT_EXECUTED": "Non exécuté",
    "FAILED": "Échec",
    "NO_EVIDENCE": "Aucun élément retenu",
    "INSUFFICIENT_EVIDENCE": "Éléments insuffisants",
    "COPIED": "Copié dans ce dossier",
    "MISSING": "Fichier manquant",
    "NOT_PROVIDED": "Fichier non fourni",
    "No file was supplied with this claim.": "Aucun fichier n’a été fourni pour cette pièce du dossier.",
    "The original file was not supplied for export.": "Le fichier original n’a pas été fourni pour cet export.",
    "The registered source file could not be found.": "Le fichier original référencé est introuvable.",
    "CLAIM_FORM": "Déclaration de sinistre",
    "ACCIDENT_REPORT": "Constat",
    "REPAIR_QUOTE": "Devis de réparation",
    "POLICE_REPORT": "Rapport de police",
    "REQUEST_ADDITIONAL_INFORMATION": "Demander des informations complémentaires",
    "REQUEST_CLARIFICATION": "Demander une clarification",
    "CONTINUE_STANDARD_REVIEW": "Poursuivre l’examen habituel",
    "MANUAL_REVIEW": "Examen humain",
    "documents": "Documents",
    "consistency": "Cohérence documentaire",
    "triage": "Orientation du dossier",
    "anomaly": "Anomalies statistiques",
    "history": "Historique",
    "investigation": "Signaux de revue",
    "vision": "Observations visuelles",
    "visual_consistency": "Cohérence visuelle",
    "similar_claims": "Sinistres similaires",
    "knowledge": "Contrat et procédure",
    "agent": "Recommandation de l’agent",
}


def _text(value) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _label(value) -> str:
    return _LABELS.get(str(value), str(value or "Non renseigné").replace("_", " "))


def _relative_url(value) -> str | None:
    """Reject URLs, drives, parent traversal and protocol-relative asset links."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or any(part in {".", ".."} for part in path.parts):
        return None
    return quote(path.as_posix(), safe="/")


def _paragraph(value, translations=None, *, tag="p") -> str:
    source = str(value or "")
    translated = (translations or {}).get(source, source)
    lang = "fr" if translated != source else "en"
    return f'<{tag} class="source-text" lang="{lang}">{_text(translated)}</{tag}>'


def _data(section) -> dict:
    return section.data if isinstance(section.data, dict) else {}


def _json(value) -> str:
    return _text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
