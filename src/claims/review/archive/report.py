"""Render portable review reports; all report assets stay inside the export."""

from collections.abc import Mapping, Sequence

from claims.review.models import ClaimReview

from .reporting.content import _audit, _recommendation
from .reporting.layout import _assets_sections, _header, _overview
from .reporting.style import _page
from .reporting.text import _relative_url, _text


def render_report(review: ClaimReview, manifest: dict, *, translations: Mapping[str, str] | None = None) -> str:
    """Build an offline French report from a frozen review and copied asset manifest."""
    title = manifest.get("title") or f"Sinistre {review.claim.claim_id}"
    files = manifest.get("files", [])
    photos = [asset for asset in files if asset.get("kind") == "image"]
    documents = [asset for asset in files if asset.get("kind") == "document"]
    body = (
        _header(review, title)
        + _overview(review, photos, translations)
        + _assets_sections(review, photos, documents)
        + _recommendation(review, translations)
        + _audit(review, manifest)
        + '<footer class="footer">Ce dossier contient ses propres pièces et résultats. '
        "Sa consultation ne lance aucune nouvelle analyse.</footer>"
    )
    return _page(title, body)


def render_index(entries: Sequence[dict]) -> str:
    """Render a portable catalogue; entries point to reports inside analysis_results."""
    cards = []
    for entry in entries:
        url = _relative_url(entry.get("path"))
        title = _text(entry.get("title") or entry.get("claim_id"))
        heading = f'<a href="{url}">{title}</a>' if url else title
        cards.append(
            f'<article class="card catalogue-card"><p class="eyebrow">Analyse enregistrée</p><h2>{heading}</h2>'
            f"<p><code>{_text(entry.get('claim_id'))}</code></p>"
            f"<p><small>{_text(entry.get('analysis_id'))}</small></p>"
            f"<p><small>Exporté le {_text(entry.get('exported_at') or 'Non renseigné')}</small></p></article>"
        )
    body = (
        '<header class="header"><p class="eyebrow">Analysis results</p><h1>Dossiers analysés</h1>'
        '<p class="muted">Ouvrez un dossier pour lire son résultat, '
        "puis consulter les pièces jointes copiées avec son analyse.</p></header>"
        f'<div class="grid">{"".join(cards) or "<p>Aucune analyse exportée.</p>"}</div>'
        '<footer class="footer">Les rapports sont autonomes et consultables sans backend.</footer>'
    )
    return _page("Dossiers analysés", body)
