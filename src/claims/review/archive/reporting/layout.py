"""Compact result-first report sections using only frozen recorded facts."""

from .content import _file
from .photos import _photo
from .text import _data, _label, _paragraph, _text


def _header(review, title):
    claim = review.claim
    amount = f"{claim.repair_estimate.amount:,.2f}".replace(",", " ").replace(".", ",")
    return (
        '<header class="header"><p class="eyebrow">Dossier de sinistre · Analyse enregistrée</p>'
        f'<h1>{_text(title)}</h1><p class="muted">{_text(claim.claim_id)}</p><dl class="facts">'
        "<div><dt>Véhicule</dt>"
        f"<dd>{_text(claim.vehicle.make)} {_text(claim.vehicle.model)} · {claim.vehicle.year}</dd></div>"
        "<div><dt>Sinistre</dt>"
        f"<dd>{_text(claim.incident.date.isoformat())} · {_text(claim.incident.location)}</dd></div>"
        "<div><dt>Devis de réparation</dt>"
        f"<dd>{_text(amount)} {_text(claim.repair_estimate.currency)}</dd></div></dl></header>"
        '<p class="notice">Résultats enregistrés · La décision finale reste humaine.</p>'
        '<nav class="navigation" aria-label="Rubriques du rapport"><a href="#resultats">1. Résultats du système</a>'
        '<a href="#documents">2. Documents</a><a href="#photos">3. Photos</a>'
        '<a href="#provenance">Provenance</a></nav>'
    )


def _overview(review, photos, translations):
    agent = _data(review.agent)
    recommendation = (
        agent.get("recommendation")
        if review.agent.status == "AVAILABLE" and agent.get("status") == "COMPLETED"
        else None
    )
    signals = _data(review.investigation).get("signals", []) if review.investigation.status == "AVAILABLE" else []
    result = signals[0].get("explanation") if signals and isinstance(signals[0], dict) else None
    if not result and isinstance(recommendation, dict):
        result = recommendation.get("case_summary")
    if not result:
        result = review.investigation.reason or review.agent.reason or "No saved assessment is available."
    action = (
        _label(recommendation.get("recommended_next_action"))
        if isinstance(recommendation, dict)
        else "Aucune recommandation enregistrée"
    )
    photo = (
        _photo(photos[0], review)
        if photos
        else '<div class="card photo-placeholder"><h3>Aucune photo jointe à cet export</h3></div>'
    )
    return (
        f'<section id="resultats" class="hero" aria-label="Résultats du système">{photo}<div class="card result">'
        "<h2>Ce que le système a relevé</h2>"
        + _paragraph(result, translations)
        + f'<p class="eyebrow">Prochaine étape recommandée</p><h3>{_text(action)}</h3>'
        '<a class="button" href="#recommendation">Comprendre la recommandation →</a></div></section>'
    )


def _assets_sections(review, photos, documents):
    return (
        '<section id="documents" class="section"><h2>Documents du dossier</h2><div class="grid">'
        + ("".join(_file(asset) for asset in documents) or '<p class="muted">Aucun document joint à cet export.</p>')
        + '</div></section><section id="photos" class="section"><h2>Photographies et annotations</h2>'
        '<p class="muted">Les contours affichés proviennent des observations enregistrées.</p><div class="grid">'
        + (
            "".join(_photo(asset, review) for asset in photos)
            or '<p class="muted">Aucune photo jointe à cet export.</p>'
        )
        + "</div></section>"
    )
