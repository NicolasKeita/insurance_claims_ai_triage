"""Recorded recommendations, component provenance and copied files."""

from claims.review.assembly import SECTION_NAMES

from .text import _data, _json, _label, _paragraph, _relative_url, _text


def _file(asset) -> str:
    title = _label(asset.get("document_type") or asset.get("filename"))
    url = _relative_url(asset.get("path")) if asset.get("status") == "COPIED" else None
    heading = f'<a href="{url}">{_text(title)}</a>' if url else _text(title)
    status = _label(asset.get("status"))
    details = ""
    if url:
        details = f"<small>{_text(asset.get('size_bytes', ''))} octets</small>"
        if asset.get("sha256"):
            details += f"<details><summary>Empreinte SHA-256</summary><code>{_text(asset['sha256'])}</code></details>"
    elif asset.get("reason"):
        details = f'<p class="muted">{_text(_label(asset["reason"]))}</p>'
    return (
        f'<article class="card file"><h3>{heading}</h3><code>{_text(asset.get("filename"))}</code>'
        f'<span class="badge">{_text(status)}</span>{details}</article>'
    )


def _recommendation(review, translations) -> str:
    agent = _data(review.agent)
    recommendation = (
        agent.get("recommendation")
        if review.agent.status == "AVAILABLE" and agent.get("status") == "COMPLETED"
        else None
    )
    if not isinstance(recommendation, dict):
        return (
            '<section id="recommendation" class="section card"><h2>Recommandation</h2>'
            f"<p>{_text(_label(review.agent.status))}</p>"
            + _paragraph(review.agent.reason or "No saved recommendation is available.", translations)
            + "</section>"
        )
    rationale = []
    for item in recommendation.get("rationale", []):
        if isinstance(item, dict):
            references = " · ".join(map(str, item.get("evidence_ids", [])))
            rationale.append(
                "<li>"
                + _paragraph(item.get("statement"), translations)
                + f'<p class="references">{_text(references)}</p></li>'
            )
    uncertainties = "".join(
        "<li>" + _paragraph(item, translations) + "</li>" for item in recommendation.get("uncertainties", [])
    )
    return (
        '<section id="recommendation" class="section card recommendation"><h2>Comprendre la recommandation</h2>'
        f"<h3>{_text(_label(recommendation.get('recommended_next_action')))}</h3>"
        + _paragraph(recommendation.get("case_summary"), translations)
        + "<details><summary>Justification et incertitudes</summary><div><h3>Justification</h3><ol>"
        + ("".join(rationale) or "<li>Aucune justification enregistrée.</li>")
        + "</ol><h3>Incertitudes à examiner</h3><ul>"
        + (uncertainties or "<li>Aucune incertitude enregistrée par cette exécution.</li>")
        + "</ul><p>La décision finale reste humaine.</p></div></details></section>"
    )


def _audit(review, manifest) -> str:
    rows = []
    for name in SECTION_NAMES:
        section = getattr(review, name)
        rows.append(
            f'<tr><th scope="row">{_text(_label(name))}</th><td>{_text(_label(section.status))}</td>'
            f"<td>{_text(section.updated_at.isoformat() if section.updated_at else 'Non renseigné')}</td>"
            f"<td>{_text(section.source or 'Non renseignée')}</td></tr>"
        )
    return (
        '<section id="provenance" class="section card"><h2>Historique et provenance</h2>'
        '<p class="muted">Cet export conserve les résultats enregistrés au moment de sa création. '
        "Les composants peuvent avoir été analysés à des dates différentes, indiquées ci-dessous.</p>"
        f"<p><small>Identifiant de l’analyse exportée : {_text(manifest.get('analysis_id'))}</small></p>"
        f"<p><small>Capture des résultats : {_text(manifest.get('captured_at') or 'Non renseignée')}</small></p>"
        f"<p><small>Export du dossier : {_text(manifest.get('exported_at') or 'Non renseigné')}</small></p>"
        '<details><summary>État et source de chaque composant</summary><div class="table-wrap"><table>'
        "<thead><tr><th>Composant</th><th>État</th><th>Date enregistrée</th><th>Source enregistrée</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></details>"
        "<details><summary>Versions et références conservées</summary>"
        f"<pre>{_json(review.audit.model_dump(mode='json'))}</pre></details>"
        f"<details><summary>Manifeste de l’export</summary><pre>{_json(manifest)}</pre></details>"
        '<p><small><a href="resultats.json">Résultats structurés JSON</a> · '
        '<a href="manifest.json">Manifeste JSON</a></small></p></section>'
    )
