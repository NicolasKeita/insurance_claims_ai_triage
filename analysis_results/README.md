# Résultats des analyses

- Exports de sinistres déjà analysés. Aucun modèle relancé.
- Ouvrir `index.html`, puis le `rapport.html` choisi. Consultation hors ligne.
- Chaque export contient les résultats JSON, les PDF/photos disponibles et un manifeste de provenance.

## Exporter

Depuis la racine, avec [l'environnement Python du projet](../docs/INSTALLATION.md) :

```powershell
.\.venv\Scripts\python.exe scripts/export_analysis_results.py --demo-dir artifacts/Demonstrations/review_demo_step33_saved
```

- `--review-file chemin/revue.json` : autre source, au format `ClaimReview`.
- `--case-dir chemin/dossier` : originaux avec `claim.json`, `documents/`, `images/`. Répétable ; les faits doivent correspondre.
- Les exports existants sont préservés. Les pièces absentes restent signalées.
- Pour partager une analyse, copier ou compresser son sous-dossier complet.

Pour les résultats PostgreSQL, définir `DATABASE_URL`, puis :

```powershell
.\.venv\Scripts\python.exe scripts/export_analysis_results.py --claim-id CLAIM-0001 --asset-root data
```

## Consulter dans la console

```powershell
.\.venv\Scripts\python.exe scripts/start_review_console.py --demo-dir analysis_results
```

- `/?claim=CLAIM-0001&analysis=ANALYSIS-…` sélectionne un export précis.
- Sans `analysis`, la console prend le dernier export du sinistre.
- [Démo statique et API d'archives avec Docker](../deploy/docker/README.md).
