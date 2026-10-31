# Résultats des analyses

Ce dossier reçoit les exports autonomes des dossiers de sinistre déjà analysés.
Ouvrir `index.html` pour choisir une analyse, puis `rapport.html` dans son dossier.
Les PDF et photos sont **copiés**, avec les résultats figés et leur provenance.
Les rapports se consultent hors ligne, sans backend et sans relancer les modèles.

```text
analysis_results/
├── index.html
├── manifest.json
└── CLAIM-0001_ANALYSIS-…/
    ├── rapport.html
    ├── resultats.json
    ├── manifest.json
    ├── documents/
    └── images/
```

Depuis la racine du projet, exporter des revues préparées existantes :

```powershell
.\.venv\Scripts\python.exe scripts/export_analysis_results.py --demo-dir artifacts/review_demo_step33_saved
```

Pour fournir les PDF originaux d'un autre dossier, ajouter
`--case-dir chemin/du/dossier` (contenant `claim.json`, `documents/`, `images/`).
L'option peut être répétée pour plusieurs dossiers. Les faits du fichier
`claim.json` doivent correspondre à ceux de la revue exportée. Aucun PDF n'est
recréé à partir d'un résultat d'extraction ; les originaux absents sont signalés.

Exporter les résultats déjà enregistrés en PostgreSQL :

```powershell
.\.venv\Scripts\python.exe scripts/export_analysis_results.py --claim-id CLAIM-0001 --asset-root data
```

La variable `DATABASE_URL` doit être définie. Les pièces sont recherchées sous
`data/CLAIM-0001/documents/` et `data/CLAIM-0001/images/`. Autre entrée possible :
`--review-file chemin/revue.json` pour une revue au format `ClaimReview`.

Chaque export reçoit un identifiant issu de son contenu. Les résultats d'origine,
identifiants, dates et versions des composants sont conservés. Une capture des
derniers résultats peut combiner plusieurs dates d'analyse ; l'export ne prétend
pas qu'ils ont tous été calculés lors d'une seule exécution. Un export existant
n'est jamais écrasé. `--analysis-id` et `--title` permettent de nommer un export
d'un seul dossier explicitement sélectionné.

Copier ou compresser le sous-dossier d'une analyse permet de le transporter avec
toutes les pièces disponibles. Les sommes SHA256 et tailles sont enregistrées
dans son manifeste. Les exports générés restent locaux et sont ignorés par Git ;
ce README conserve le dossier et ses instructions dans le dépôt.

La console peut aussi charger ce même dossier ou toute la collection :

```powershell
.\.venv\Scripts\python.exe scripts/start_review_console.py --demo-dir analysis_results
```

Un lien `/?claim=CLAIM-0001&analysis=ANALYSIS-…` désigne une analyse figée précise.
Sans identifiant d'analyse, la collection utilise le dernier export de ce dossier.
Les boutons **Ouvrir le PDF** accèdent aux copies de l'archive ; les photos
et documents restent associés à l'analyse choisie même après un nouvel export.
