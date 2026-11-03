# Organisation des artefacts locaux

`artifacts/` contient les fichiers produits par les entraînements, les évaluations,
les démonstrations et les diagnostics. Ce répertoire est ignoré par Git. Il contient
des modèles et des résultats utiles : un fichier ignoré n'est pas nécessairement
temporaire, et Git ne permet pas de le restaurer après suppression.

## Structure et utilité

| Répertoire | Contenu et utilisation |
| --- | --- |
| `Modeles/vision/` | Checkpoints de segmentation des dommages et des pièces, résultats d'entraînement, manifestes et splits des données. Les commandes d'inférence et d'évaluation chargent les checkpoints locaux. |
| `Modeles/triage_v1/` | Modèle de triage entraîné, fichiers associés et résultats d'analyse. |
| `Modeles/anomaly_v1/` | Détecteur d'anomalies entraîné et métadonnées associées. |
| `Experiences_MLflow/` | Suivi des expériences MLflow : bases locales, modèles, métriques et artefacts enregistrés. |
| `Evaluations/` | Exécutions et résultats des évaluations du système et de l'extraction documentaire : cas synthétiques, observations, rapports, métriques et traces. Certaines démonstrations ont été préparées à partir de ces résultats enregistrés. |
| `Demonstrations/` | Snapshots `review_demo*`, archives de revue, fichiers de lancement locaux, exports ZIP du site statique et rendus PDF de contrôle. |
| `Recherche_et_agents/` | Résultats des agents, évaluation RAG, manifestes de recherche de sinistres similaires et exemples d'historique des sinistres. |
| `Diagnostics_et_validation/` | Diagnostics vision, résultats de validation de l'infrastructure et des tests, audits et journaux. Les anciens fichiers isolés à la racine sont regroupés dans `Journaux_et_rapports/`. |
| `Historique_technique/` | Traces des anciennes migrations Python, fusions d'environnement, refactorisations et nettoyages ; journaux des réorganisations locales. |

Les noms historiques des modèles et exécutions sont conservés sous ces catégories.
Par exemple, le checkpoint des dommages se trouve dans
`artifacts/Modeles/vision/damage_segmentation_v1/train/weights/best.pt`.

Les modèles entraînés et les bases MLflow servent aux traitements locaux. Les
résultats d'évaluation, manifestes, snapshots et diagnostics gardent la provenance
des calculs déjà effectués. Les journaux, rendus de contrôle et anciennes archives
ZIP peuvent faire l'objet d'un tri distinct lorsqu'ils ne sont plus nécessaires.

## Démonstration statique

La démo statique utilise les cas, JSON, PDF, photographies et superpositions
versionnés sous `frontend/public/demo/`. La commande
`npm --prefix frontend run build:demo` les intègre au site dans `frontend/dist/`.
Le site construit ne lit pas `artifacts/` et ne charge aucun modèle.

`artifacts/Demonstrations/static_demo/` contient des archives ZIP d'anciens exports.
Ces archives ne sont pas nécessaires au fonctionnement ni à la construction du
site à partir du jeu de démonstration déjà versionné.

Pour renouveler explicitement les cas embarqués, l'exporteur lit une collection
d'analyses existante, par défaut `analysis_results/` :

```powershell
python scripts/export_static_demo.py --source-dir analysis_results --output-dir frontend/public/demo --replace
```

Les snapshots sous `artifacts/Demonstrations/review_demo*` et les résultats sous
`artifacts/Evaluations/` peuvent servir à préparer ou exporter une nouvelle
collection. La console connectée à l'API peut également lire un snapshot local
choisi par `--demo-dir` ou `CLAIMS_REVIEW_DEMO_DIR`. Ce sont des usages distincts
du site statique déjà construit.

## Réorganisation d'une ancienne copie locale

Depuis la racine du dépôt, la commande suivante affiche le plan sans déplacer les
fichiers :

```powershell
python scripts/reorganize_artifacts.py
```

Pour appliquer ce plan :

```powershell
python scripts/reorganize_artifacts.py --apply
```

La migration déplace les anciens dossiers dans les catégories ci-dessus. Les
noms des exécutions et les contenus historiques de résultats sont conservés.
En particulier, les anciens chemins enregistrés dans les JSON, CSV et journaux de
provenance ne sont pas réécrits pour refléter leur nouvel emplacement. Les chemins
utilisés par les configurations et les commandes actives suivent la nouvelle
organisation.

La migration enregistre son contrôle sous
`artifacts/Historique_technique/reorganisations/<horodatage UTC>/` :

| Fichier ou dossier | Rôle |
| --- | --- |
| `before.json` | Inventaire des fichiers avant déplacement et empreintes SHA-256. |
| `after.json` | Inventaire après déplacement, permettant de vérifier la conservation du contenu. |
| `summary.json` | Résumé des déplacements et des vérifications. |
| `sqlite_backups/` | Copies exactes des bases SQLite avant les adaptations nécessaires des URI d'artefacts MLflow ; les chemins relatifs des bases y sont conservés. |

Les seules modifications de contenu prévues concernent les URI opérationnelles
d'artefacts dans les bases SQLite MLflow. Elles sont ciblées et précédées d'une
sauvegarde ; les autres fichiers sont déplacés en conservant leurs octets.
Cette commande de réorganisation ne lance pas d'entraînement ni d'évaluation.
