# Refactoring des diagnostics Python

## Référence et résultat

Le point de départ est le commit `21688bf`, avec un arbre Git propre.
L'application utilise Python >= 3.11 et setuptools (`setuptools.build_meta`),
avec une version statique unique dans `pyproject.toml`.
Les validations ont été exécutées avec Python 3.14.7 dans `.venv`.

| Contrôle | Avant | Après |
| --- | --- | --- |
| `nk-python check src tests` | 107 warnings, 1 diagnostic d'import, sortie 1 | 0 warning, 0 diagnostic d'import, 0 fichier à reformater, sortie 0 |
| Collecte pytest | 860 tests | 862 tests |
| Tests unitaires | 813 réussis, 47 désélectionnés | 815 réussis, 47 désélectionnés |
| Version applicative | 0.1.7 | 0.1.8 |

La comparaison des identités de tests, paramètres compris, conserve les 860 cas
initiaux. Les deux ajouts contrôlent les chemins par défaut après déplacement
et les imports publics sans chargement des frameworks vision.

Les premiers essais dans le bac à sable Windows rencontraient des erreurs
d'accès aux fichiers temporaires et un blocage de la création des sockets
internes d'asyncio. La référence complète, rejouée hors de ces restrictions
dans une copie temporaire du commit initial, passe : aucun échec fonctionnel
préexistant n'a été constaté. Les validations suivantes utilisent les permissions
nécessaires, sans modification de l'environnement Python.

## Responsabilités extraites

| Périmètre public conservé | Responsabilités internes |
| --- | --- |
| `agent.models` | Contrats de planification, preuves, recommandations, traces et résultats terminaux |
| `agent.graph` | Assemblage du graphe, planification, invocation des outils, enregistrement des preuves, finalisation et validation |
| `agent.tools` | Dispatch autorisé, lectures des évaluations stockées, perception stockée, recherche de précédents, recherche documentaire et bornage des données |
| `agent.service` et `agent.planner` | Cycle de vie et audit, contexte borné, prompts et génération structurée |
| `vision.dataset` | Modèles communs, accès aux fichiers, découverte COCO, références, géométrie, conversion, préparation atomique et inventaire des données préparées |
| `vision.cardd_hf` | Contrats de source, schéma et annotations, contours et trous des masques, normalisation, inspection et chargement différé |
| `vision.cardd_hf_split` | Identité et empreintes, regroupement des doublons, allocation exacte, équilibrage multilabel, validation et persistance des manifestes gelés |
| `vision.cardd_hf_preparation` | Contrats de provenance, empreintes, contexte de conversion, géométrie, fichiers préparés et publication atomique |
| `vision.carparts_dataset` | Schéma, inspection des images et labels, préparation officielle et provenance |
| `vision.evaluation` | Réutilisation des abstractions existantes de métriques, checkpoint et requête ; préparation, inférence, dispositif réellement observé et rapports |
| `vision.error_analysis` et `vision.fusion` | Appariement des boîtes/masques, sélection bornée des exemples, rapports et relecture des associations persistées |

Les contextes de conversion portent les mappings immuables et les sorties
accumulées d'une préparation. L'allocateur porte les capacités et déficits de
classes. Ils évitent de propager un ensemble incohérent de paramètres entre
étapes. Les `__init__.py` restent des façades d'import légères.

Les dépendances des adaptateurs de l'agent pointent directement vers leurs
repositories. Les monkeypatches ciblent le module propriétaire de la référence.
L'analyse statique des imports applicatifs au niveau des modules ne détecte
aucun cycle.

## Organisation des tests et contrats vérifiés

Les suites sont regroupées par domaine, puis par contrat : exécution et sécurité
de l'agent, chargement/extraction des déclarations, mappings et transactions,
historique, connaissances, retrieval, ML, datasets, perception et fusion.
Les doubles communs et fixtures Qdrant sont dans `tests/support`. Les fixtures
propres à un domaine ont des modules dédiés auprès des suites concernées.
Les prérequis, connexions, migrations et schémas PostgreSQL sont séparés.

Les marqueurs et la classification unit/intégration sont conservés. Le workflow
CI garde sa commande `python -m pytest tests/e2e -m unit`. Les chemins Alembic
et CLI des tests déplacés utilisent la racine du projet. Les chemins par défaut
restent sous les mêmes répertoires `data/` et `artifacts/` du dépôt.

Les observations de requêtes PostgreSQL utilisent l'événement SQLAlchemy
`before_execute`, dont la signature publique contient cinq paramètres. Les
assertions de nombre et de contenu des requêtes sont conservées ; leur
validation sur serveur reste soumise à la disponibilité de PostgreSQL.

Aucun seuil, exclusion, suppression de diagnostic, dépendance ou test désactivé
n'a été ajouté. Aucun tag, release, push ou déploiement n'a été effectué.

## Validations

| Commande | Résultat |
| --- | --- |
| `.\.venv\Scripts\nk-python.exe check src tests` | Succès, zéro diagnostic |
| `.\.venv\Scripts\python.exe -m pytest --collect-only -q` | 862 tests, aucune perte parmi les 860 identités initiales |
| `.\.venv\Scripts\python.exe -m pytest -m unit` | 815 réussis, 47 désélectionnés |
| `.\.venv\Scripts\python.exe -m pytest tests/e2e -m unit` | 49 réussis |
| `.\.venv\Scripts\python.exe -m pytest -m vision -q` | 1 réussi avec le checkpoint local, aucun téléchargement |
| `.\.venv\Scripts\python.exe -m pytest -m 'postgres or qdrant or rag' -q` | Bloqué par le contrôle des prérequis : moteur Docker indisponible |
| `python -m pip wheel . --no-deps --no-build-isolation ...` | Construction locale réussie |
| Installation de la wheel avec `--no-deps --no-index --target ...` | Succès, 12 packages publics importés depuis la wheel installée |
| `git diff --check 21688bf` | Succès sur l'ensemble de la tâche |

Les 46 cas dépendant de PostgreSQL/Qdrant/RAG n'ont pas pu être exécutés : le
canal `dockerDesktopLinuxEngine` n'existe pas, donc les services de test dédiés
ne sont pas disponibles. Aucun service n'a été démarré. L'intégration vision,
indépendante de ces services, a été exécutée.

## Commits et version

- `28658d5` : contrats, graphe et adaptateurs de l'agent.
- `56a3b78` : datasets, provenance, conversion et évaluation vision.
- `697a157` : suites par domaine, fixtures partagées, imports et chemins.
- Le dernier commit ajoute ce bilan et l'unique incrément PATCH de la tâche.

La version passe une seule fois de **0.1.7 à 0.1.8**, dans `pyproject.toml`,
après réussite des validations disponibles. Les versions des dépendances restent
inchangées.
