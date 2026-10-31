# STEP 33 — Case Review Console / Portfolio Demo UI

Date : 28 octobre 2026.

## Summary

Ajout d'une console de revue de sinistre pour un gestionnaire humain. Le frontend
présente les résultats existants, leurs limites et leurs preuves. FastAPI reste
l'unique API ; les services métier, PostgreSQL, Qdrant et les modèles conservent
leurs responsabilités. Aucun nouveau modèle, entraînement ou benchmark exécuté.

Inspection préalable : routes et schémas FastAPI, dépôts SQLAlchemy/PostgreSQL,
historiques d'évaluations, snapshots vision, evidence registry et runs de l'agent,
contrats RAG/similar claims, fixtures canoniques et observations E2E existantes,
dépendances Python et absence de frontend. Version initiale : **0.1.39**.

## Technology

React 19, TypeScript et Vite dans `frontend/`. Cette application légère suffit
pour une console locale et un build statique ; aucun framework serveur frontend
supplémentaire. Vitest, Testing Library, ESLint et Prettier assurent les contrôles.
Le lockfile npm est conservé. Aucun changement des dépendances Python existantes.

`frontend/` contient uniquement la présentation et la navigation. L'agrégation
se trouve dans `claims.review`, avec un adaptateur PostgreSQL et un adaptateur
de snapshots de démonstration. Le package respecte la limite existante de huit
sous-répertoires/fichiers Python directs imposée par le linter du projet.

## Backend aggregation

Routes ajoutées dans l'application existante `api.main:app` :

| Route GET | Usage |
| --- | --- |
| `/v1/claims/{claim_id}/review` | Vue agrégée du claim et des évaluations disponibles |
| `/v1/review/demo-claims` | Catalogue des snapshots préparés ; liste vide en mode PostgreSQL |
| `/v1/claims/{claim_id}/review/images/{filename}` | Photographie enregistrée et autorisée |

Le service lit les dépôts existants : claim, derniers triage/anomaly/investigation,
profil historique, vision et run agent. Les sources RAG et les sinistres similaires
peuvent être récupérés dans les preuves du run agent sauvegardé. Les résultats ne
sont pas recalculés. Aucun appel YOLO, embedding, Qdrant, Ollama ou MLflow modèle
n'est déclenché par une consultation ou un refresh.

Une panne d'un composant conserve les autres sections. Claim absent : 404.
Persistance non configurée, snapshots absents ou invalides : 503 avec un message
sans détails de connexion. Les chemins d'images doivent rester dans la racine
configurée, appartenir au claim et avoir une extension autorisée.

## Screens implemented

Écran **Claim Review** : identité, statut, type, date/lieu, véhicule, montant et
priorité de revue avec score et version de politique. Navigation latérale,
sélecteur de démo, ouverture par ID et refresh explicite des résultats stockés.

Sections disponibles :

- Documents : disponibilité, fichier, méthode NATIVE/OCR/HYBRID lorsqu'enregistrée,
  provenance et champs structurés dépliables.
- Document Consistency : champs concordants/divergents, dommages supplémentaires
  du devis et lignes non mappées.
- Investigation Signals : code, catégorie, sévérité, explication et faits probants.
- ML Triage : workflow opérationnel, confiance, probabilités des classes et modèle.
- Anomaly Assessment : résultat, score, seuil, percentile et caractéristiques inhabituelles.
- Historical Context : comptes antérieurs et fenêtres 30/90/365 jours, intervalle,
  montants compatibles et nombre de claims antérieurement atypiques.
- Visual Evidence : images, masques polygonaux/boîtes enregistrés, type de dommage,
  pièce, confiances, containment et statut d'association.
- Visual Consistency : corroborations déclaration/devis, dommages visuels additionnels,
  déclaration non confirmée et observations non mappées.
- Similar Claims : identité, date, type, montant et métrique **Similarity**.
- Policy & Procedure Evidence : réponse enregistrée, insuffisance, titres, types
  POLICY/PROCEDURE, sections, chunks, texte source et similarité connue.
- Agent Recommendation : action conseillée, résumé, justification sourcée,
  références, incertitudes, outils et nécessité d'une revue humaine.

Le panneau **Audit / Provenance**, sous **Technical details**, rassemble timestamps,
versions des modèles/politiques/prompts, statuts, documents et IDs de preuves.

## Evidence navigation

Les références de l'agent, des signaux et de l'audit ouvrent un panneau de preuve
et ciblent la section source. Les IDs de chunks et d'observations ciblent leurs
éléments précis lorsque présents. Le panneau gère le focus clavier, Tab et Escape.
Un détail absent est annoncé explicitement et sa référence reste visible.

Vérification sur les quatre snapshots réels/partiels préparés : respectivement
25, 26, 9 et 30 entrées de preuve ; les références structurées des recommandations
sont toutes résolues (4, 5, 0 et 1). Un clic réel dans la console sur la preuve
d'investigation du dossier visuel a affiché ses faits structurés.

Les traces de planification, chemins locaux d'assets et duplications de faits
sérialisés en JSON borné sont retirés de la vue publique. Les résultats métier et
les preuves d'origine restent inchangés dans leurs stockages. Aucun chain-of-thought
n'est affiché.

## Demo mode

La préparation explicite réutilise `data/evaluation/e2e_v2/scenarios.json` et le
constructeur existant de fixtures. Elle produit des PDF contrôlés, puis appelle
les vrais parseurs documentaires, la comparaison et les règles déterministes de
cohérence/complétude. Aucun résultat ML/agent/vision n'est simulé pour remplir l'UI.

| Claim | Démonstration |
| --- | --- |
| `E2E-COHERENT-0001` | Documents cohérents |
| `E2E-QUOTE-0001` | Dommage supplémentaire dans le devis |
| `E2E-MISSING-0001` | Police report manquant, autres documents exploitables |
| `E2E-VISUAL-0001` | Optionnel : observations réelles sauvegardées sur image synthétique autorisée |

La priorisation minimale est marquée **partielle**, avec ses entrées non évaluées.
Les répertoires existants ne sont pas écrasés. Les requêtes lisent les snapshots
préparés et ne relancent pas leur préparation.

Deux préparations réellement réalisées dans cette tâche :

- `artifacts/review_demo_step33_minimal` : trois dossiers sans service externe.
- `artifacts/review_demo_step33_saved` : quatre dossiers, enrichis à partir des
  observations réelles locales existantes. Contrôle de provenance, identité des
  scénarios et SHA256 des images ; rejet des sorties identifiées comme fakes.

L'import a seulement lu les artefacts des étapes antérieures. Aucun scénario E2E
n'a été exécuté. Les snapshots générés et l'image source restent ignorés par Git.

## Partial/unavailable states

Loading et erreurs réseau disposent de vues dédiées. Chaque section distingue
AVAILABLE, UNAVAILABLE, NOT_EXECUTED, FAILED, NO_EVIDENCE et INSUFFICIENT_EVIDENCE.
Une évaluation indisponible ne devient ni zéro ni résultat négatif.

Les anciens artefacts sans méthode d'extraction ou similarité gardent une valeur
inconnue. Un historique incomplet ne reçoit pas les zéros par défaut du modèle.
FAIL/SKIP prévaut sur une ancienne charge utile : pas de faux résultat négatif,
de masques obsolètes ou de réponse RAG prétendument générée. Une photo originale
peut être montrée sans overlay ; une image absente a une raison explicite.

Le dossier partiel a été contrôlé dans le navigateur : signal MISSING_DOCUMENTS,
scope limité aux documents et Anomaly **Not executed**, sans résultat inventé.
Les sinistres similaires futurs, du même jour ou correspondant au claim courant
sont exclus du tableau et de ses preuves structurées.

## Safety semantics

Vérifications implémentées et testées :

- Aucun résultat/conclusion de fraude ; aucune action approve, deny ou payment.
- `Review prioritization — not a fraud probability.`
- `Anomaly detection identifies unusual patterns; it is not a fraud classifier.`
- `Not visually confirmed does not mean the declared damage is absent.`
- **HUMAN REVIEW REQUIRED**, visible même sans recommandation agent.
- POLICY et PROCEDURE explicitement distincts ; une procédure est une consigne
  interne, pas une clause contractuelle.
- Similarity est une métrique de recherche et un contexte comparatif.

Les recommandations historiques conservent leurs incertitudes, y compris les
limites de leur contexte d'exécution. L'interface n'en fait pas une décision.

## Tests

Résultats réellement obtenus :

| Contrôle | Résultat |
| --- | --- |
| Régression backend unitaire hors E2E | 1637 passed, 2 erreurs temporaires de renommage Windows, 53 deselected |
| Relance ciblée des deux fichiers concernés | **36 passed**, 2.64 s ; les deux cas en erreur passent |
| Tous les tests API unitaires, sources finales | **53 passed**, 7.75 s, dont **36 nouveaux tests review/démo** |
| Frontend Vitest / Testing Library | **17 passed** |
| ESLint / Prettier | Réussis |
| TypeScript + Vite production | Réussi, environ 2.27 s |
| nk-python, 28 fichiers concernés | 0 warning, 0 correction de format/import |
| `git diff --check` | Réussi |
| Smoke HTTP réel | Review 200, image 200, claim absent 404, OpenAPI version 0.1.41 |

Les tests API utilisent des dépôts contrôlés et des loaders qui échouent si un
modèle/service d'inférence est initialisé. Ils couvrent dossiers présents/absents,
états partiels/complets, métadonnées, preuves, historique, erreurs, chemins images
et lectures sans recalcul. Les tests frontend couvrent rendu, états, signaux,
preuves, clavier, avertissement humain, sémantique des non-détections et GET-only.

La première tentative en sandbox a été interrompue : création de la paire de
sockets asyncio Windows bloquée avant toute requête. Comme à l'étape 32, les tests
API ont été exécutés hors sandbox. Les deux erreurs WinError 5 du run général
concernaient uniquement des renommages temporaires de tests existants ; leur
relance ciblée a réussi sans modifier ces tests ni les services concernés.
Warnings connus : alias AnyIO/Starlette déprécié et sérialisations enum de fixtures.

**Tests E2E non exécutés faute de demande explicite**, suite `tests/e2e/` toujours
exclue. Les tests unitaires existants du lanceur de benchmark simulent ses
sous-processus ; ils ne déclenchent aucun des dix scénarios. Aucun GPU sollicité.

## Performance

Sur ce poste, lecture HTTP initiale du dossier cohérent : environ **613 ms**,
client PowerShell compris. Après démarrage du serveur final : **75.8, 15.1 et
16.8 ms** pour trois lectures successives. Mesures locales indicatives, pas un
benchmark de charge. Le dossier visuel contient trois observations déjà calculées.

Build frontend : environ **254 kB JS / 78 kB gzip**, **26 kB CSS / 6.6 kB gzip**.
La UI ne charge aucun modèle et ne nécessite aucun batch GPU.

## Screenshots

Captures réelles du navigateur enregistrées et vérifiées sous `docs/images/` :

- `step33-coherent-claim.jpg`
- `step33-investigation-signals.jpg`
- `step33-visual-evidence.jpg`
- `step33-agent-recommendation.jpg`

La photographie visible est l'asset synthétique déjà autorisé du scénario
canonique ; ses observations proviennent de modèles précédemment exécutés.
Aucune photographie tierce CarDD n'est redistribuée. Contrôle responsive à
820 × 1180 : cartes lisibles en une colonne, navigation utilisable, pas de
débordement horizontal du document. La taille navigateur a ensuite été rétablie.

## Limitations

Le schéma PostgreSQL existant conserve les documents et leur disponibilité, mais
pas les snapshots d'extraction/cohérence. Ces champs sont donc inconnus en mode
base et présentés dans la démo préparée lorsqu'ils existent. Aucune nouvelle
table ou migration n'a été ajoutée pour reconstruire artificiellement ces faits.

Les preuves stockées de l'agent ne conservent pas toujours la similarité de
recherche ni une réponse RAG autonome ; ces données restent inconnues. Les vrais
résultats importés dépendent d'artefacts locaux déjà présents et ne sont pas
distribués dans Git. La démo minimale reste reproductible sans ces artefacts.

Le full stack PostgreSQL/Qdrant/Ollama et les nouvelles inférences vision n'ont
pas été relancés. Les services externes et checkpoints restent nécessaires pour
générer de nouveaux résultats correspondants. L'UI offre des lectures ; les
opérations coûteuses existantes restent explicites dans les services/API/scripts.

## Reproduction

Depuis la racine du dépôt, PowerShell, Python 3.12 avec dépendances projet
installées et Node.js 22.12+.

### Backend et démo minimale

Utiliser un nouveau répertoire de sortie :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_demo.py --output-dir artifacts/review_demo_fresh
$env:CLAIMS_REVIEW_DEMO = '1'
$env:CLAIMS_REVIEW_DEMO_DIR = (Join-Path $PWD 'artifacts/review_demo_fresh')
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Frontend

Dans un second terminal, depuis la racine :

```powershell
Set-Location frontend
npm ci
npm run dev
```

Ouvrir `http://127.0.0.1:5173`, puis choisir un dossier. Pour le serveur laissé
disponible lors de la validation, les snapshots sélectionnés sont ceux de
`artifacts/review_demo_step33_saved`.

### Démo avec résultats sauvegardés

Commande réellement utilisée, en lecture des seuls runs déjà présents :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_demo.py --output-dir artifacts/review_demo_step33_saved --observations artifacts/evaluation/e2e_v2/visual_additional_damage_generated_gpu_partiel_damage960 --observations artifacts/evaluation/e2e_v2/serial_damage960_conf035/run_20261027_112250_854978
$env:CLAIMS_REVIEW_DEMO = '1'
$env:CLAIMS_REVIEW_DEMO_DIR = (Join-Path $PWD 'artifacts/review_demo_step33_saved')
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Cette sortie existe désormais ; choisir un autre nom pour répéter la préparation.

### Démo full-stack

Configuration reproductible, documentée sans exécution des services dans cette tâche :

```powershell
Remove-Item Env:CLAIMS_REVIEW_DEMO -ErrorAction SilentlyContinue
Remove-Item Env:CLAIMS_REVIEW_DEMO_DIR -ErrorAction SilentlyContinue
docker compose up -d --wait postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
$env:CLAIMS_LLM_MODEL = 'gemma3:4b'
$env:CLAIMS_REVIEW_ASSET_ROOT = (Join-Path $PWD 'data')
$env:DAMAGE_MODEL_PATH = (Join-Path $PWD 'artifacts/vision/damage_segmentation_v1/train/weights/best.pt')
$env:PART_MODEL_PATH = (Join-Path $PWD 'artifacts/vision/vehicle_parts_v1/train/weights/best.pt')
.\.venv\Scripts\python.exe -m alembic upgrade head
# Une seule fois sur une base de démonstration neuve :
.\.venv\Scripts\python.exe scripts/import_reference_claim.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Adapter les identifiants/ports aux overrides Compose locaux. Ollama et le modèle
choisi doivent être disponibles séparément pour une génération explicite ; les
index Qdrant et checkpoints optionnels suivent les sections existantes du README.
Les nouvelles inférences restent séquentielles, un claim/image à la fois.

### Validation

```powershell
.\.venv\Scripts\python.exe -u -m pytest tests -m unit --ignore=tests/e2e --basetemp=.pytest-tmp-step33-unit-validated -q
.\.venv\Scripts\python.exe -u -m pytest tests/infrastructure/evaluation/test_serial_suite.py tests/intelligence/vision/datasets/coco/conversion/test_conversion.py -m unit --ignore=tests/e2e --basetemp=.pytest-tmp-step33-retry -q
.\.venv\Scripts\python.exe -u -m pytest tests/api -m unit --ignore=tests/e2e --basetemp=artifacts/step33_api_pytest -q --junitxml=artifacts/step33_api_tests.xml
.\.venv\Scripts\nk-python.exe check src/claims/review src/api/main.py src/api/review.py src/api/version.py tests/api/review scripts/prepare_review_demo.py
Set-Location frontend
npm test
npm run lint
npm run format:check
npm run build
```

## Versioning

Source unique : `pyproject.toml`, version statique Python. Le frontend privé
n'introduit aucune version applicative indépendante. FastAPI lit la version du
pyproject en checkout et les métadonnées du paquet installé en distribution.

- Backend/démo validés : **0.1.39 → 0.1.40**, commit `652dff6`.
- Console/frontend, captures et documentation validés : **0.1.40 → 0.1.41**.

Un PATCH par commit, sans changement MAJOR/MINOR, tag, release, push ou déploiement.

## Update — 29 octobre 2026 : backend indisponible

La console distingue désormais une connexion FastAPI impossible, un claim absent
(404), un service/configuration indisponible (503) et une erreur serveur réelle
(500). Le proxy Vite, en développement et en preview, émet un 502 JSON explicite
`BACKEND_UNREACHABLE` en cas d'erreur de transport ; les réponses HTTP de FastAPI
ne sont pas réécrites. Aucun appel au endpoint `/health` qui charge des modèles.

L'écran **Backend unavailable** affiche les commandes PowerShell de démarrage
depuis la racine du dépôt et précise que PostgreSQL/Qdrant ne sont pas nécessaires
pour les snapshots de démo. Docker ne lance pas FastAPI. Le bouton **Retry
connection** recharge le catalogue et le claim, même après un premier démarrage
de l'UI sans backend ni claim sélectionné. Les annulations de requêtes ne sont
pas présentées comme des pannes.

Validation : **32 tests frontend réussis**, ESLint, Prettier et build TypeScript/
Vite réussis. Le backend de démonstration a été arrêté de manière contrôlée :
le proxy réel a renvoyé 502 avec le code attendu, l'écran a montré les instructions,
puis le redémarrage et un clic sur Retry ont restauré les quatre dossiers et ouvert
`E2E-COHERENT-0001`. La réponse 404 d'un claim absent reste inchangée. Capture :
`docs/images/step33-backend-unavailable.jpg`. README complété.

Version : **0.1.41 → 0.1.42**, source unique `pyproject.toml`. Aucune dépendance
modifiée, aucun test E2E ni inférence GPU exécutés. Les contrôles unitaires frontend
et le smoke manuel ci-dessus ne lancent aucun scénario du benchmark.

## Update — 29 octobre 2026 : mode sombre

La console utilise désormais un thème sombre par défaut, avec un bouton clair/
sombre accessible dans l'en-tête. Le choix est conservé dans ce navigateur et
appliqué avant le chargement de React afin d'éviter un flash blanc. Les accès au
stockage navigateur sont protégés ; le changement de thème reste utilisable si
ce stockage est indisponible. Le mode clair reste disponible sur choix explicite.

La palette utilise des variables sémantiques communes aux deux thèmes : page,
panneaux, navigation, champs, badges, tableaux, alertes, commandes, focus et
panneau de preuves. Les photos et les masques SVG ne subissent aucun filtre ni
inversion. Vérifications dans le navigateur : changement clair/sombre et
persistance après rechargement, drawer sombre, photo chargée sans filtre. Les
contrastes de textes contrôlés sur le rendu réel sont compris entre 7.25:1 et
8.69:1 pour les labels, navigation, badges et avertissement humain mesurés.

Validation : **41 tests frontend réussis**, dont neuf tests de préférence,
clavier, stockage indisponible et initialisation avant React ; ESLint, Prettier
et build TypeScript/Vite réussis. Le runtime Node local émet un avertissement
expérimental Web Storage dans un worker ; les 41 tests passent, et le stockage
du navigateur réel a été vérifié. Capture : `docs/images/step33-dark-mode.jpg`.

Version **0.1.42 → 0.1.43** dans `pyproject.toml`. Aucune dépendance modifiée.
Tests E2E non exécutés faute de demande explicite ; aucune inférence GPU.

## Update — 31 octobre 2026 : parcours autonome et lancement commun

L'accueil propose les cas réellement présents dans le catalogue, avec des titres
et descriptions courts. Aucun identifiant n'est requis pour les explorer ; le
premier cas n'est plus ouvert automatiquement. Les liens `?claim=...` continuent
d'ouvrir directement le dossier demandé. La saisie manuelle d'un ID reste dans
une zone repliée, ouverte lorsque le catalogue est vide.

Chaque dossier commence sur **Overview** : prochaine action enregistrée,
trois signaux au maximum, documents/photos disponibles et accès aux incertitudes.
**Evidence** ouvre un seul groupe de preuves à la fois ; **Technical details**
conserve les métadonnées complètes, scores, versions et audit. Les liens de
preuves révèlent leur groupe avant d'ouvrir le drawer, avec restitution du focus.
L'ouverture et le retour au catalogue placent le focus sur le titre et ramènent
en haut de page. Les flèches, Home et End permettent de parcourir les onglets.

Les résultats restent ceux de l'API : aucun score recalculé, aucune recommandation
inventée. Une recommandation absente reste absente ; le périmètre d'une évaluation
partielle reste lisible intégralement. L'avertissement humain et la distinction
entre priorité et probabilité de fraude restent visibles. Le mode sombre par
défaut et le choix de thème persistant sont conservés.

Captures du navigateur réel à 1265 × 713 :
`docs/images/step33-guided-home.jpg` et
`docs/images/step33-guided-overview.jpg`. La synthèse du cas visuel montre le
signal et la photo entière dans la première vue. Vérifications manuelles :
catalogue, synthèse, recommandation complète, drawer de preuve, sortie Escape,
accès direct aux photos, dossier incomplet, retour aux cas et thème clair/sombre.

Le lanceur stdlib `scripts/start_review_console.py` démarre FastAPI en mode
snapshots préparés puis Vite dans le même terminal. Les prérequis, le manifeste
et les ports sont vérifiés ; un catalogue HTTP 200 et la page Vite confirment le
démarrage. Aucun appel à `/health`, installation, préparation, Docker ou inférence
automatique. Les erreurs sont explicites et les processus du lanceur sont
nettoyés en cas d'échec ou de Ctrl+C, avec arrêt des descendants Windows si requis.

Commande validée pour les quatre dossiers déjà préparés dans ce workspace :

```powershell
.\.venv\Scripts\python.exe scripts/start_review_console.py --demo-dir artifacts/review_demo_step33_saved
```

Sans `--demo-dir`, le lanceur utilise `CLAIMS_REVIEW_DEMO_DIR` si défini, sinon
`artifacts/review_demo`. README complété pour la préparation initiale, le lancement
commun, les cas stockés et le mode PostgreSQL distinct.

Validation : **48 tests frontend** et **20 tests unitaires du lanceur** réussis ;
ESLint, Prettier, build TypeScript/Vite et `nk-python check` des neuf fichiers
Python du lanceur/tests réussis sans avertissement de style. Le premier pytest
a rencontré un refus de création de répertoire temporaire du sandbox ; la
relance avec `--basetemp artifacts/pytest_launcher_20261031_unit` passe entièrement.
Le runtime Node conserve son avertissement expérimental Web Storage dans un
worker ; les tests de thème et la vérification dans le navigateur passent.

Smoke réel du lanceur : port 8000 occupé refusé sans arrêter le serveur existant,
démarrage API/Vite puis lectures catalogue/dossier/photo HTTP 200, et Ctrl+C.
Après arrêt : aucun listener 8000/5173 ni processus enfant du lanceur restant.
Tests E2E non exécutés faute de demande explicite ; aucune inférence GPU.

Version **0.1.43 → 0.1.44**, source unique `pyproject.toml`. Aucune dépendance
modifiée, aucun tag, push ou déploiement.

## Update — 31 octobre 2026 : cas photo par défaut et schéma explicatif

L'accueil ouvre automatiquement `E2E-VISUAL-0001` lorsqu'il figure dans le
catalogue, sinon le premier cas disponible. Un lien `?claim=...` reste
prioritaire, y compris pour un claim hors catalogue ou introuvable ; aucune
erreur n'est masquée par un repli silencieux. Un catalogue vide conserve la
saisie d'identifiant. Retry recharge le catalogue et ouvre le défaut lorsque
aucun dossier n'avait encore été demandé.

La première vue présente un schéma HTML/CSS compact à flèches : **PDFs & photos
→ Document AI + vision → LLM agent + policy evidence → review guidance**.
Il décrit le parcours conceptuel ; la mention des résultats précédemment
générés reste visible. Le LLM exploite les faits structurés et les preuves
enregistrées, et n'analyse pas directement les pixels dans cette démo.

Une grande photo du dossier apparaît au milieu de la page, sous ce schéma,
avec la prochaine action et les signaux à droite. L'image entière est conservée
en `object-fit: contain`, sans filtre ni modification des observations. La
légende **AI-generated demo image** dépend de `provenance.origin`, jamais du nom
de fichier. Les absences ou erreurs de chargement affichent leur raison ; aucune
photo d'un autre claim n'est substituée. Les documents et les incertitudes
restent accessibles dans une ligne secondaire compacte.

**Explore other cases** ouvre facultativement le catalogue ; **Try another
case** est replié dans la navigation. Les onglets Evidence et Technical details,
les liens de sources, le focus de navigation, les raisons d'indisponibilité,
le périmètre partiel et l'avertissement humain sont préservés. Le choix des
autres cas n'est plus une étape obligatoire avant de voir le résultat.

Capture : `docs/images/step33-default-photo-demo.jpg`. Vérification réelle de
l'accueil sans paramètre : cas visuel sélectionné, photo chargée à 250px de haut,
largeur du contenu égale à celle du viewport (1265px), sans débordement
horizontal. Schéma, photo entière et résultats principaux visibles ensemble.
Accès aux preuves photo, choix facultatif d'un autre cas et cas sans photo
également contrôlés. Les serveurs déjà actifs ont été réutilisés.

Validation : **57 tests frontend réussis** (dont défaut visuel, priorité URL,
repli, retries, navigation, provenance explicite et erreur de photographie),
ESLint, Prettier et build TypeScript/Vite réussis. Aucun changement backend,
aucune dépendance ni nouvelle inférence. Tests E2E non exécutés faute de demande
explicite. README et capture principale mis à jour.

Version **0.1.44 → 0.1.45**, source unique `pyproject.toml`. Aucun tag, push ou
déploiement.

## Update — 31 octobre 2026 : illustration générée du workflow

Le schéma HTML à quatre cases est remplacé par une image produite avec l'outil
intégré **image_gen**, sans CLI ni appel à une API avec clé utilisateur.
Le PNG original de 2172 × 724 pixels est conservé dans
`frontend/src/assets/review-workflow-v1.png` ; le prompt exact et la provenance
figurent dans `frontend/src/assets/review-workflow-v1.prompt.md`.

L'illustration représente les PDF/photos, l'analyse Document AI/vision, le LLM
avec les preuves de police d'assurance, puis la revue humaine. Elle est placée
après le dossier pour garder la photo du véhicule et les résultats principaux
visibles dès l'arrivée. Aucun recadrage ni filtre n'est appliqué. Un texte
alternatif complet décrit le parcours et un lien permet d'ouvrir le PNG en grand.
La légende rappelle qu'il s'agit d'un workflow illustré, de résultats enregistrés
et d'une décision humaine. Cette image n'est pas une preuve du dossier.

L'import du PNG passe par Vite et le build inclut l'asset dans `dist/assets`.
Le premier build a révélé l'absence des déclarations de types pour les assets ;
l'ajout conventionnel de `vite/client` aux types TypeScript corrige l'import.
Aucune dépendance ajoutée ou mise à jour.

Validation : **57 tests frontend réussis**, ESLint, Prettier et build
TypeScript/Vite réussis après correction. Vérification dans le navigateur :
photo entière dans la première vue, illustration chargée avec ses quatre libellés
et flèches, lien vers le PNG présent et description accessible. Captures :
`docs/images/step33-generated-workflow-overview.jpg` et
`docs/images/step33-generated-workflow.jpg`. README mis à jour.

Aucun modèle d'analyse des claims ni benchmark exécuté. Tests E2E non exécutés
faute de demande explicite. Version **0.1.45 → 0.1.46**, source unique
`pyproject.toml`. Aucun tag, push ou déploiement.

## Update — schéma compact et légende de démonstration

Deux illustrations image_gen ont été préparées :
`frontend/src/assets/review-workflow-en-v2.png` et
`frontend/src/assets/review-workflow-fr-v2.png`, avec leurs prompts conservés.
Le premier libellé devient **Claim file (PDFs & photos)** / **Dossier de sinistre
(PDF et photos)**. Les compositions concentrent tous les objets et libellés dans
une bande centrale ; la fenêtre CSS ne retire que les marges sombres du canevas.
Le schéma est remonté avant les onglets de la console, avec ouverture du PNG
complet accessible au clavier. La légende du véhicule devient **Demo image**.

Étape visuelle validée par build TypeScript/Vite, lint ciblé et contrôle dans le
navigateur : schéma avant Overview, voiture endommagée visible dès la première
vue, flèches et libellés lisibles. La sélection de la variante FR fait partie de
l'étape de localisation suivante. Aucun modèle du projet ni benchmark lancé.
E2E non exécutés faute de demande explicite.

Version de cette étape **0.1.46 → 0.1.47**, source `pyproject.toml`.

## Update — 31 octobre 2026 : console français / anglais

La console démarre en français et propose les boutons **FR / EN**, nommés
**Français / English** pour l'accessibilité. Le choix est mémorisé séparément du
thème sombre. Un stockage refusé ne bloque pas la sélection. Le titre de page,
`document.lang`, les libellés, erreurs, cas de démonstration et formats de dates,
montants, nombres et pourcentages suivent la langue choisie sans recharger le
dossier ni déclencher de nouvelle analyse.

Sources uniques de traduction : `frontend/locales/fr.toml` et
`frontend/locales/en.toml`, **485 clés identiques** avec placeholders validés.
Un plugin Vite/Vitest utilise le Python existant de `.venv` et `tomllib` pour
produire un module virtuel au build/développement/tests. Les dictionnaires sont
embarqués dans le JavaScript de production ; aucun parseur TOML, service de
traduction ou nouvelle dépendance ne s'exécute dans le navigateur. Les fragments
temporaires de travail ont été fusionnés puis retirés. La modification des TOML
invalide le module et recharge la page de développement.

Les quatre cas préparés disposent de **85 traductions narratives contrôlées** :
raisons d'indisponibilité, synthèses, justifications, incertitudes, signaux et
extraits contractuels. UUID, références, nombres, exclusions et montants ont été
conservés. Les textes originaux de recommandation et des extraits restent dans
des zones repliées, avec leur langue d'origine indiquée pour les lecteurs
d'écran. Les récits inconnus restent dans leur texte source ; de nouveaux jeux
de données peuvent demander des entrées de catalogue supplémentaires. Aucun
payload API ou snapshot n'a été modifié.

L'illustration sélectionne maintenant la version FR ou EN selon la langue. Elle
reste avant les onglets Synthèse / Overview et comporte un texte alternatif
localisé ainsi qu'un lien vers le PNG complet. La légende de la voiture devient
**Image de démonstration / Demo image**, sans modification de sa provenance
enregistrée. Les images ont été produites avec l'outil image_gen intégré ; les
deux PNG et prompts sont conservés sous `frontend/src/assets/`.

Captures réelles : `docs/images/step33-bilingual-fr.jpg` et
`docs/images/step33-bilingual-en.jpg`. Vérifications dans le navigateur : première
vue FR, changement EN, persistance après rechargement, retour FR, recommandation
traduite, ouverture d'une preuve avec son identifiant exact, titres et libellés
localisés. README mis à jour. Le backend déjà actif a été réutilisé.

Validation finale : **65 tests frontend** et **8 tests unitaires du compilateur
TOML** réussis ; ESLint, Prettier, TypeScript/Vite et `nk-python check` des trois
fichiers Python concernés passent. Les tests couvrent le défaut FR, les choix
persistants, le stockage indisponible, les formats, les récits inconnus, les clés
de prototype JavaScript, les références intactes et la bascule sans GET
supplémentaire. Le runtime Node conserve son avertissement expérimental Web
Storage dans les workers ; les tests et le navigateur réel passent.

Tests E2E non exécutés faute de demande explicite. Aucun benchmark ni modèle
d'analyse de sinistre exécuté. Version de la localisation **0.1.47 → 0.1.48** ;
version totale de la tâche **0.1.46 → 0.1.48**, source unique `pyproject.toml`.
Aucun changement de dépendance, tag, push ou déploiement.

## Update — 31 octobre 2026 : rendre le résultat et le parcours explicites

Le cas par défaut s'intitule maintenant **Collision à l’avant du véhicule /
Front collision**, dans le titre comme dans le sélecteur. Le schéma précède
désormais ce titre. Les onglets numérotés **Résultats du système → Pièces et
sources / System results → Evidence** et la courte indication dans la barre
latérale rendent le parcours visible.

La carte à côté de la photo montre directement **Ce que le système a relevé** :
l'explication complète du premier signal enregistré, sans dupliquer ce signal
dans une seconde carte. Pour le cas de collision avant, il s'agit du dommage au
capot absent des pièces déclarées endommagées. La prochaine action reste celle
de la recommandation sauvegardée. Le bouton principal **Comprendre la
recommandation / Understand the recommendation** ouvre et focalise la
recommandation complète. **Vérifier les constats sur la photo / Check the photo
evidence** ouvre et focalise les observations visuelles et leurs annotations.

En l'absence de signal, la synthèse d'une recommandation disponible et terminée
sert de résultat. Les autres signaux, évaluations partielles, indisponibilités et
incertitudes restent accessibles. Aucun constat n'est inventé et aucun payload,
snapshot ou modèle n'a été modifié. Les six nouvelles clés des catalogues FR/EN
portent leur total à **491**, avec réutilisation de la clé de collision avant.

Validation : **71 tests frontend** passent, dont six cas supplémentaires pour le
résultat initial, sa synthèse de secours, le focus des deux boutons, l'ordre du
schéma et les recommandations incomplètes. ESLint, Prettier et le build
TypeScript/Vite passent. Vérification manuelle FR/EN : résultat visible à côté de
la photo, action principale visible à l'arrivée, accès à la justification et aux
observations visuelles. Captures : `docs/images/step33-guided-results-fr.jpg` et
`docs/images/step33-guided-results-en.jpg`. Le backend local existant a été
réutilisé ; aucune nouvelle analyse de sinistre n'a été lancée.

Les tests E2E n'ont pas été exécutés faute de demande explicite. Version
**0.1.48 → 0.1.49**, source unique `pyproject.toml`. Aucun changement de
dépendance, push, tag ou déploiement.

## Update — 31 octobre 2026 : exports autonomes dans analysis_results

La commande `scripts/export_analysis_results.py` exporte des revues déjà
calculées depuis un répertoire préparé, un fichier `ClaimReview` ou les résultats
stockés en PostgreSQL d'un dossier explicitement sélectionné. Elle ne lance ni
modèle, ni agent, ni extraction, ni benchmark. La racine lisible
`analysis_results/` contient un index HTML et un manifeste de collection ; chaque
analyse a son sous-dossier `CLAIM-ID_ANALYSIS-ID/` avec `rapport.html`,
`resultats.json`, `manifest.json`, `documents/` et `images/` selon les pièces
présentes. Les fichiers générés sont ignorés par Git ; le README de cette racine
est versionné.

Les PDF et photos sont copiés physiquement, sans lien symbolique ni référence
vers un fichier externe. Les sources sont explicitement sélectionnées et les
faits d'un `claim.json` fourni doivent correspondre à la revue. Aucune pièce
absente n'est recréée. Les noms dangereux, sorties de répertoire et liens
symboliques sont refusés. Les empreintes SHA256 et tailles des copies sont
conservées ; une empreinte d'image d'origine déjà connue doit correspondre.
Un identifiant de contenu fixe chaque export et un dossier existant n'est
jamais écrasé. Les identifiants, dates et versions des résultats d'origine
restent distincts : l'export ne fabrique pas une exécution globale unique.

Le rapport français sombre contient le résultat, la recommandation, les PDF,
les photos originales, les annotations SVG enregistrées et la provenance. Il
embarque son CSS et ne contient ni script, ni CDN, ni requête réseau. Tous ses
liens sont relatifs et les textes sont échappés. Les traductions narratives
existantes sont incorporées à la génération, sans service de traduction.

Validation de cette étape : **41 tests unitaires réussis, 2 ignorés** parce que
Windows ne permet pas la création effective de liens symboliques dans cet
environnement. Les contrôles de style Python passent. Une copie déplacée reste
lisible après suppression des originaux dans les tests. Un essai local avec les
quatre cas existants a copié **15 PDF et une photo**, en conservant le document
non fourni. Le HTML réel a été vérifié par parsing : ses ressources locales
existent dans l'export et aucune ressource externe n'est nécessaire. Le
navigateur de validation bloque le protocole `file:` ; aucune prévisualisation
visuelle hors ligne ni contournement de cette restriction n'a été effectué.

Tests E2E non exécutés faute de demande explicite. Version de l'export
**0.1.49 → 0.1.50**, source unique `pyproject.toml`. Pas de nouvelle dépendance,
de changement de modèle, de push ou de déploiement.
