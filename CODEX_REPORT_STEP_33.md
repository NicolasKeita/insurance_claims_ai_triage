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
