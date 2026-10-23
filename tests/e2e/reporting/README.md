# Comprendre les rapports et la fiabilité de l'évaluation

[Retour au guide des tests E2E](../README.md).

Ces tests vérifient que les rapports permettent de juger honnêtement le système :
comprendre un échec, repérer une dégradation et reconnaître ce qui n'a pas pu être
testé. Ils ne cherchent pas eux-mêmes à détecter une fraude dans un dossier.

Ce catalogue décrit **10 fonctions de test**, soit **15 cas en développant les
paramétrages** : 8 fonctions dans ce dossier et 2 dans
[`test_real_mode_contracts.py`](../test_real_mode_contracts.py).
Les descriptions expliquent les attentes du code ; elles ne constituent pas un
résultat d'exécution. Les tests end-to-end ne doivent être exécutés que sur
demande explicite, y compris les tests hors ligne décrits ici ; voir les
[instructions du dépôt](../../../AGENTS.md).

Dans les explications ci-dessous, « réussi » signifie `PASS`, « échoué » signifie
`FAIL` et « non vérifié complètement » signifie `SKIP`. Un dossier en `SKIP`
peut avoir été partiellement traité : certains contrôles restent impossibles.

## Comprendre pourquoi un test a échoué — [`test_artifacts.py`](test_artifacts.py)

### `test_failure_artifacts_include_expected_actual_component_and_trace`

**Pourquoi ce test est utile :** quand un contrôle échoue, il faut pouvoir
comprendre l'écart avant de corriger le système. Un simple message « échec » ne
suffit pas.

**Situation :** le contrôle de l'historique des sinistres attend de retrouver
3 déclarations récentes, mais le résultat en contient 4. Le test prépare
volontairement cet écart et demande la création du rapport.

**Résultat attendu :** le rapport conserve les deux nombres, indique que l'écart
concerne l'historique et permet de retrouver les éléments qui l'expliquent. Les
fichiers de synthèse et de détail doivent aussi être créés.

**Erreur à éviter :** perdre les preuves de l'échec et obliger le lecteur à
deviner ce qui s'est mal passé. Ce test porte sur la lisibilité du diagnostic ;
il ne vérifie pas la détection d'un risque de fraude lié à ces sinistres.

<details>
<summary>Détails techniques et limites</summary>

Le rapport déterministe contient une assertion en échec
`fact:history.claims_last_30_days`, avec `expected=3`, `actual=4`,
`component="history"` et `source="traces/history.json#history"`.
Après `write_report()`, ces quatre valeurs doivent être conservées dans la
première assertion de `failures/history.json`. Les fichiers `summary.json`,
`scenario_results.json`, `scenario_results.csv`, `timings.csv` et
`traces/history.json` doivent exister. Leur contenu détaillé n'est pas comparé
ici. L'échec est construit artificiellement.

Code concerné : [écriture des artefacts](../../../src/evaluation/reporting/artifacts.py).

</details>

## Repérer une dégradation du système — [`test_baseline.py`](test_baseline.py)

Une référence, appelée *baseline* dans le code, est une photographie des
résultats précédemment acceptés. Elle permet de voir si une nouvelle évaluation
perd une qualité déjà acquise.

### `test_regression_comparison_detects_stable_behavior_regressions`

**Pourquoi ce test est utile :** une mise à jour ne doit pas faire disparaître
un contrôle utile sans que le rapport le signale.

**Situation :** une évaluation de référence réussit. Dans une nouvelle version
de ses résultats, on fait échouer volontairement l'une des cinq garanties
suivantes :

| Garantie perdue | Ce que cela représenterait pour le lecteur du dossier |
| --- | --- |
| Signal obligatoire sur un dommage ajouté au devis | Un point qui devait attirer l'attention n'est plus signalé. |
| Absence d'un signal visuel interdit | Le système signale un dommage supplémentaire qu'il ne devait pas signaler dans ce cas. |
| Sources documentaires attendues | La réponse ne s'appuie plus sur les documents attendus. |
| Citations valides | La réponse ne renvoie plus correctement aux passages qui la justifient. |
| Citations des preuves par l'agent | La conclusion de l'agent ne cite plus correctement ses preuves. |

**Résultat attendu :** pour chacune de ces **5 variantes**, la comparaison doit
signaler à la fois la dégradation du scénario et la garantie précise qui a été
perdue.

**Erreur à éviter :** présenter une nouvelle version comme équivalente alors
qu'elle produit une réponse moins fiable ou moins vérifiable.

<details>
<summary>Détails techniques et limites</summary>

Le test copie le rapport de `baseline_report()`, passe une assertion et son
scénario de `PASS` à `FAIL`, puis appelle `compare_baseline()`. Les régressions
doivent contenir `scenario_regression` et le nom exact de l'assertion modifiée :
`required_signal:ADDITIONAL_QUOTE_DAMAGE`,
`forbidden_signal:VISUAL_ADDITIONAL_DAMAGE`, `rag:expected_sources`,
`rag:citation_validity` ou `agent:evidence_citations`.

Les signaux et les citations ne sont pas recalculés. Le test vérifie que la
comparaison reconnaît une perte déclarée dans les résultats, pas que l'IA
détecte réellement chacun de ces problèmes.

</details>

### `test_changing_expectations_under_the_same_version_is_a_regression`

**Pourquoi ce test est utile :** on ne doit pas pouvoir rendre un test vert en
changeant discrètement ce qu'on attend de lui.

**Situation :** la référence attend 3 sinistres récents et en trouve 3. On
remplace ensuite les deux valeurs par 2, sans changer la version de
l'évaluation. Le nouveau résultat paraît encore réussi si l'on regarde
seulement son statut.

**Résultat attendu :** la comparaison doit signaler que la règle de vérification
a changé : on attendait 3, on attend désormais 2.

**Erreur à éviter :** masquer une différence de résultat en déplaçant l'objectif
à atteindre, tout en laissant croire que l'évaluation est restée la même.

<details>
<summary>Détails techniques</summary>

Une assertion `fact:history.claims_last_30_days` est ajoutée avec
`expected=actual=3`. Dans une copie du rapport, ces deux valeurs deviennent 2.
`compare_baseline()` doit retourner une régression
`benchmark_expectation_changed` qui identifie cette assertion et conserve
`expected=3`, `actual=2` dans son diagnostic.

</details>

### `test_partial_real_or_deterministic_runs_cannot_create_real_baseline`

**Pourquoi ce test est utile :** la référence utilisée pour juger les prochaines
versions doit provenir d'une évaluation réelle, complète et réussie.

**Situation :** on essaie d'enregistrer comme référence trois résultats qui ne
remplissent pas ces conditions : une évaluation réelle incomplète, une
évaluation reposant sur des réponses simulées, puis une évaluation réelle avec
un scénario en échec.

**Résultat attendu :** les trois tentatives sont refusées et aucun fichier de
référence n'est créé.

**Erreur à éviter :** prendre un résultat incomplet, fictif ou défaillant comme
preuve du bon fonctionnement du système, puis mesurer les versions suivantes
par rapport à cette mauvaise référence.

<details>
<summary>Détails techniques</summary>

Une seule fonction vérifie successivement trois appels à `create_baseline()` :
rapport `real/PARTIAL`, rapport `deterministic/COMPLETE`, puis rapport
`real/COMPLETE` dont le scénario est `FAIL`. Chaque appel doit renvoyer exactement
`False` et aucun `baseline_summary.json` ne doit être créé. Ces trois tentatives
ne constituent pas trois cas paramétrés supplémentaires.

</details>

### `test_baseline_ignores_latency_and_records_versions`

**Pourquoi ce test est utile :** pour comparer le comportement de deux versions,
il faut savoir quels modèles ont été utilisés et vérifier que l'on compare le
même exercice.

**Situation :** on enregistre une référence en conservant l'identité du modèle.
On change ensuite uniquement la durée du traitement, puis on tente une
comparaison avec une autre version de l'évaluation.

**Résultat attendu :** l'identité du modèle est conservée. Une différence de
durée seule n'est pas signalée comme une dégradation du comportement. En
revanche, comparer deux versions différentes de l'évaluation doit être refusé.

**Erreur à éviter :** confondre une variation de vitesse avec une différence de
résultat, ou comparer comme équivalents deux exercices dont les règles ont
changé. La rapidité n'est pas évaluée par ce test.

<details>
<summary>Détails techniques et limites</summary>

La référence est créée avec `timings.total=1.25` et doit conserver
`scenarios.golden.model_versions={"anomaly": "model-v1"}`. Le scénario
sauvegardé ne doit pas contenir de clé `timings`. Passer la durée à 999 ne doit
produire aucune régression. Remplacer la version sauvegardée du benchmark par
`e2e_claims_future` doit lever `ValueError` avec `version mismatch` dans le
message. Seule la version du modèle `anomaly` est explicitement vérifiée ; aucun
modèle réel n'est exécuté.

</details>

Code concerné : [instantanés et comparaison des baselines](../../../src/evaluation/reporting/baseline.py).

## Présenter des chiffres honnêtes — [`test_metrics.py`](test_metrics.py)

### `test_skips_never_inflate_pass_rate_and_no_executions_have_no_rate`

**Pourquoi ce test est utile :** un système qu'on n'a pas pu tester ne doit pas
être présenté comme fiable grâce à un taux de réussite trompeur.

**Situation :** sur trois scénarios, un réussit, un échoue et le troisième n'a
pas pu être vérifié complètement. On examine aussi le cas où tous les scénarios
restent impossibles à vérifier complètement.

**Résultat attendu :** le taux de réussite est de **50 %** : un succès sur les
deux scénarios pour lesquels on dispose d'un résultat de réussite ou d'échec.
Le scénario non vérifié complètement reste visible à part. Si tous les scénarios
restent dans ce dernier cas, aucun taux de réussite n'est disponible.

**Erreur à éviter :** faire passer « non testé » pour « réussi », ou afficher un
pourcentage qui donne une fausse impression de qualité.

<details>
<summary>Détails techniques</summary>

`aggregate_metrics()` reçoit les statuts `PASS`, `FAIL`, `SKIP`. Le test attend
`scenario_pass_rate=0.5`, `scenario_pass_rate_denominator=2` et
`scenario_counts={"PASS": 1, "FAIL": 1, "SKIP": 1}`. Avec le seul résultat
`SKIP`, `scenario_pass_rate` doit valoir `None`.

</details>

### `test_component_completion_separates_attempts_from_unavailable_requests`

**Pourquoi ce test est utile :** il faut distinguer les tentatives qui
aboutissent des demandes que l'outil n'a même pas pu prendre en charge.

**Situation :** trois dossiers nécessitent l'analyse des photos. L'outil
réussit pour le premier, échoue pour le deuxième et ne peut pas être utilisé
pour le troisième parce que le modèle nécessaire manque.

**Résultat attendu :** le rapport donne deux informations : **50 % de réussite
sur les deux tentatives**, mais seulement **un tiers des trois demandes
satisfaites**. Il indique aussi qu'une demande n'a pas pu être tentée.

**Erreur à éviter :** cacher un problème de disponibilité derrière un bon score
calculé uniquement sur les dossiers que l'outil a pu traiter.

<details>
<summary>Détails techniques et limites</summary>

Les trois résultats ont un composant `vision` respectivement `PASS`, `FAIL` et
`SKIP`. Dans `component_completion.vision`, le test attend `success_rate=0.5`,
`requested_completion_rate` approximativement égal à `1/3`, `attempted=2` et
`skipped=1`. L'absence du modèle est représentée par un résultat construit ;
aucun modèle de vision n'est chargé.

</details>

### `test_agent_completion_counts_a_failure_without_a_result_payload`

**Pourquoi ce test est utile :** si l'agent plante avant de rendre sa conclusion,
ce dossier doit compter comme une tentative échouée.

**Situation :** l'agent termine un dossier, échoue sur un autre sans produire de
réponse, et ne peut pas démarrer sur un troisième parce que son service est
indisponible.

**Résultat attendu :** son taux de dossiers terminés est de **50 %** : une
réussite sur deux tentatives. Si le seul dossier considéré n'a pas pu être
tenté, le taux n'est pas disponible.

**Erreur à éviter :** oublier les échecs sans réponse et afficher artificiellement
100 % de réussite en ne comptant que les réponses reçues.

<details>
<summary>Détails techniques et limites</summary>

Le résultat `PASS` contient `observation.agent={"status": "COMPLETED"}`.
Les résultats `FAIL` (`llm_exception`) et `SKIP` (`ollama_unavailable`) n'ont
aucune sortie d'agent. `agent_completion_rate` doit valoir 0.5 pour l'ensemble
et `None` pour le seul résultat ignoré. Aucun appel au modèle de langage n'est
réalisé ; ces situations sont représentées par des objets de résultat.

</details>

Code concerné : [agrégation](../../../src/evaluation/metrics/aggregation.py) et
[compteurs de complétion](../../../src/evaluation/metrics/completion.py).

## Reconnaître ce qu'on ne peut pas évaluer — [`../test_real_mode_contracts.py`](../test_real_mode_contracts.py)

### `test_unavailable_real_dependencies_never_become_fake_predictions`

**Pourquoi ce test est utile :** lorsqu'une évaluation demande d'utiliser les
vrais outils d'IA, leur absence doit rester visible dans le rapport.

**Situation :** le système peut lire les documents d'un dossier, mais ses
services d'IA et de recherche sont déclarés indisponibles. On vérifie deux
choix de fonctionnement : signaler le dossier comme non vérifié complètement,
ou le marquer en échec.

**Résultat attendu :** la lecture des documents peut réussir, mais les analyses
indisponibles ne doivent produire aucune prédiction inventée. Le rapport
indique que l'évaluation est incomplète et applique le statut choisi. Dans le
cas « non vérifié complètement », il ne fournit aucun taux de réussite.

**Erreur à éviter :** remplacer silencieusement les outils absents par des
réponses simulées et faire croire que la vraie IA a été évaluée avec succès.

<details>
<summary>Détails techniques et limites</summary>

Le scénario `policy_answerable` est évalué en mode `real` avec un
`PreflightReport` injecté qui déclare PostgreSQL, Qdrant, les collections de
connaissances et de sinistres similaires, les embeddings, Ollama et les modèles
de triage, d'anomalie et de vision indisponibles. Les méthodes
`FakeTriagePredictor.predict` et `FakeAnomalyPredictor.assess` sont remplacées
par une fonction qui échoue si elles sont appelées.

Les **2 cas paramétrés** utilisent `dependency_unavailable="SKIP"` ou `"FAIL"`.
Le rapport doit être `PARTIAL`, avec un scénario au statut choisi,
`document=PASS`, `facts.extraction.success=True` et les composants `triage`,
`anomaly`, `retrieval` au statut choisi. Aucune clé `triage` ou `anomaly` ne doit
apparaître dans `facts` ; `model_versions={}`, `agent=None` et `rag=None`.
Dans le cas `SKIP`, aucune assertion ne doit être `FAIL` et
`scenario_pass_rate=None`.

Le code applicatif d'ingestion et d'extraction traite des documents et PDF
synthétiques dans un répertoire temporaire. Aucun service réel n'est démarré ou
sondé. Les deux méthodes surveillées empêchent les remplacements factices
nommés ; le test ne prouve pas l'absence de tout faux service ailleurs.

</details>

### `test_fake_visual_golden_labels_are_not_reused_as_real_image_ground_truth`

**Pourquoi ce test est utile :** pour vérifier ce que l'IA voit sur une photo,
il faut disposer d'une vraie image et savoir quel dommage elle montre.

**Situation :** un scénario de démonstration prévoit un dommage supplémentaire
sur le capot alors que seul le pare-chocs avant est déclaré. Ce résultat a été
défini pour des détecteurs simulés ; aucune image réelle de référence
correspondante n'est fournie. On demande pourtant une évaluation réelle.

**Résultat attendu :** l'analyse visuelle ne peut pas être exécutée et le
scénario reste non vérifié complètement, avec une explication sur l'absence
d'image de référence. Aucun
signal de dommage supplémentaire vu sur une photo ne doit être produit.

**Erreur à éviter :** présenter le dommage prévu par un scénario de démonstration
comme un dommage réellement observé sur une photo, puis prétendre avoir validé
la qualité de l'IA visuelle.

<details>
<summary>Détails techniques et limites</summary>

Le scénario `visual_additional_damage` est exécuté en mode `real` avec un
`PreflightReport()` vide injecté. Il décrit un dommage `HOOD` pour des
détecteurs factices alors que seul `FRONT_BUMPER` est déclaré. Le test attend
un scénario `SKIP`, un composant `vision=SKIP` avec un motif contenant
`no matching real image ground truth`, `observation.vision=None` et l'absence
du signal `VISUAL_ADDITIONAL_DAMAGE`.

Aucun détecteur réel n'est évalué et aucune qualité de détection n'est mesurée.

</details>

Ces deux tests utilisent le véritable adaptateur `RealClaimsEvaluationPipeline`
et les scénarios du [manifeste versionné](../../../data/evaluation/e2e_v2/scenarios.json),
en contrôlant artificiellement l'état des services. Ils vérifient comment le
système représente les données manquantes, pas la disponibilité des services
déployés.

Code concerné : [pipeline réel](../../../src/evaluation/real/pipeline.py),
[politique d'indisponibilité](../../../src/evaluation/real/context.py),
[étape vision](../../../src/evaluation/real/stages/vision.py) et
[évaluation des scénarios](../../../src/evaluation/runner.py).

## Comment ces vérifications sont préparées

Les tests de rapports utilisent des résultats construits par
[`_support.py`](_support.py), sans lancer le traitement applicatif des dossiers.
`result()` leur attribue les versions synthétiques `anomaly=model-v1` et
`investigation=policy-v3`. `baseline_report()` prépare un scénario `golden`
réussi, déclaré `real` et `COMPLETE` dans les données de test : cette déclaration
n'est pas la preuve qu'une pile réelle a été exécutée. Les fichiers produits
sont isolés dans les répertoires temporaires fournis par `tmp_path`.
