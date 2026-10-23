# Comprendre la logique des tests end-to-end

Ces tests mettent l'IA dans des situations de gestion de sinistres et vérifient
si elle réagit de la manière attendue. Pour chaque test, la documentation
explique **la situation, la raison métier et la réaction attendue**.

Commencer par le **[catalogue des situations métier](SCENARIOS.md)**. Par exemple,
[un client qui enchaîne les sinistres](SCENARIOS.md#recent_claim_history) doit
attirer l'attention pour une investigation sur un risque potentiel de fraude.
Plusieurs sinistres ne suffisent pas à prouver que cette personne fraude.

Les tests poursuivent trois objectifs complémentaires :

- **Repérer des situations qui méritent une investigation** : des déclarations
  répétées, une réparation ajoutée au devis ou une différence avec les photos
  peuvent demander une explication.
- **Éviter les soupçons injustifiés** : un dossier cohérent ou un dommage que
  l'IA n'arrive pas à voir ne doit pas recevoir une alerte inventée.
- **Aider le gestionnaire avec des informations fiables** : signaler une pièce
  manquante, retrouver une règle dans le contrat et reconnaître ce qu'on ne sait
  pas sont aussi des comportements à vérifier.

## Où trouver chaque explication ?

| Ce que vous voulez comprendre | Où lire |
| --- | --- |
| Les dix situations métier, le risque recherché et la réaction attendue | [Catalogue des scénarios](SCENARIOS.md) |
| Les deux essais complémentaires sur les fichiers disponibles | [Essais complémentaires](SCENARIOS.md#diagnostics-complementaires) |
| Pourquoi vérifier les décisions de l'IA et les preuves qu'elle utilise | [Tests des décisions et des preuves](deterministic/README.md) |
| Pourquoi vérifier les comptes rendus et les résultats annoncés | [Tests des rapports et de leur fiabilité](reporting/README.md) |
| Pourquoi contrôler la liste des scénarios à exécuter | [Contrôles de la liste des tests, ci-dessous](#controles-de-la-liste-des-tests) |
| Comment lancer une évaluation et consulter ses fichiers | [Guide d'évaluation](../../docs/END_TO_END_EVALUATION.md) |

Les fiches commencent par une explication en termes courants. Les noms de
champs, codes et valeurs précises sont accessibles dans des détails repliables.
Les inventaires couvrent **27 fonctions de test**, soit **49 cas en comptant
leurs variantes**, dont les dix situations métier.

<a id="controles-de-la-liste-des-tests"></a>

## Contrôles de la liste des tests

Ces quatre tests vérifient que les situations à essayer sont bien définies et
que la sélection correspond à ce qu'on a demandé. Ils empêchent de donner un
résultat rassurant alors qu'on n'aurait pas essayé les bonnes situations.

### `test_versioned_benchmark_has_all_golden_cases_and_existing_declared_files`

**Pourquoi ce test existe.** Un compte rendu n'a de sens que si toutes les
situations prévues sont présentes et disposent des fichiers nécessaires.

**La situation.** On vérifie la liste des dix scénarios, ce que chacun doit
permettre de constater et la présence des fichiers annoncés.

**Le résultat attendu.** Aucune situation prévue ne manque et chaque fichier
annoncé existe. Cela garantit la présence des exemples à essayer ; leur analyse
est vérifiée par les autres tests.

<details>
<summary>Détails techniques et vérifications exactes</summary>

Source : [`test_scenarios.py`](test_scenarios.py).

**Objectif :** garantir la présence du jeu de référence attendu et de ses
fichiers déclarés.

**Préparation et action :** charger le manifeste par défaut avec
`load_scenarios()`, comparer les identifiants à `EXPECTED_IDS`, puis parcourir
chaque scénario et ses chemins de fichiers depuis la racine du dépôt.

**Résultat attendu :** exactement les dix identifiants attendus, la version de
benchmark `e2e_claims_v1`, des faits attendus, des actions agent acceptables et
une liste non vide de fichiers pour chaque scénario. Chaque fichier déclaré
doit exister. Ce contrôle ne vérifie pas à lui seul le contenu des PDF, les
modèles ou la disponibilité des services externes.

</details>

### `test_unknown_version_is_rejected_before_execution`

**Pourquoi ce test existe.** Une nouvelle liste de tests peut demander des
vérifications que l'application ne connaît pas encore. Lancer quand même
l'évaluation pourrait produire un résultat trompeur.

**La situation.** On fournit une liste qui porte un numéro de version absent,
ancien ou non pris en charge.

**Le résultat attendu.** L'application refuse cette liste avec une explication
avant d'essayer les scénarios. Elle ne fait pas semblant de savoir les vérifier.

<details>
<summary>Détails techniques et vérifications exactes</summary>

Source : [`test_scenarios.py`](test_scenarios.py).

**Objectif :** refuser un manifeste dont la version n'est pas prise en charge.

**Préparation et action :** copier le manifeste dans un fichier temporaire,
remplacer `benchmark_version`, puis tenter son chargement.

**Variantes :** chaîne vide, `e2e_claims_v0` et `e2e_claims_v2`.

**Résultat attendu :** pour chacune des trois variantes, une `ValidationError`
contenant `Unsupported E2E benchmark version`. Aucun scénario n'est exécuté.

</details>

### `test_duplicate_ids_and_unknown_schema_fields_are_rejected`

**Pourquoi ce test existe.** Une erreur dans la description des tests ne doit
pas passer silencieusement dans les résultats.

**La situation.** On inscrit deux fois le même scénario, puis, dans un autre
essai, on ajoute une consigne dont l'application ne connaît pas le nom.

**Le résultat attendu.** L'application signale chacune de ces erreurs au lieu
de compter deux fois la même situation ou d'ignorer une consigne mal écrite.

<details>
<summary>Détails techniques et vérifications exactes</summary>

Source : [`test_scenarios.py`](test_scenarios.py).

**Objectif :** éviter les scénarios ambigus et les fautes dans les noms de
champs du manifeste.

**Préparation et action :** effectuer deux mutations indépendantes d'une copie
du manifeste : dupliquer le premier scénario, puis ajouter le champ inconnu
`unknown_expectation=true` au premier scénario d'une nouvelle copie.

**Résultat attendu :** le doublon produit une `ValidationError` mentionnant
`unique` ; le champ inconnu produit une `ValidationError` mentionnant
`extra_forbidden`.

</details>

### `test_filter_preserves_benchmark_order_and_rejects_typos`

**Pourquoi ce test existe.** Quand on demande quelques tests seulement, on doit
être certain de lancer ceux qu'on a choisis.

**La situation.** On sélectionne deux situations dans un ordre différent de
celui de la liste, puis on demande un test dont le nom contient une erreur.

**Le résultat attendu.** La première demande conserve uniquement les deux
situations choisies dans l'ordre prévu par la liste. La faute de frappe est
signalée ; elle ne doit pas donner l'impression qu'une évaluation vide a réussi.

<details>
<summary>Détails techniques et vérifications exactes</summary>

Source : [`test_scenarios.py`](test_scenarios.py).

**Objectif :** rendre la sélection explicite et empêcher une faute de frappe
de produire une évaluation vide trompeuse.

**Préparation et action :** demander `visual_non_detection`, puis
`simple_coherent_claim` dans cet ordre ; effectuer ensuite une sélection sur
l'identifiant inexistant `coherent_claim_typo`.

**Résultat attendu :** la première sélection renvoie seulement
`simple_coherent_claim`, puis `visual_non_detection`, dans l'ordre du
benchmark. La seconde lève une `ValueError` contenant
`Unknown E2E scenario IDs: coherent_claim_typo`.

</details>

## Pour vérifier les résultats en détail

<details>
<summary>Fonctionnement de l'évaluation, statuts et fichiers de résultats</summary>

## Ce que l'on teste

Le parcours métier part d'un dossier de sinistre : documents, extraction des
champs, comparaison des pièces, construction des caractéristiques, triage,
anomalies, historique, signaux d'investigation, puis recherche documentaire
(RAG), vision et recommandation de l'agent selon le scénario. L'agent assiste
une revue humaine ; il ne décide pas d'un paiement ou de l'acceptation d'un
sinistre.

Il faut distinguer les deux niveaux d'exécution :

- **Suite pytest dans ce dossier** : exécution hors ligne, stockage en mémoire,
  prédictions ML, embeddings et réponses LLM contrôlés. Les services métier
  d'extraction, de comparaison, d'investigation et de validation restent
  sollicités. Des tests modifient volontairement une observation pour vérifier
  que l'évaluateur détecte l'erreur. Les tests de contrats du mode réel injectent
  des dépendances indisponibles ; ils ne lancent pas la pile complète.
- **Évaluation réelle par le script dédié** : utilisation des services, modèles
  et fichiers locaux disponibles, après vérification des prérequis. Une
  dépendance absente produit un statut explicite et n'est pas remplacée par une
  prédiction simulée. Les attentes réservées au mode déterministe ne sont pas
  imposées aux modèles réels. Les deux diagnostics complémentaires sont évalués
  séparément des dix scénarios de référence.

Le détail du chemin réellement parcouru se trouve dans
[`ClaimsEvaluationPipeline`](../../src/evaluation/pipeline.py), le
[pipeline réel](../../src/evaluation/real/pipeline.py) et les
[assertions d'évaluation](../../src/evaluation/assertions/__init__.py).

## Résultats attendus et lecture d'un échec

Les fichiers JSON du benchmark sont la référence exécutable des attentes ; les
fiches en expliquent le sens. Les noms tels que
`features.missing_document_count` désignent des champs structurés de
l'observation, pas une phrase que le modèle doit reproduire.

| Résultat d'un scénario évalué | Signification |
| --- | --- |
| `PASS` | Toutes les attentes obligatoires sont satisfaites, sans composant en échec ni erreur d'exécution. |
| `FAIL` | Au moins une attente obligatoire, un composant ou l'exécution a échoué. |
| `SKIP` | Une attente obligatoire reste invérifiable parce qu'une exécution nécessaire est indisponible, sans échec prioritaire. |

Un `SKIP` ne compte jamais comme un succès. Le taux de réussite est
`PASS / (PASS + FAIL)` ; il est absent (`null`) si aucun scénario n'est en
`PASS` ou `FAIL`, notamment si tous sont en `SKIP`. Un rapport reste `PARTIAL`
si la pile requise n'a pas entièrement
réussi. Le statut d'un scénario et celui d'une fonction pytest sont distincts :
un test qui vérifie le rejet d'une fausse citation doit réussir lorsque
l'évaluateur signale correctement cette citation en `FAIL`.

Lors d'une exécution par le script, `scenario_results.json` conserve les
assertions avec `name`, `component`, `expected`, `actual`, `status` et `source`.
Les fichiers `traces/<scenario_id>.json` et les artefacts `failures/` permettent
de retrouver les observations et leurs sources. Le
[guide d'évaluation](../../docs/END_TO_END_EVALUATION.md#assertions-metrics-and-artifacts)
décrit l'ensemble des sorties. Les tests pytest utilisent leurs assertions et
des répertoires temporaires ; ils ne produisent pas tous un rapport persistant
dans `artifacts/`.

</details>

## Maintenir ces explications

Quand un test change, actualiser d'abord sa situation, sa raison métier et la
réaction attendue, puis les détails de vérification. Le lecteur doit pouvoir
comprendre pourquoi le test existe sans connaître les noms des champs du code.
Les données et attentes précises restent dans le
[fichier des scénarios](../../data/evaluation/e2e_v1/scenarios.json) ; le
[guide](../../docs/END_TO_END_EVALUATION.md#frozen-golden-cases) explique leur
versionnement. La version de ces scénarios est distincte de la version de
l'application dans `pyproject.toml`.

Conformément à [`AGENTS.md`](../../AGENTS.md), un agent ne doit lancer aucun
test de `tests/e2e/`, même marqué `unit`, ni le script d'évaluation ou une
commande équivalente sans demande explicite. Une demande de documentation ou
de validation ne suffit pas. Aucun test E2E n'a été exécuté pour cette révision.
