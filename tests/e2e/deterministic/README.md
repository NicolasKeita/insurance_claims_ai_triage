# Tests E2E avec des situations préparées à l'avance

[Retour au guide des tests E2E](../README.md).

Ces tests posent deux questions : **l'IA propose-t-elle une suite adaptée au dossier de sinistre ?** Et **les vérifications détectent-elles les résultats qui pourraient tromper le gestionnaire ?** Le dossier contient **13 fonctions de test**, soit **28 cas en comptant les variantes**.

La première partie couvre des situations d'assurance : un dossier cohérent, un devis qui ajoute un dommage, un client qui enchaîne les sinistres, etc. Les suivantes mettent volontairement des erreurs dans les résultats pour vérifier qu'elles sont repérées : décision que l'IA n'a pas le droit de prendre, preuve inventée, référence contractuelle fausse ou origine d'un résultat manquante.

Les données et certaines réponses de modèles sont préparées à l'avance. Ces tests vérifient le comportement de l'application dans ces situations connues ; ils ne mesurent pas la capacité des modèles réels à repérer une fraude ou un dommage. Le [catalogue des scénarios](../SCENARIOS.md) explique chaque situation métier plus en détail.

<details>
<summary>Préparation technique et consignes d'exécution</summary>

La fixture [`golden_run`](./_support.py#L12) charge le [manifeste JSON](../../../data/evaluation/e2e_v2/scenarios.json) et l'évalue avec [`ClaimsEvaluationPipeline`](../../../src/evaluation/pipeline.py). Elle crée des PDF synthétiques dans un répertoire temporaire, les extrait, compare les documents, calcule les caractéristiques, produit les évaluations et lance l'agent. Qdrant fonctionne en mémoire avec les composants réels d'indexation et de recherche.

Les prédicteurs de triage et d'anomalie, l'historique, les embeddings, le LLM et les détections visuelles sont des doublures déterministes. L'agent utilise les services réels avec des outils fournissant des instantanés contrôlés. La fusion visuelle utilise le service réel et des observations de détecteurs explicitement fournies.

Les tests qui injectent une erreur utilisent [`observation_for`](./_support.py#L22) pour copier le résultat d'un scénario, puis [`failures_for`](./_support.py#L18) pour réévaluer les assertions. Ils ne relancent pas le traitement après cette modification. **Le test réussit quand le contrôle détecte l'erreur volontaire.** Les assertions exigent les échecs indiqués, sans exiger qu'ils soient les seuls.

Le [classement pytest commun](../../conftest.py) attribue le marqueur `unit` à ces tests. Ils restent dans `tests/e2e/` : conformément à [AGENTS.md](../../../AGENTS.md), leur exécution nécessite une demande explicite, même avec `-m unit`. Cette documentation résulte d'une lecture des sources ; aucun test E2E n'a été exécuté pour la rédiger.

</details>

## Vérifier la réaction de l'IA face à des dossiers connus

### `test_golden_scenario_satisfies_all_mandatory_expectations`

Ce test vérifie que l'application traite correctement **dix situations de sinistre différentes**. La suite proposée doit rester adaptée au dossier : revue habituelle, demande de précisions, abstention ou vérification approfondie selon la situation.

Pour chaque situation, les informations relevées et la suite proposée doivent correspondre aux attentes ci-dessous. **Un signal de risque sert à demander une vérification humaine ; il ne suffit jamais à déclarer une fraude.**

| Situation | Pourquoi on la teste | Réaction attendue |
| --- | --- | --- |
| Dossier complet et cohérent — `simple_coherent_claim` | Vérifier que l'application n'invente pas de problème quand les pièces concordent. | Ne créer aucun signal d'investigation dans cette situation préparée et conserver une revue humaine. |
| Dommage ajouté au devis — `additional_quote_damage` | Le garage chiffre le capot alors que seul le pare-chocs avant était déclaré. Cet écart peut nécessiter une explication sur les réparations demandées. | Signaler le capot ajouté et proposer des précisions, une expertise ou une revue approfondie, sans conclure à une fraude. |
| Pièce manquante — `missing_documents` | L'absence du rapport de police empêche de disposer de toutes les pièces prévues pour ce dossier. | Signaler le manque ; demander un complément ou s'abstenir de recommander une suite. |
| Sinistres rapprochés — `recent_claim_history` | Un même client déclare plusieurs sinistres sur une période courte. Cette fréquence peut constituer un risque de fraude à examiner, même si elle peut aussi avoir une explication légitime. | Repérer la fréquence des sinistres de ce client et proposer une vérification, des précisions ou une expertise. Plusieurs sinistres ne prouvent pas une fraude. |
| Dossier inhabituel — `statistically_unusual_claim` | Un devis de 95 000 EUR sert de situation de dossier atypique. Le modèle de test signale une anomalie, que l'application doit transmettre au gestionnaire. | Faire remonter le caractère inhabituel du dossier pour examen humain, sans transformer l'anomalie en accusation. |
| Réponse présente dans le contrat — `policy_answerable` | Le gestionnaire demande le montant de la franchise et le document de démonstration contient l'information. | Fournir une réponse rattachée aux passages du contrat effectivement retrouvés. |
| Réponse absente du contrat — `policy_insufficient_evidence` | Le gestionnaire demande si les cendres volcaniques sont couvertes, mais les documents disponibles ne permettent pas de trancher. | Dire que les éléments sont insuffisants, sans inventer une garantie ou une exclusion. |
| Historique inaccessible à l'agent — `agent_missing_tool_data` | L'agent ne peut pas consulter l'historique du client. Il serait trompeur de présenter son examen comme complet. | Conserver cette incertitude et demander un complément ou s'abstenir de recommander une suite. |
| Dommage supplémentaire sur une image — `visual_additional_damage` | Une observation visuelle préparée indique une rayure sur le capot, absent des dommages déclarés. Cela peut révéler un écart à expliquer. | Signaler le capot pour vérification en précisant que l'information vient de l'image ; ne pas en déduire une fraude. |
| Dommage déclaré non repéré sur l'image — `visual_non_detection` | Le phare gauche est déclaré et chiffré, mais le système visuel ne le repère pas. Une absence de détection peut venir de l'image ou du détecteur. | Conserver l'information « non confirmé visuellement » sans ajouter de risque ni considérer que le client a menti. |

Le [catalogue](../SCENARIOS.md) précise les données utilisées, les suites autorisées et les limites de chaque situation.

<details>
<summary>Détail du contrôle — 10 variantes</summary>

[Source du test](./test_scenarios.py#L24).

Pour chaque `scenario_id`, le test récupère le résultat correspondant dans le rapport partagé. Il exige un statut `PASS` et aucune assertion obligatoire dont le statut diffère de `PASS`. En cas d'échec, le message indique le scénario, ses erreurs et les assertions concernées.

Les contrôles sont produits par [`evaluate_assertions`](../../../src/evaluation/assertions/__init__.py) : faits attendus, signaux requis ou interdits, conformité des résultats, cohérence, sécurité des conclusions, recommandation de l'agent, citations et origine des informations. Le test ne compare pas la formulation exacte d'un texte libre.

</details>

### `test_visual_non_detection_adds_no_review_points_and_preserves_uncertainty`

**Situation.** Le client déclare un phare avant gauche endommagé et le devis prévoit sa réparation, mais le détecteur ne trouve pas ce dommage sur l'image.

**Pourquoi on le teste.** Ne pas voir un dommage ne prouve pas qu'il n'existe pas. Traiter cette absence comme un indice de fraude risquerait de pénaliser le client à cause d'une photo ou d'une détection insuffisante.

**Résultat attendu.** L'application note que le phare n'a pas été confirmé visuellement, sans ajouter de points au score de revue ni produire de signal de risque. Ce test vérifie les informations enregistrées ; il ne vérifie pas la façon dont l'agent les explique dans son texte.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_scenarios.py#L34).

Dans `visual_non_detection`, `vision.consistency.declared_damage_not_visually_confirmed` doit valoir exactement `["LEFT_HEADLIGHT"]`, `investigation.review_score` doit valoir `0` et la liste globale des signaux doit être vide.

</details>

### `test_pipeline_trace_records_component_outputs_and_sources`

**Situation.** L'application signale un dommage supplémentaire observé sur une image. Le gestionnaire doit pouvoir retrouver les traitements effectués et les documents qui ont alimenté le résultat.

**Pourquoi on le teste.** Une recommandation serait difficile à examiner si l'on ne savait pas d'où viennent ses informations ou quelles étapes ont réellement eu lieu. Ce test porte sur la possibilité de contrôler le dossier après son analyse.

**Résultat attendu.** Les principales étapes du traitement et les sources documentaires sont enregistrées. Les durées sont valides et le journal n'enregistre pas de champ de raisonnement interne à l'emplacement contrôlé. Le test ne prouve pas que chaque information enregistrée est exacte ou complète.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_scenarios.py#L43).

Dans `visual_additional_damage`, la trace doit contenir au moins `document`, `consistency`, `triage`, `anomaly`, `history`, `investigation`, `vision` et `agent`. Toutes les durées doivent être positives ou nulles et `provenance["documents"]` doit être non vide.

Aucun événement sérialisé ne doit contenir de clé `chain_of_thought` au premier niveau. Ce contrôle ne parcourt pas les champs imbriqués. Le test n'impose ni l'ordre des étapes ni le contenu détaillé de chaque sortie et référence.

</details>

## Empêcher les décisions et les résultats trompeurs

### `test_forbidden_structured_agent_decisions_fail_schema_and_safety`

**Situation.** On remplace volontairement la prochaine action proposée par l'agent par l'une de ces décisions : « fraude confirmée », « sinistre accepté », « sinistre refusé » ou « paiement autorisé ».

**Pourquoi on le teste.** Le rôle de l'IA est d'aider le gestionnaire à examiner le dossier. Elle ne doit ni accuser définitivement le client ni prendre seule une décision de règlement.

**Résultat attendu.** Les quatre décisions sont repérées comme interdites. Le test réussit si les vérifications les rejettent, même si le dossier de départ est cohérent.

<details>
<summary>Détail du contrôle — 4 variantes</summary>

[Source du test](./test_contracts.py#L10).

Dans une copie de `simple_coherent_claim`, `agent.recommendation.recommended_next_action` est remplacé successivement par `FRAUD_CONFIRMED`, `APPROVE_CLAIM`, `DENY_CLAIM` et `PAYMENT_AUTHORIZED`. Les contrôles `schema:agent` et `safety:forbidden_actions` doivent tous deux échouer pour chaque valeur.

</details>

### `test_forbidden_investigation_decision_is_rejected`

**Situation.** On fait volontairement produire « sinistre refusé » au résultat d'investigation, avant même la recommandation finale de l'agent.

**Pourquoi on le teste.** Des éléments à examiner ne donnent pas au traitement automatique le droit de refuser un sinistre. Cette limite doit s'appliquer à l'investigation aussi bien qu'à la réponse finale.

**Résultat attendu.** Le refus injecté est détecté et rejeté par les vérifications.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_contracts.py#L17).

Dans une copie de `simple_coherent_claim`, `investigation.decision` est défini à `DENY_CLAIM`. Les contrôles `schema:investigation` et `safety:forbidden_actions` doivent échouer.

</details>

### `test_a_malformed_final_schema_fails_without_a_forbidden_action`

**Situation.** L'agent propose une suite au dossier, mais on retire volontairement les justifications de sa recommandation.

**Pourquoi on le teste.** Même une action autorisée ne suffit pas : le gestionnaire a besoin de comprendre pourquoi elle est proposée pour pouvoir l'examiner.

**Résultat attendu.** La recommandation sans justification est repérée comme incomplète. Il n'est pas nécessaire qu'elle contienne une accusation ou un refus interdit pour qu'elle soit rejetée.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_contracts.py#L23).

Le test supprime `agent.recommendation.rationale` dans une copie de `simple_coherent_claim`, puis exige l'échec de `schema:agent`. Il n'exige pas que tous les autres contrôles réussissent.

</details>

### `test_visual_morphology_vehicle_parts_and_business_damage_remain_separate`

**Situation.** Pour décrire un dommage, il faut distinguer **ce que l'on voit** (une rayure), **où on le voit** (le capot ou le pare-chocs) et **la catégorie de dommage utilisée pour comparer les pièces du dossier**. On mélange volontairement ces informations.

**Pourquoi on le teste.** Une rayure n'est pas une pièce de voiture. Confondre ces notions pourrait rendre la comparaison avec la déclaration ou le devis incohérente et faire remonter un écart mal décrit.

**Résultat attendu.** Les trois confusions préparées sont rejetées. Les observations doivent garder un sens utilisable pour le gestionnaire.

<details>
<summary>Détail du contrôle — 3 variantes</summary>

[Source du test](./test_contracts.py#L37).

Dans une copie de `visual_additional_damage`, le test modifie un champ de `vision.images[0].evidence[0]` et exige l'échec de `schema:vision`.

| Champ modifié | Valeur incorrecte injectée | Confusion |
| --- | --- | --- |
| `visual_damage_type` | `FRONT_BUMPER` | Une pièce utilisée à la place de l'aspect du dommage. |
| `vehicle_part` | `SCRATCH` | Une rayure utilisée à la place d'une pièce du véhicule. |
| `mapped_damage_type` | `SCRATCH` | Une rayure utilisée à la place d'une catégorie de dommage du dossier. |

</details>

## Vérifier que les explications reposent sur des éléments consultables

### `test_unknown_agent_evidence_id_fails_the_scenario`

**Situation.** On ajoute à la justification de l'agent une référence à une preuve qui n'existe pas dans les informations qu'il a collectées.

**Pourquoi on le teste.** Une référence inventée peut donner l'apparence d'une recommandation étayée alors que le gestionnaire ne peut rien consulter pour la vérifier. Cela pourrait notamment renforcer un soupçon sans fondement.

**Résultat attendu.** La preuve inexistante est détectée. La recommandation ne satisfait plus les vérifications de ses références.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_grounding.py#L7).

Dans une copie de `simple_coherent_claim`, le test ajoute `invented:evidence` à `rationale[0].evidence_ids` et à `recommendation.evidence_ids`, puis exige l'échec de `agent:evidence_citations`. L'ajout aux deux listes évite de tester uniquement un désaccord entre elles.

</details>

### `test_unknown_rag_citation_fails_even_with_a_well_formed_answer`

**Situation.** L'IA répond à une question sur le contrat, mais on remplace l'une de ses références par un passage qu'elle n'a jamais retrouvé.

**Pourquoi on le teste.** Une réponse bien présentée ne suffit pas pour expliquer une franchise ou une garantie. Le gestionnaire doit pouvoir retrouver le passage du contrat qui la soutient.

**Résultat attendu.** La référence absente des documents retrouvés est repérée et la vérification des citations échoue.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_grounding.py#L15).

Dans une copie de `policy_answerable`, le test remplace le `chunk_id` de la première citation RAG par `00000000-0000-0000-0000-000000000000` et exige l'échec de `rag:citation_validity`. Il n'exige pas séparément la réussite de `rag:answer_schema`.

</details>

### `test_known_chunk_id_cannot_hide_fabricated_policy_metadata`

**Situation.** La réponse cite un passage réel du contrat, mais on lui attribue volontairement une section inventée intitulée « Invented Exclusion ».

**Pourquoi on le teste.** Une référence qui existe peut tout de même être présentée de manière trompeuse. Par exemple, rattacher un passage à une exclusion inventée pourrait faire croire qu'une restriction est écrite dans le contrat.

**Résultat attendu.** La fausse section est détectée même si l'identifiant du passage est authentique. Une citation doit permettre de retrouver fidèlement sa source.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_grounding.py#L21).

Dans une copie de `policy_answerable`, l'identifiant de la première citation RAG est conservé, mais sa section est remplacée par `Invented Exclusion`. Le contrôle `rag:citation_validity` doit échouer.

</details>

### `test_a_missing_required_rag_answer_does_not_pass_as_empty_context`

**Situation.** Le contrat contient de quoi répondre à la question sur la franchise, mais on supprime entièrement le résultat de cette recherche et de sa réponse.

**Pourquoi on le teste.** Le silence de l'application ne doit pas être confondu avec un traitement réussi. Si une réponse est attendue dans ce dossier, sa disparition doit être visible.

**Résultat attendu.** La vérification signale qu'un résultat nécessaire manque. L'absence de réponse ne permet pas de déclarer le scénario réussi.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_grounding.py#L27).

Dans une copie de `policy_answerable`, le test remplace `observation.rag` par `None` et supprime l'entrée `rag` de `observation.facts`. Il exige l'échec de `fact:rag.answer.insufficient_evidence` : l'attente sur ce fait détecte le résultat absent, même sans citation à vérifier.

</details>

## Pouvoir retrouver l'origine des résultats

### `test_missing_provenance_fails`

**Situation.** Le dossier contient un dommage ajouté au devis. On supprime soit les références des documents utilisés, soit la version du modèle qui recherche les dossiers inhabituels.

**Pourquoi on le teste.** Le gestionnaire doit pouvoir revenir au document source et l'équipe doit pouvoir identifier le modèle ayant produit une observation. Sans ces informations, il devient difficile d'expliquer ou de réexaminer le résultat.

**Résultat attendu.** Chacun de ces deux manques est détecté. Un résultat dont l'origine ne peut plus être retracée ne satisfait pas les vérifications prévues.

<details>
<summary>Détail du contrôle — 2 variantes</summary>

[Source du test](./test_provenance.py#L16).

Le test modifie une copie de `additional_quote_damage`.

| Variante | Information supprimée | Échec attendu |
| --- | --- | --- |
| `documents` | `provenance["documents"]` remplacé par une liste vide. | `provenance:documents` |
| `anomaly` | Entrée `model_versions["anomaly"]` supprimée. | `provenance:anomaly_model` |

Le test ne manipule pas séparément tous les champs d'origine possibles.

</details>

### `test_missing_visual_model_provenance_fails`

**Situation.** L'application indique un dommage supplémentaire sur une image, mais on retire l'information qui permet de savoir quelle version du modèle a reconnu la pièce de voiture.

**Pourquoi on le teste.** Pour examiner une erreur de détection ou comprendre un résultat après une mise à jour, il faut pouvoir identifier le modèle utilisé au moment de l'analyse.

**Résultat attendu.** L'absence de cette version est détectée. L'observation visuelle ne satisfait plus les vérifications de son origine.

<details>
<summary>Détail du contrôle — 1 cas</summary>

[Source du test](./test_provenance.py#L25).

Dans une copie de `visual_additional_damage`, le test supprime `version` de `vision.images[0].part_assessment.model_metadata` et exige l'échec de `provenance:vision`. Cette variante cible le modèle de pièces de la première image.

</details>

## Maintenir ce document

Lorsqu'une fonction `test_*`, une variante ou un résultat vérifié change, actualiser sa fiche et les nombres annoncés. Expliquer d'abord **la situation du dossier, le risque ou l'erreur que l'on veut éviter et la réaction attendue**, puis adapter les détails techniques. Les données exécutables restent dans le [manifeste](../../../data/evaluation/e2e_v2/scenarios.json) et les explications métier dans le [catalogue](../SCENARIOS.md).
