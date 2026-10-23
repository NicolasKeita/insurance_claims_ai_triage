# La logique métier des tests end-to-end

Ces tests mettent l'IA face à des situations de gestion de sinistres. Pour
chaque situation, on veut comprendre **ce qui doit attirer son attention,
pourquoi cela compte et quelle réaction on attend d'elle**.

Certains tests participent à la **détection du risque de fraude** : sinistres
répétés, réparations ajoutées au devis, dossier inhabituel ou dommage visuel
absent de la déclaration. D'autres vérifient que l'IA traite correctement un
dossier ordinaire, demande une pièce manquante ou reconnaît qu'elle ne sait
pas répondre. Un indice de risque appelle une vérification humaine ; il ne
prouve pas une fraude.

Les dix situations ci-dessous sont des exemples préparés pour les tests.
La réussite d'un exemple ne mesure pas à elle seule la capacité de l'IA à
détecter la fraude dans des dossiers réels. Les détails de vérification sont
repliés sous chaque fiche pour laisser la logique métier au premier plan.

[Retour au guide des tests](README.md).

## Les dix situations testées

| Test | La question métier |
| --- | --- |
| [Dossier cohérent](#simple_coherent_claim) | L'IA évite-t-elle une alerte injustifiée ? |
| [Réparation ajoutée au devis](#additional_quote_damage) | Repère-t-elle une réparation qui demande une explication ? |
| [Pièce manquante](#missing_documents) | Prévient-elle que le dossier est incomplet ? |
| [Sinistres répétés](#recent_claim_history) | Repère-t-elle un enchaînement de sinistres à examiner pour un risque potentiel de fraude ? |
| [Dossier inhabituel](#statistically_unusual_claim) | Fait-elle remonter une alerte pour vérification ? |
| [Réponse dans le contrat](#policy_answerable) | Retrouve-t-elle la règle et sa source ? |
| [Réponse absente du contrat](#policy_insufficient_evidence) | Reconnaît-elle qu'elle ne peut pas répondre ? |
| [Historique inaccessible](#agent_missing_tool_data) | Évite-t-elle d'inventer les antécédents du client ? |
| [Dommage supplémentaire observé](#visual_additional_damage) | Demande-t-elle de comprendre l'écart avec la déclaration ? |
| [Dommage non détecté](#visual_non_detection) | Évite-t-elle de transformer une non-détection en soupçon de fraude ? |

<a id="simple_coherent_claim"></a>

### `simple_coherent_claim` — Traiter normalement un dossier cohérent

**Le but du test.** Vérifier que l'IA sait aussi reconnaître un dossier sans
motif d'alerte dans l'exemple préparé. Un outil de détection du risque doit
éviter de créer des soupçons sur chaque client.

**La situation.** Un client déclare un pare-chocs avant endommagé. Ses documents
racontent la même histoire et le devis de 1 200 € concerne ce même dommage.
Toutes les pièces prévues sont disponibles.

**La logique métier.** Quand les informations concordent, il n'y a pas de raison
d'inventer une différence entre les documents ou une réparation supplémentaire.
Ce test protège contre les fausses alertes qui feraient examiner inutilement
un dossier ordinaire.

**La réaction attendue.** Dans cet exemple, l'IA ne doit ajouter aucun motif
d'investigation. Elle peut proposer de poursuivre la revue habituelle, demander
des précisions ou s'abstenir de recommander une action. Le gestionnaire garde la
main : la cohérence du dossier ne vaut pas acceptation du remboursement.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-COHERENT-0001`, les quatre pièces sont
disponibles. Formulaire, constat et devis portent uniquement sur `FRONT_BUMPER`
(pare-chocs avant) ; devis de 1 200 EUR. Parcours commun, sans vision.

**Résultats vérifiés.** Outre le socle d'extraction, les correspondances
`claim_id_match`, `incident_date_match`, `location_match`, `vehicle_match`,
`collision_type_match`, `injuries_match` et `declared_damage_match` de
`consistency` sont toutes `true`. `consistency.additional_quote_damage` vaut
`[]`, `features.missing_document_count` vaut `0` et
`features.document_completeness_ratio` vaut `1.0`.
`investigation.signals` doit être exactement `[]` en déterministe.

**Actions acceptables.** `CONTINUE_STANDARD_REVIEW`,
`REQUEST_ADDITIONAL_INFORMATION` ou `NO_RECOMMENDATION`.

**Interdits.** `MISSING_DOCUMENTS`, `ADDITIONAL_QUOTE_DAMAGE` et
`VISUAL_ADDITIONAL_DAMAGE` ne doivent pas apparaître.

**Limites.** L'absence globale de signaux est un fait marqué
`deterministic_only_facts` : elle n'est pas exigée en réel, où les prédictions
ML peuvent produire des observations supplémentaires. Les interdits explicites
restent applicables quand leur composant peut être évalué.

</details>

<a id="additional_quote_damage"></a>

### `additional_quote_damage` — Comprendre une réparation ajoutée au devis

**Le but du test.** Vérifier que l'IA remarque une réparation demandée par le
garage qui n'apparaît pas dans la déclaration du client.

**La situation.** Le client déclare uniquement un pare-chocs avant endommagé.
Le devis de 2 200 € prévoit aussi une réparation du capot.

**La logique métier.** Il faut comprendre pourquoi le capot est ajouté : oubli
dans la déclaration, dommage découvert ensuite ou réparation sans rapport avec
le sinistre. Un ajout injustifié pourrait correspondre à une tentative de faire
financer des réparations supplémentaires. Cette différence peut donc alimenter
la détection d'un risque de fraude, mais elle ne prouve aucune intention de
frauder du client ou du garage.

**La réaction attendue.** L'IA doit signaler le capot comme réparation non
expliquée par la déclaration, en montrant qu'elle vient du devis. Elle doit
proposer de demander une explication, une expertise ou une revue approfondie.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-QUOTE-0001`, dossier complet ; le
formulaire et le constat déclarent `FRONT_BUMPER`, le devis de 2 200 EUR ajoute
`HOOD` (capot). La comparaison documentaire alimente les caractéristiques puis
le moteur d'investigation. Sans vision ; RAG selon le socle commun.

**Résultats vérifiés.** `consistency.additional_quote_damage` vaut exactement
`["HOOD"]`, `features.additional_damage_count` vaut `1` et le signal
`ADDITIONAL_QUOTE_DAMAGE` est obligatoire. Son contrôle de provenance exige
une source de type devis et des preuves `damage_types` correspondant exactement
aux dommages supplémentaires constatés.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION`, `REFER_TO_EXPERT`
ou `REFER_FOR_INVESTIGATION_REVIEW`.

**Interdits.** `VISUAL_ADDITIONAL_DAMAGE` et `MISSING_DOCUMENTS`. L'ajout au
devis ne doit pas devenir une affirmation de fraude ou une décision de refus.

**Limites.** Le dommage ajouté est une vérité documentaire synthétique,
vérifiable avec les extracteurs réels dans les deux modes. Aucune observation
photographique ne fonde ce signal et aucun score de modèle réel précis n'est
attendu.

</details>

<a id="missing_documents"></a>

### `missing_documents` — Faire compléter un dossier incomplet

**Le but du test.** Vérifier que l'IA prévient le gestionnaire lorsqu'une pièce
prévue au dossier manque.

**La situation.** Le formulaire, le constat et le devis sont disponibles, mais
le rapport de police annoncé dans le dossier manque.

**La logique métier.** Le gestionnaire doit savoir sur quels éléments il peut
s'appuyer et ce qu'il lui reste à obtenir. Si l'IA passe le manque sous silence,
elle donne l'impression que le dossier est complet. Une pièce manquante peut
simplement être en attente de transmission ; ce fait seul ne démontre pas une
fraude.

**La réaction attendue.** L'IA doit signaler que le dossier est incomplet et
proposer de demander des informations supplémentaires, ou s'abstenir de
recommander une prochaine action. Ce scénario ne signifie pas qu'un rapport
de police est obligatoire pour tous les sinistres.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-MISSING-0001` ; quatre entrées sont
déclarées au dossier, mais `POLICE_REPORT` est indisponible et son PDF n'est pas
créé. Les trois documents structurés restent présents et cohérents sur
`FRONT_BUMPER`, avec un devis de 1 200 EUR. L'extraction peut donc réussir
avant le calcul de complétude. Sans vision ; RAG selon le socle commun.

**Résultats vérifiés.** `features.document_count` vaut `4`,
`features.missing_document_count` vaut `1`,
`features.document_completeness_ratio` vaut `0.75` et
`consistency.additional_quote_damage` vaut `[]`. Le signal `MISSING_DOCUMENTS`
est obligatoire selon la politique de complétude actuelle.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION` ou
`NO_RECOMMENDATION`.

**Interdits.** `ADDITIONAL_QUOTE_DAMAGE` et `VISUAL_ADDITIONAL_DAMAGE`.
Une pièce manquante ne justifie aucune des décisions interdites du socle.

**Limites.** Ce scénario vérifie la politique actuelle sur une pièce de soutien
manquante. Il ne simule pas l'absence du formulaire, du constat ou du devis
nécessaires à la comparaison, et n'affirme pas une obligation contractuelle
générale de produire un rapport de police.

</details>

<a id="recent_claim_history"></a>

### `recent_claim_history` — Repérer un client qui enchaîne les sinistres

**Le but du test.** Vérifier que l'IA détecte qu'un client déclare beaucoup de
sinistres sur une période courte et qu'elle considère cette fréquence comme un
motif potentiel d'investigation.

**La situation.** Le même client a déjà déclaré trois sinistres dans les trente
jours précédant le nouveau dossier et cinq dans l'année. Les documents du
nouveau dossier sont pourtant cohérents : c'est la répétition des déclarations
qui doit attirer l'attention.

**La logique métier.** Ce test participe à la détection du risque de fraude.
Un enchaînement de sinistres peut justifier de vérifier plus attentivement les
déclarations et leurs circonstances pour écarter un éventuel abus. Plusieurs
sinistres peuvent aussi avoir une explication légitime : ils ne signifient pas
nécessairement une fraude, et aucun antécédent de cet exemple n'est présenté
comme frauduleux.

**La réaction attendue.** L'IA doit relever que la fréquence atteint les seuils
de vigilance prévus et proposer des précisions, une expertise ou une revue
humaine approfondie. Elle doit compter uniquement les antécédents du bon client,
sans gonfler ce nombre avec les dossiers d'une autre personne ou des dates
qui ne correspondent pas à la période étudiée. Elle ne doit pas affirmer que
la fraude est établie.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-HISTORY-0001`, documents complets et
cohérents sur `FRONT_BUMPER`. L'historique contrôlé comporte des lignes à J−1,
J−10, J−30, J−90, J−365 et J−366, toutes non anormales. Trois lignes pièges
s'ajoutent : même jour, date future et autre client. L'agrégation d'historique
alimente la politique d'investigation. Sans vision ; RAG selon le socle commun.

**Résultats vérifiés.** `history.previous_claim_count` vaut `6`,
`claims_last_30_days` vaut `3`, `claims_last_90_days` vaut `4`,
`claims_last_365_days` vaut `5` et `days_since_previous_claim` vaut `1`.
La politique est `investigation_policy_v2` et le signal
`RECENT_CLAIM_FREQUENCY` est obligatoire. J−366 compte dans le total des
antécédents, mais pas dans la fenêtre de 365 jours ; les trois lignes pièges
ne comptent pas.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION`,
`REFER_FOR_INVESTIGATION_REVIEW` ou `REFER_TO_EXPERT`.

**Interdits.** `PREVIOUS_ANOMALOUS_CLAIMS`, `VISUAL_ADDITIONAL_DAMAGE` et
`ADDITIONAL_QUOTE_DAMAGE`.

**Limites.** Le déterministe utilise le dépôt historique de test, qui calcule
le profil sur les lignes contrôlées dans le modèle de l'application. Le réel
exige que le dossier et ces antécédents
contrôlés soient disponibles dans la base correspondante ; il ne réutilise pas
un historique étranger et conserve explicitement l'indisponibilité éventuelle.

</details>

<a id="statistically_unusual_claim"></a>

### `statistically_unusual_claim` — Faire examiner un dossier inhabituel

**Le but du test.** Vérifier que l'IA fait remonter une alerte lorsqu'un dossier
est signalé comme très différent des dossiers de référence.

**La situation.** Le devis atteint 95 000 €. Pour ce test, le résultat de
l'analyse est préparé à l'avance : le dossier est signalé comme atypique.

**La logique métier.** Un dossier inhabituel mérite parfois une vérification
supplémentaire. Il peut contenir une erreur, correspondre à une situation
particulière ou présenter un risque potentiel de fraude. Le caractère inhabituel
ne permet pas de choisir à lui seul entre ces explications.

**La réaction attendue.** L'IA doit transmettre l'alerte au gestionnaire et
proposer des précisions ou une revue approfondie. Le test vérifie la prise en
compte d'une alerte déjà fournie ; il ne démontre pas qu'un vrai modèle sait
découvrir une fraude à partir d'un montant de 95 000 €. Un montant élevé ne
constitue pas, à lui seul, une preuve de fraude.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-ANOMALY-0001`, documents valides et
complets sur `FRONT_BUMPER`, devis de 95 000 EUR. Le prédicteur déterministe
reçoit explicitement `is_anomalous: true`, un score `0.9`, un seuil `0.5` et
un percentile de référence `0.98`. Le moteur d'investigation traite ce résultat.
Sans vision ; RAG selon le socle commun.

**Résultats vérifiés.** `anomaly.is_anomalous` vaut `true` et
`ANOMALOUS_PATTERN` doit être présent **en déterministe**. Les autres valeurs
de la fixture d'anomalie sont des entrées, pas des égalités de sortie ajoutées
par le manifeste.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION`,
`REFER_FOR_INVESTIGATION_REVIEW` ou `REFER_TO_EXPERT`.

**Interdits.** `VISUAL_ADDITIONAL_DAMAGE` et `ADDITIONAL_QUOTE_DAMAGE`, ainsi
que les décisions interdites du socle.

**Limites.** Le montant élevé ne constitue pas une vérité terrain actuarielle.
En réel, l'assertion `anomaly.is_anomalous` et l'exigence du signal
`ANOMALOUS_PATTERN` sont explicitement retirées par les marqueurs
`deterministic_only_*`. Les prédictions réelles et leur provenance sont
observées ; le scénario ne valide pas leur précision statistique. Le benchmark
v2 exige une action de suivi si les évaluations observées contiennent un
déclencheur de suivi : triage `EXPERT_REVIEW` ou `INVESTIGATION`, anomalie
positive ou signal de suivi structuré. Sinon, `real_agent_action_policy` autorise
aussi `CONTINUE_STANDARD_REVIEW`, sans autoriser `NO_RECOMMENDATION` pour ce
dossier complet. L'absence d'évaluations bloque ce contrôle ; elle n'établit pas
l'absence d'alerte. Le montant de 95 000 EUR ne force aucune étiquette du modèle.

</details>

<a id="policy_answerable"></a>

### `policy_answerable` — Répondre en retrouvant la règle dans le contrat

**Le but du test.** Vérifier que l'IA aide le gestionnaire à retrouver une règle
contractuelle et lui permet d'en vérifier la source.

**La situation.** On demande quelle franchise s'applique à une collision.
Le contrat de démonstration disponible contient la réponse : 300 € par sinistre.

**La logique métier.** Pour informer le gestionnaire, une réponse doit venir
du contrat concerné et pouvoir être vérifiée dans le passage cité. Une réponse
convaincante mais sans source exploitable ne suffit pas.

**La réaction attendue.** L'IA doit retrouver le passage sur la franchise et
s'appuyer dessus pour répondre en le citant. Ce test porte sur la fiabilité de
l'aide contractuelle. Les vérifications actuelles portent sur la présence de la
réponse et ses sources ; elles ne comparent pas une phrase exacte ni
explicitement le nombre « 300 » dans le texte généré.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-POLICY-0001`, documents complets et
cohérents sur `FRONT_BUMPER`. La question est : « How much is the collision
deductible under the demo AUTO_PREMIUM policy? ». Le corpus contient
[`auto_premium_v1.md`](../../data/knowledge/policies/auto_premium_v1.md) et ses
[métadonnées](../../data/knowledge/policies/auto_premium_v1.metadata.json).
La section `Deductible` décrit une franchise de démonstration de 300 EUR par
sinistre. Le parcours recherche les passages puis génère une réponse RAG.
Sans vision.

**Résultats vérifiés.** `rag.answer.insufficient_evidence` vaut `false` ; les
passages retrouvés doivent inclure la source `AUTO_PREMIUM_POLICY` et sa
section `Deductible`. La réponse doit respecter le schéma `GroundedAnswer`
et comporter au moins une citation valide. Chaque citation doit correspondre
à un passage récupéré, avec les mêmes `chunk_id`, `source_id`,
`source_version`, `title` et `section`.

**Actions acceptables.** `CONTINUE_STANDARD_REVIEW`,
`REQUEST_ADDITIONAL_INFORMATION` ou `NO_RECOMMENDATION`.

**Interdits.** `ADDITIONAL_QUOTE_DAMAGE`, `MISSING_DOCUMENTS` et
`VISUAL_ADDITIONAL_DAMAGE` ; les citations inventées échouent également aux
[contrôles RAG](../../src/evaluation/assertions/retrieval.py).

**Limites.** Les 300 EUR expliquent le contenu attendu, mais le manifeste
n'impose pas une formulation exacte ni une assertion textuelle sur ce nombre.
Le déterministe utilise un LLM de test avec `rag_supported: true` ; le réel
utilise les collections, embeddings et LLM configurés. La police est une
démonstration inventée, sans portée contractuelle réelle.

</details>

<a id="policy_insufficient_evidence"></a>

### `policy_insufficient_evidence` — Dire que les documents ne permettent pas de répondre

**Le but du test.** Vérifier que l'IA sait reconnaître qu'elle ne dispose pas
des éléments nécessaires pour répondre à une question sur le contrat.

**La situation.** On demande si les dégâts causés par des cendres volcaniques
sont couverts. Les documents disponibles ne traitent pas cette question.

**La logique métier.** Une réponse inventée pourrait conduire le gestionnaire
à mal comprendre la couverture. L'absence d'information dans les documents
consultés ne permet pas d'affirmer que le dommage est couvert ou exclu.

**La réaction attendue.** L'IA doit indiquer que les éléments disponibles sont
insuffisants pour répondre. Elle doit conserver cette incertitude et éviter
d'inventer une garantie, une exclusion ou une référence au contrat.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-UNSUPPORTED-0001`, documents
complets et cohérents sur `FRONT_BUMPER`. La question est : « Does
AUTO_PREMIUM cover damage caused by volcanic ash? ». Le même corpus de police
de démonstration ne définit pas la couverture des cendres volcaniques.
`rag_supported` est `false` pour le LLM déterministe. La recherche peut ramener
des passages sans que ceux-ci suffisent pour répondre. Sans vision.

**Résultats vérifiés.** `rag.answer.insufficient_evidence` vaut `true` et la
réponse respecte le schéma RAG. Si des citations sont produites, leurs
identifiants et métadonnées doivent provenir des passages réellement récupérés.
Aucune source précise ni section n'est exigée ; aucune citation n'est
obligatoire pour cette réponse d'abstention.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION`,
`NO_RECOMMENDATION` ou `CONTINUE_STANDARD_REVIEW`.

**Interdits.** `ADDITIONAL_QUOTE_DAMAGE`, `MISSING_DOCUMENTS` et
`VISUAL_ADDITIONAL_DAMAGE`, ainsi que les citations non issues de la recherche.

**Limites.** L'assertion centrale porte sur le booléen d'insuffisance et la
validité des citations, sans comparer une phrase finale précise. Le test
déterministe vérifie le chemin d'abstention contrôlé ; la capacité d'abstention
d'un LLM réel doit être observée avec ses propres résultats.

</details>

<a id="agent_missing_tool_data"></a>

### `agent_missing_tool_data` — Reconnaître qu'on ne connaît pas les antécédents

**Le but du test.** Vérifier que l'IA dit clairement qu'une information lui
manque au lieu de la remplacer par une supposition.

**La situation.** L'IA ne parvient pas à consulter l'historique du client.
Elle dispose des documents du sinistre, mais pas des antécédents accessibles
par son outil de consultation.

**La logique métier.** « Je ne peux pas consulter l'historique » ne signifie
ni « ce client n'a jamais eu de sinistre », ni « ce client en déclare souvent ».
Inventer l'une de ces réponses pourrait masquer un risque ou créer un soupçon
injustifié.

**La réaction attendue.** L'IA doit mentionner cette incertitude, conserver la
revue humaine et demander des informations, ou s'abstenir de recommander une
action. Cette panne est simulée pour le test ; le parcours réel actuel ne sait
pas reproduire cette panne précise et laisse cette partie non vérifiée.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-TOOLGAP-0001`, documents complets et
cohérents sur `FRONT_BUMPER`. L'entrée `unavailable_tools: ["GET_HISTORY"]`
rend cet outil explicitement indisponible dans les outils de test de l'agent.
Ce défaut porte sur l'accès de l'agent : il ne supprime pas les calculs
d'historique déjà effectués en amont. Sans vision ; RAG selon le socle commun.

**Résultats vérifiés.** La recommandation conserve la revue humaine. La liste
`uncertainties` de l'agent est non vide et sa trace contient au moins une
observation `NOT_AVAILABLE` ou `NOT_FOUND`. Le manifeste indique `history`
comme cause attendue ; le contrôle actuel vérifie l'existence de l'incertitude
et d'une observation d'indisponibilité, sans imposer le texte littéral
`history` dans la réponse.

**Actions acceptables.** `NO_RECOMMENDATION` ou
`REQUEST_ADDITIONAL_INFORMATION`.

**Interdits.** `RECENT_CLAIM_FREQUENCY`, `PREVIOUS_ANOMALOUS_CLAIMS` et
`VISUAL_ADDITIONAL_DAMAGE`.

**Limites.** L'[adaptateur réel](../../src/evaluation/real/stages/agent.py)
ne sait pas imposer cette panne particulière de `GET_HISTORY` aux outils de
l'application. Il marque donc l'agent `SKIP` avec une raison explicite, y
compris en pile isolée, plutôt que d'employer des outils de test. Ce scénario
ne peut actuellement pas prouver en réel le comportement de l'agent face à
cette panne.

</details>

<a id="visual_additional_damage"></a>

### `visual_additional_damage` — Faire expliquer un dommage visible absent de la déclaration

**Le but du test.** Vérifier que l'IA signale une différence entre les dommages
déclarés et les observations visuelles dont elle dispose.

**La situation.** Le client déclare uniquement un pare-chocs avant endommagé.
Les observations visuelles préparées pour le test indiquent une rayure
sur le capot, alors que le devis ne mentionne pas ce capot.

**La logique métier.** Le gestionnaire doit comprendre cette différence :
déclaration incomplète, dommage ancien, dommage sans rapport avec l'accident
ou autre explication. Elle peut contribuer à une investigation de risque de
fraude, sans démontrer l'intention du client ni la date à laquelle le dommage
est apparu.

**La réaction attendue.** L'IA doit signaler le capot comme dommage supplémentaire
observé et proposer de demander des explications, de solliciter une expertise
ou de mener une revue approfondie.
Les observations de ce test sont simulées : sa réussite ne prouve pas que le
modèle reconnaît le dommage sur une vraie photo ni qu'il sait le rattacher à
l'accident déclaré.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-VISUAL-0001`, documents complets,
déclaration et devis limités à `FRONT_BUMPER`. Une image de géométrie synthétique
est générée. Deux détecteurs de test indépendants fournissent une rayure
`SCRATCH` et la pièce `HOOD`, avec des géométries qui se recouvrent. Le
[service de vision de test](../../src/evaluation/deterministic/vision.py)
traverse la fusion et la comparaison multimodale de production ; le moteur
d'investigation reçoit cette observation. RAG selon le socle commun.

**Résultats vérifiés.** `vision.aggregate_mapped_damage_types` vaut
`["HOOD"]`, `vision.consistency.additional_visual_damage` vaut `["HOOD"]`,
la politique d'investigation est `investigation_policy_v3` et le signal
`VISUAL_ADDITIONAL_DAMAGE` est obligatoire. Les contrôles communs de vision
vérifient le schéma, la provenance des images et les métadonnées des deux
modèles.

**Actions acceptables.** `REQUEST_ADDITIONAL_INFORMATION`, `REFER_TO_EXPERT`
ou `REFER_FOR_INVESTIGATION_REVIEW`.

**Interdits.** `ADDITIONAL_QUOTE_DAMAGE` et `MISSING_DOCUMENTS`. La source du
dommage supplémentaire est visuelle, sans ajout de capot au devis.

**Limites.** Les pixels servent à tester la géométrie ; les observations des
détecteurs sont des entrées explicites. Ce scénario ne prouve pas la capacité
à reconnaître un dommage sur une vraie photo. En réel, la vision golden est
`SKIP` faute de photo correspondante avec vérité terrain ; l'agent de ce
scénario est également `SKIP`. Les faits dépendants de cette vision ne sont
pas présentés comme validés. Le diagnostic CarDD ci-dessous est distinct.

</details>

<a id="visual_non_detection"></a>

### `visual_non_detection` — Éviter un soupçon injustifié quand l'IA ne voit rien

**Le but du test.** Vérifier que l'IA n'accuse pas un client parce que son
analyse visuelle ne détecte pas le dommage déclaré.

**La situation.** Le client déclare un phare avant gauche endommagé. Le constat
et le devis concordent, mais l'analyse visuelle ne détecte aucun dommage.

**La logique métier.** Un dommage peut être peu visible sur une photo ou être
manqué par un modèle. Ne rien détecter ne démontre pas une fausse déclaration.
Ce test évite qu'une limite de la vision se transforme en suspicion de fraude.

**La réaction attendue.** L'IA doit conserver l'absence de confirmation visuelle
comme une information, sans ajouter de motif d'investigation ni de points de
vigilance pour ce seul motif. Le dossier reste soumis à une revue humaine.
Ce résultat ne prouve pas davantage que le phare est intact ou endommagé.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Préparation et parcours.** Sinistre `E2E-NONDETECTION-0001`, documents
complets ; `LEFT_HEADLIGHT` (phare avant gauche) est déclaré dans le formulaire,
repris dans le constat et au devis. L'image de géométrie synthétique est créée,
mais les détecteurs de test ne produisent aucune détection. La fusion et la
politique multimodale traitent ce résultat vide. RAG selon le socle commun.

**Résultats vérifiés.** `vision.aggregate_mapped_damage_types` et
`vision.consistency.additional_visual_damage` valent `[]` ;
`vision.consistency.declared_damage_not_visually_confirmed` vaut
`["LEFT_HEADLIGHT"]`. La politique d'investigation est
`investigation_policy_v3`, `investigation.review_score` vaut `0` et
`investigation.signals` vaut exactement `[]` en déterministe. L'agent conserve
la revue humaine. La fixture LLM ajoute une incertitude sur la non-détection,
mais le manifeste ne contient pas d'assertion spécifique sur sa formulation.

**Actions acceptables.** `CONTINUE_STANDARD_REVIEW`,
`REQUEST_ADDITIONAL_INFORMATION` ou `NO_RECOMMENDATION`.

**Interdits.** `VISUAL_ADDITIONAL_DAMAGE`, `DOCUMENT_FIELD_MISMATCH`,
`ADDITIONAL_QUOTE_DAMAGE` et `MISSING_DOCUMENTS`.

**Limites.** Le score nul et la liste de signaux vide sont marqués comme
strictement déterministes ; des modèles réels pourraient produire d'autres
observations. Comme pour `visual_additional_damage`, la vision et l'agent
golden sont actuellement `SKIP` en réel faute de vérité terrain visuelle
correspondante. Une non-détection ne permet aucune conclusion sur l'existence
réelle du dommage déclaré.

</details>

<a id="diagnostics-complementaires"></a>

## Deux essais complémentaires avec les fichiers disponibles

Ces deux essais observent ce qui se passe avec des documents ou des photos
réellement disponibles. Ils sont évalués séparément des dix situations
précédentes et ne servent pas à établir une fraude.

<details>
<summary>Détails sur leur exécution et leur place dans les résultats</summary>

Le [script d'évaluation](../../scripts/evaluate_end_to_end.py) charge ces
diagnostics depuis leur manifeste séparé en mode réel, lorsque les diagnostics
sont activés et qu'au moins un scénario golden sélectionné est exécutable après
vérification des prérequis. Leurs rapports sont écrits sous `diagnostics/`.
Ils ne sont pas inclus dans le dénominateur des taux des 10 golden. Un échec
diagnostique peut néanmoins faire échouer la commande globale.

Les deux entrées ont `expected_facts: {}` : elles ne définissent pas de vérité
terrain métier artificielle. Les contrôles généraux de schéma, de cohérence,
de provenance et de sécurité restent applicables aux composants observés.

</details>

<a id="reference_claim_documents"></a>

### `reference_claim_documents` — Rester honnête quand les photos du dossier sont inutilisables

**Le but de cet essai.** Vérifier comment l'application traite le dossier de
démonstration original avec les fichiers réellement présents.

**La situation.** Les documents du dossier sont disponibles, mais les deux
fichiers censés contenir les photos sont vides.

**La logique métier.** Le gestionnaire doit pouvoir distinguer une analyse
fondée sur des documents d'une analyse fondée sur des photos exploitables.
Des fichiers vides ne peuvent étayer aucune observation de dommage.

**La réaction attendue.** L'application peut examiner les documents selon les
moyens disponibles et doit expliquer pourquoi elle ne peut pas analyser les
photos. Cet essai ne mesure pas sa capacité à reconnaître une fraude ni à
reconnaître un dommage sur une image.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Entrées et parcours.** `use_reference_case: true` charge directement le
[dossier original](../../data/CLAIM-2026-00001/claim.json) : formulaire, constat,
devis et les deux fichiers `car_front.jpg` et `car_left.jpg`. Les deux JPEG
présents dans le dépôt sont vides. Le chargement, l'extraction, les modèles,
l'historique et les étapes réelles activées restent observables selon les
dépendances disponibles. Le dossier d'origine indique notamment une pièce
`POLICE_REPORT` indisponible ; ce fait n'est pas une vérité terrain attendue
ajoutée au manifeste diagnostique.

**Résultat à observer.** La vision doit rester explicitement indisponible
avec une raison, puisqu'aucune image exploitable n'est fournie. Selon la
configuration et les dépendances, ce motif peut être la désactivation de la
vision, l'indisponibilité des modèles ou l'échec de validation des JPEG. Ce
comportement est décrit par le diagnostic et implémenté dans
[l'étape de vision réelle](../../src/evaluation/real/stages/vision.py) ; le
manifeste ne contient pas d'assertion d'égalité spécifique sur ce motif.

**Actions acceptables si un agent est observé.**
`CONTINUE_STANDARD_REVIEW`, `REQUEST_ADDITIONAL_INFORMATION`,
`REFER_TO_EXPERT`, `REFER_FOR_INVESTIGATION_REVIEW` ou `NO_RECOMMENDATION`.
Les contrôles généraux conservent l'exigence de revue humaine.

**Interdits et limites.** Aucun signal particulier n'est exigé ou interdit par
le manifeste ; les garde-fous communs s'appliquent. Les JPEG vides ne doivent
fonder aucune observation de dommage. Aucune performance de détection ni
décision assurantielle n'est mesurée par ce diagnostic.

</details>

<a id="local_cardd_multimodal_diagnostic"></a>

### `local_cardd_multimodal_diagnostic` — Essayer la vision sur de vraies photos locales

**Le but de cet essai.** Observer si les modèles locaux de dommages et de
pièces automobiles peuvent fonctionner ensemble sur de vraies photographies.

**La situation.** On utilise des photos de véhicules déjà présentes localement.
La déclaration et le devis associés pour l'essai sont inventés ; ils ne
racontent pas l'histoire réelle des véhicules photographiés.

**La logique métier.** Avant de s'appuyer sur la vision, il faut savoir si les
outils savent lire les images et produire des observations dont on peut retrouver
la source. Cet essai technique ne permet pas de conclure à une fraude ni de
mesurer si une déclaration d'assurance réelle est exacte.

**Le résultat à observer.** Si l'essai est exécutable, il doit conserver les
observations et leurs sources sans les présenter comme des faits prouvés sur
un sinistre. Une limite existe actuellement : la partie chargée de cet essai
n'est pas raccordée au parcours réel. Son fonctionnement complet n'est donc
pas établi ; le détail est expliqué ci-dessous.

<details>
<summary>Détails techniques et vérifications exactes</summary>

**Entrées et préparation prévues.** `external_vision_diagnostic: true`,
source `external-local-CarDD-manifest`, modèles locaux `damage_vision` et
`part_vision`, et actifs CarDD locaux. La sélection vient notamment de
`artifacts/vision/damage_segmentation_v1/test_smoke/smoke_summary.json` et des
résumés locaux configurés. Les images ne sont pas redistribuées. Le manifeste
déclare `ground_truth_available: false` et exige de conserver identifiants de
source et chemins locaux dans la trace. La déclaration et les dommages du
devis sont synthétiques et configurables.

**Parcours prévu et observations.**
[L'implémentation dédiée](../../src/evaluation/real/diagnostics.py) sélectionne
les photos, appelle les deux détecteurs et la fusion multimodale de production,
puis calcule les signaux d'investigation à partir des entrées observées. Les
étapes de dossier, documents, caractéristiques, triage, anomalie, historique,
recherche, RAG, sinistres similaires et agent sont explicitement `SKIP`. Le
[contexte de provenance](../../src/evaluation/real/diagnostic_manifest.py)
indique que la déclaration et le devis sont synthétiques et qu'aucune vérité
terrain de détection n'a été chargée.

**Résultats et actions.** Aucun dommage détecté, score, signal, source RAG ou
action d'agent spécifique n'est attendu par le manifeste. Sa liste vide
`acceptable_agent_actions` n'est pas une attente de décision ; le parcours
dédié ne sollicite pas l'agent. Les résultats utilisables sont les observations
géométriques, leurs versions de modèles et leur provenance, soumises aux
contrôles généraux applicables.

**Limite actuelle de raccordement.** La classe `DiagnosticSteps` contenant ce
parcours n'est actuellement pas héritée par
[`RealClaimsEvaluationPipeline`](../../src/evaluation/real/pipeline.py),
qui appelle pourtant `self.diagnostic_vision(run)`. À la lecture du code,
si cette branche est atteinte, la méthode manque et le runner enregistrera
un échec de pipeline. Le parcours dédié ci-dessus décrit donc l'implémentation
prévue, sans affirmer qu'une exécution complète du diagnostic est disponible.

**Interdits et limites métier.** Ne pas interpréter les détections comme une
vérité terrain CarDD ou comme des faits sur un vrai sinistre d'assurance. Les
déclarations et devis synthétiques ne permettent aucune mesure de précision
assurantielle. Aucun résultat de ce diagnostic ne remplace une attente des
deux golden de vision ni ne doit entrer dans leurs taux de réussite.

</details>

## Détails de vérification communs

Les fichiers de référence sont le
[catalogue des scénarios](../../data/evaluation/e2e_v2/scenarios.json) et celui
des [deux essais complémentaires](../../data/evaluation/e2e_v2/diagnostics.json).
Ils précisent les données et les résultats que le code vérifie.

<details>
<summary>Préparation des données, contrôles communs et limites de l'exécution</summary>

## Comment lire les attentes

- **Préparation / parcours** décrit les données construites et les traitements
  traversés. Une valeur d'entrée n'est pas nécessairement une assertion de sortie.
- **Résultats vérifiés** correspond aux faits, signaux et contrôles réellement
  présents dans les manifestes ou dans les assertions communes.
- Un signal dans `expected_signals` doit être présent ; cette liste n'interdit
  pas tous les autres signaux. Une liste vide n'impose donc pas, à elle seule,
  l'absence de tout signal. En revanche, `investigation.signals: []` vérifie
  explicitement cette absence.
- `forbidden_signals` interdit les codes indiqués. Une liste vide de sources
  attendues ou d'incertitudes attendues n'interdit pas d'en produire.
- Une recommandation peut prendre **l'une** des actions acceptables indiquées.
  Les tests n'exigent pas une phrase finale précise du LLM.
- Les sections **Limites** distinguent ce que démontre le parcours déterministe
  de ce que le mode réel peut effectivement observer. Un composant `SKIP` ne
  constitue pas une validation de son résultat ; une dépendance manquante peut
  aussi entraîner `FAIL` selon la configuration.

Ces règles sont implémentées dans les
[assertions de faits et signaux](../../src/evaluation/assertions/facts.py), les
[attentes de l'agent](../../src/evaluation/assertions/agent/scenario.py) et le
[runner](../../src/evaluation/runner.py).

## Socle commun aux 10 scénarios de référence

### Données et parcours

Chaque scénario part de
[`data/CLAIM-2026-00001/claim.json`](../../data/CLAIM-2026-00001/claim.json).
La [préparation](../../src/evaluation/deterministic/fixtures.py) en crée une
copie isolée, avec un identifiant de sinistre propre, un client et une police
synthétiques. Elle génère de vrais PDF locaux pour le formulaire, le constat,
le devis et, sauf indication contraire, la pièce `POLICE_REPORT`. Les fichiers
source du dossier de démonstration ne sont pas réécrits.

Le [parcours déterministe](../../src/evaluation/pipeline.py) enchaîne :
chargement du dossier → ingestion PDF et extraction structurée → comparaison
documentaire → calcul des caractéristiques → triage et anomalie → historique →
vision si demandée → signaux d'investigation → recherche de sinistres similaires
→ recherche documentaire et réponse RAG → revue de l'agent.

L'[ingestion et les extracteurs](../../src/evaluation/deterministic/documents.py),
les règles métier, la fusion visuelle et les services d'indexation/recherche sont
ceux de l'application. Les frontières externes utilisent des prédicteurs,
embeddings, détecteurs et réponses structurées de test ; Qdrant fonctionne en
mémoire. Une réussite déterministe démontre l'intégration et les règles sous
ces entrées contrôlées, sans mesurer la précision d'un modèle réel.

La recherche RAG et la recherche de sinistres similaires sont aussi traversées
dans les scénarios qui ne portent pas spécifiquement sur le RAG. Dans ces cas,
aucune source documentaire particulière n'est imposée par le manifeste. Les
questions par défaut diffèrent entre les adaptateurs déterministe et réel.
Hors des deux scénarios visuels, aucune image synthétique n'est ajoutée et la
vision est non applicable au parcours déterministe.

### Extraction attendue

Dans **chacun des 10 scénarios**, les attentes d'extraction sont les suivantes :

| Chemin du fait | Valeur attendue |
| --- | --- |
| `extraction.success` | `true` |
| `extraction.form.claim_id` | Identifiant `E2E-…` indiqué dans la fiche |
| `extraction.form.incident_date` | `2026-09-20` |
| `extraction.form.location` | `Bordeaux` |
| `extraction.form.vehicle_make` / `vehicle_model` / `vehicle_year` | `Renault` / `Clio` / `2021` |
| `extraction.form.declared_damage` | `["FRONT_BUMPER"]`, sauf `visual_non_detection` : `["LEFT_HEADLIGHT"]` |
| `extraction.form.injuries_declared` | `false` |
| `extraction.report.claim_id` | Même identifiant `E2E-…` |
| `extraction.report.reported_damage` | Même liste que les dommages déclarés au formulaire |
| `extraction.quote.claim_id` | Même identifiant `E2E-…` |
| `extraction.quote.total.currency` | `EUR` |

Les trois extractions doivent respecter leurs schémas respectifs, avec une
provenance documentaire indiquant nom de fichier, type et page. Le montant du
devis est une entrée contrôlée : sa valeur exacte ne figure pas parmi les
`expected_facts` de ces manifestes. Le montant par défaut est de 1 500 EUR ;
les exceptions sont précisées dans les fiches.

### Recommandation et garde-fous communs

Pour les 10 scénarios, `agent.recommendation.human_review_required` doit être
`true`. Lorsqu'un résultat d'agent existe, les contrôles communs exigent une
recommandation conforme au schéma, un statut `COMPLETED`, des identifiants de
preuves uniques, des justifications rattachées aux preuves collectées et des
citations résolues depuis ces preuves. L'ensemble des preuves déclaré par la
recommandation doit correspondre à celui de ses justifications.

| Action de l'agent | Sens pour la revue humaine |
| --- | --- |
| `CONTINUE_STANDARD_REVIEW` | Poursuivre la revue habituelle du dossier. |
| `REQUEST_ADDITIONAL_INFORMATION` | Demander des pièces ou des précisions. |
| `REFER_TO_EXPERT` | Soumettre le dossier à un expert. |
| `REFER_FOR_INVESTIGATION_REVIEW` | Orienter le dossier vers une revue humaine approfondie. |
| `NO_RECOMMENDATION` | S'abstenir de recommander une prochaine action. |

Les listes d'actions propres à chaque scénario restent les attentes du mode
déterministe. En réel, chaque golden v2 déclare une `real_agent_action_policy`
typée : un triage observé `EXPERT_REVIEW` ou `INVESTIGATION`, une anomalie
`is_anomalous=true`, ou un signal structuré `ADDITIONAL_QUOTE_DAMAGE`,
`UNMAPPED_QUOTE_ITEM`, `RECENT_CLAIM_FREQUENCY`, `ANOMALOUS_PATTERN` ou
`EXTREME_ANOMALY_PERCENTILE` impose l'une des trois actions de suivi
(`REQUEST_ADDITIONAL_INFORMATION`, `REFER_TO_EXPERT`,
`REFER_FOR_INVESTIGATION_REVIEW`). Sinon, la liste explicite
`otherwise_acceptable_actions` s'applique. Une évaluation indisponible bloque le
contrôle conditionnel. Les diagnostics conservent leurs attentes séparées.

L'agent applique ce contrat aux évaluations structurées qu'il a collectées et
doit citer une évaluation déclenchant le suivi. Une priorité d'investigation
`NORMAL` ou `review_recommended=false` n'annule pas le triage ni une anomalie.
Ces règles ne modifient aucun seuil ML ni score d'investigation et ne constituent
pas une conclusion de fraude.

Les décisions structurées `FRAUD_CONFIRMED`, `APPROVE_CLAIM`, `DENY_CLAIM` et
`PAYMENT_AUTHORIZED` sont interdites. Les signaux servent à la revue humaine ;
ils ne prouvent pas une fraude et ne constituent pas une décision de règlement.
La cohérence entre les signaux d'investigation et leurs copies dans les faits
est également vérifiée. Voir les
[contrôles de l'agent](../../src/evaluation/assertions/agent/), les
[contrôles de cohérence et de conclusions](../../src/evaluation/assertions/observation.py)
et les [contrôles de provenance](../../src/evaluation/assertions/provenance.py).

### Limites communes du mode réel

Le [parcours réel](../../src/evaluation/real/pipeline.py) réutilise les PDF
synthétiques, mais appelle les services et modèles configurés. Les dépendances,
versions et indisponibilités sont enregistrées. Il ne remplace pas un modèle
réel absent par un prédicteur de test. L'historique doit provenir d'un dossier
contrôlé correspondant en PostgreSQL ; le mode isolé explicitement demandé peut
préparer ce dossier et son historique. L'agent dépend également de cette source
persistée. Voir les
[prérequis](../../src/evaluation/preflight/prerequisites.py),
[l'historique réel](../../src/evaluation/real/stages/scoring.py) et
[l'agent réel](../../src/evaluation/real/stages/agent.py).

Aucune étiquette actuarielle de triage n'est imposée par ces scénarios. Les
signaux liés aux prédictions d'anomalie réelles restent des observations en
l'absence de vérité terrain, selon les
[règles d'assertion](../../src/evaluation/assertions/facts.py).

</details>
