# Tests end-to-end

Vérifier le parcours d'un dossier, les recommandations et les preuves citées.
L'agent aide à la revue humaine. Il ne décide ni du règlement ni d'une fraude.

## Toolchain

- Python 3.12+, environnement `.venv`, dépendances du projet et pytest.
- Installation : [guide](../../docs/INSTALLATION.md).
- Suite pytest : données contrôlées et stockage en mémoire, sans services externes.
- Mode réel : services et modèles locaux configurés dans le [guide d'évaluation](../../docs/END_TO_END_EVALUATION.md).

## Commandes

Depuis la racine du dépôt. **Exécution uniquement sur demande explicite**, même
pour les tests marqués `unit` : [consignes](../../AGENTS.md).

```powershell
# Suite E2E hors ligne.
.\.venv\Scripts\python.exe -m pytest tests/e2e -q

# Exporter une évaluation déterministe.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode deterministic --output-dir artifacts/Evaluations/e2e_v2/deterministic
```

Sans demande E2E, exclure ce dossier :

```powershell
.\.venv\Scripts\python.exe -m pytest --ignore=tests/e2e
```

## Couverture

- [Scénarios métier](SCENARIOS.md) : dix dossiers de référence et diagnostics complémentaires.
- [Manifeste](../../data/evaluation/e2e_v2/scenarios.json) : données et attentes exécutables.
- [Sélection des scénarios](test_scenarios.py) : versions, fichiers, identifiants et filtres.
- [Tests déterministes](deterministic/README.md) : recommandations, limites métier et traçabilité.
- [Tests des rapports](reporting/README.md) : artefacts, métriques et régressions.
- [Contrats du mode réel](test_real_mode_contracts.py) : dépendances ou images de référence absentes.

Les réponses ML, LLM et visuelles de la suite hors ligne sont contrôlées.
Cette suite ne mesure pas la qualité des modèles réels.

## Lire les résultats

- `PASS` : attentes obligatoires satisfaites.
- `FAIL` : attente, composant ou exécution en échec.
- `SKIP` : vérification indisponible ; ne compte pas comme un succès.
- Taux de réussite : `PASS / (PASS + FAIL)`, ou `null` sans scénario évalué.
- Un test qui injecte une erreur réussit lorsque le contrôle la détecte.

Le script conserve les assertions dans `scenario_results.json`, les observations
dans `traces/` et les diagnostics d'échec dans `failures/`.
Voir le [guide des rapports](../../docs/END_TO_END_EVALUATION.md#assertions-metrics-and-artifacts)
pour le mode réel, les baselines et les sorties complètes.
