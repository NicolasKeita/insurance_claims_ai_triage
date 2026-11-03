# Tests des rapports E2E

[Guide E2E](../README.md) · [Guide d'évaluation](../../../docs/END_TO_END_EVALUATION.md)

## Toolchain

- Python 3.12+, `.venv`, dépendances du projet et pytest.
- Résultats synthétiques construits par les [fixtures](_support.py).
- Fichiers isolés dans les répertoires temporaires de pytest.
- Aucun service externe ni modèle réel exécuté.

## Commandes

Depuis la racine, **uniquement sur demande E2E explicite**, même avec le marqueur
`unit` : [consignes](../../../AGENTS.md).

```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e/reporting -q
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_real_mode_contracts.py -q
```

## Contrôles

- [Artefacts](test_artifacts.py) : valeurs attendues et obtenues, composant et trace source.
- [Baselines](test_baseline.py) : régressions, attentes modifiées et versions des modèles.
- [Métriques](test_metrics.py) : échecs, composants indisponibles et taux de complétion.
- [Contrats du mode réel](../test_real_mode_contracts.py) : pas de prédiction simulée pour remplacer un service absent ; refus des cas visuels sans image de référence annotée.

## Règles de lecture

- `SKIP` ne compte jamais comme un succès.
- Taux de réussite : `PASS / (PASS + FAIL)`, ou `null` sans scénario évalué.
- Une baseline exige un rapport réel, complet et entièrement réussi.
- Une variation de durée seule ne constitue pas une régression de comportement.
- Un rapport synthétique déclaré `real` ne prouve pas l'exécution d'une pile réelle.
- Un test peut réussir en vérifiant le rejet attendu d'un scénario en `FAIL`.

Les contrats du mode réel contrôlent artificiellement l'état des dépendances.
Ils ne mesurent ni la disponibilité des services déployés ni la qualité de la vision.
