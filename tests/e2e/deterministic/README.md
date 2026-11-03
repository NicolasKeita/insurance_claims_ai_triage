# Tests E2E déterministes

[Guide E2E](../README.md) · [Scénarios métier](../SCENARIOS.md)

## Toolchain

- Python 3.12+, `.venv`, dépendances du projet et pytest.
- PDF synthétiques, Qdrant en mémoire et réponses ML/LLM/vision contrôlées.
- Services métier réels pour l'extraction, la comparaison et les assertions.
- Aucun service externe ni checkpoint requis.

## Commande

Depuis la racine, **uniquement sur demande E2E explicite**, même avec le marqueur
`unit` : [consignes](../../../AGENTS.md).

```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e/deterministic -q
```

## Contrôles

- [Scénarios](test_scenarios.py) : dix dossiers, signaux attendus et traces du traitement.
- [Contrats](test_contracts.py) : décisions interdites, schémas incomplets et catégories visuelles incohérentes.
- [Preuves](test_grounding.py) : références inventées, fausses citations et réponse contractuelle manquante.
- [Provenance](test_provenance.py) : sources documentaires et versions des modèles.
- Une non-détection visuelle conserve l'incertitude sans ajouter de signal de risque.
- Un signal demande une vérification humaine ; il ne prouve pas une fraude.

Les [fixtures](_support.py) évaluent le [manifeste](../../../data/evaluation/e2e_v2/scenarios.json).
Certains tests modifient ensuite les résultats et vérifient que les assertions
rejettent l'erreur injectée. Ils ne relancent pas le pipeline après modification.

Ces tests valident des comportements contrôlés, pas la qualité des modèles réels.
