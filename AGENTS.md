# Instructions pour les agents

## Tests end-to-end : demande explicite obligatoire

- Ne jamais lancer les tests end-to-end (E2E, E&E, bout en bout), même pour valider une migration, un nettoyage ou une modification du projet, sauf si l'utilisateur demande explicitement leur exécution.
- Une demande générale de validation, de lancement des tests ou de migration ne constitue pas une autorisation de lancer les tests end-to-end.
- Cette restriction couvre toute la suite `tests/e2e/`, y compris ses tests marqués `unit`, ainsi que les scénarios exécutés par `scripts/evaluate_end_to_end.py` ou une autre commande équivalente.
- Sans autorisation explicite, exclure `tests/e2e/` des commandes pytest générales : utiliser par exemple `python -m pytest --ignore=tests/e2e` ou `python -m pytest -m unit --ignore=tests/e2e`. Le filtre `-m unit` seul ne suffit pas.
- Pour une validation ciblée, sélectionner uniquement des fichiers ou dossiers hors de `tests/e2e/`. Vérifier aussi qu'un script, un sous-processus ou un outil de validation ne déclenche pas indirectement des tests end-to-end.
- Dans le compte rendu, préciser que les tests end-to-end n'ont pas été exécutés faute de demande explicite ; ne pas les présenter comme validés.
