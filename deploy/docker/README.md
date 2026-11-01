# Console Docker

Le `compose.yaml` à la racine regroupe PostgreSQL, Qdrant, le backend (`api`)
et le frontend de démonstration (`frontend_demo`).
Le profil `console` ajoute les deux derniers services. L'API Docker consulte les
analyses exportées ; elle ne lance aucun modèle et ne se connecte pas aux bases.
Le backend complet `api.main:app` reste disponible dans l'environnement Python.

## Démarrer

Prérequis : Docker Desktop avec le moteur Linux actif et un export existant dans
`analysis_results/`, contenant `manifest.json` et ses sous-dossiers. Les exports
ne sont pas versionnés ; sur un autre ordinateur, copier la collection complète
ou un sous-dossier d'analyse autonome avant de lancer les conteneurs.
Python, Node.js et une `.venv` ne sont pas nécessaires sur l'ordinateur hôte.

Depuis la racine du projet :

```powershell
# Console seule : frontend et API, sans démarrer PostgreSQL ni Qdrant.
docker compose up --build -d frontend_demo
```

Ouvrir **http://localhost:8080**. Nginx sert le frontend compilé et transmet les
requêtes `/v1/` à l'API interne. Les identifiants d'analyse dans les URLs restent
conservés, y compris pour les PDF et photos. Le port 8000 de l'API n'est pas
publié sur l'hôte. Les exports sont montés en lecture seule et conservés lors de
l'arrêt ou de la reconstruction des conteneurs.

Pour démarrer aussi les services habituels PostgreSQL et Qdrant :

```powershell
docker compose --profile console up --build -d
```

Cette commande démarre les quatre services. Elle ne relie pas l'API de lecture
aux bases et ne lance ni migration, ni import, ni nouvelle analyse. Les services
`postgres-test` et `qdrant-test` restent réservés au profil `test`.

## Choisir le dossier ou le port

```powershell
$env:CLAIMS_ANALYSIS_DIR = 'C:\chemin\vers\mes-analyses'
$env:CLAIMS_REVIEW_PORT = '8081'
docker compose up --build -d frontend_demo
```

Le dossier choisi peut être une collection ou une seule analyse autonome. Il doit
déjà exister. Un manifeste absent, vide ou illisible empêche le démarrage de
l'API ; aucun exemple ni résultat n'est créé automatiquement. Une pièce absente
reste signalée dans la revue. Pour revenir aux valeurs par défaut dans PowerShell :

```powershell
Remove-Item Env:CLAIMS_ANALYSIS_DIR -ErrorAction SilentlyContinue
Remove-Item Env:CLAIMS_REVIEW_PORT -ErrorAction SilentlyContinue
```

## État, journaux et arrêt

```powershell
docker compose ps
docker compose logs api frontend_demo
# Recharger le catalogue après ajout d'un export :
docker compose restart api
# Arrêter uniquement la console, en laissant les bases telles quelles :
docker compose stop frontend_demo api
```

L'API utilise `/readyz` pour sa disponibilité, sans charger de modèle. Nginx attend
une API saine avant de démarrer et résout son nom Docker après sa recréation.
Une panne de connexion produit le message « Backend indisponible » ; une erreur
HTTP du backend conserve son statut et son explication.

## Construction

Le frontend est construit avec Node.js et Python 3.12 pour les traductions TOML,
puis seuls ses fichiers compilés sont copiés dans Nginx. Le compilateur utilise
`CLAIMS_I18N_PYTHON` dans l'image ; en développement local, la `.venv` du projet
reste le choix par défaut.

L'API utilise les contraintes FastAPI, Pydantic et SQLAlchemy déjà présentes dans
`pyproject.toml` ; Uvicorn provient de l'extra FastAPI existant. L'image API
n'installe pas les dépendances ML. Les modèles, données, exports, secrets, Git,
métadonnées de packaging et environnements locaux sont exclus du contexte de
construction. Les deux services
s'exécutent sans utilisateur root, avec un système de fichiers en lecture seule
et un répertoire temporaire séparé. Aucun téléchargement de modèle ni aucune
inférence n'a lieu au démarrage.
