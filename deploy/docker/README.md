# Démo statique Docker

Le service `frontend_demo` construit la démo et sert uniquement ses fichiers
statiques dans Nginx. Les quatre dossiers, résultats enregistrés, PDF synthétiques
professionnels et images sont inclus dans l'image. Aucun backend, montage
d'archives, modèle, PostgreSQL ou Qdrant n'est nécessaire pour la consulter.

## Démarrer

Depuis la racine du projet, avec Docker Desktop et son moteur Linux actifs :

```powershell
docker compose up --build -d frontend_demo
```

Ouvrir **http://localhost:8080**. Cette commande démarre seulement `frontend_demo`.
Python et Node.js ne sont pas nécessaires sur l'hôte. Ils sont utilisés pendant
la construction ; seule la sortie `frontend/dist/` est copiée dans Nginx.
L'interface lit des JSON locaux et ouvre les PDF/photos inclus dans le site.
Aucune requête n'est transmise à `/v1/` et aucune analyse n'est exécutée.

```powershell
$env:CLAIMS_REVIEW_PORT = '8081'
docker compose up --build -d frontend_demo
docker compose logs frontend_demo
docker compose stop frontend_demo
Remove-Item Env:CLAIMS_REVIEW_PORT -ErrorAction SilentlyContinue
```

Nginx s'exécute sans utilisateur root, avec un système de fichiers en lecture
seule et `/tmp` séparé. `/readyz` vérifie sa disponibilité sans autre service.
Les JSON et documents manquants produisent une réponse 404, sans renvoyer le
HTML de l'application à leur place.

## Actualiser les dossiers

Le jeu statique versionné se trouve dans `frontend/public/demo/`. Une collection
d'analyses synthétiques existante peut être exportée explicitement avec
l'environnement Python du projet :

```powershell
.\.venv\Scripts\python.exe scripts/export_static_demo.py --source-dir analysis_results --output-dir frontend/public/demo --replace
docker compose up --build -d frontend_demo
```

L'export préserve les archives originales et les résultats déjà calculés. Son
manifeste distingue les copies PDF remises en forme des documents analysés à
l'origine. Les pièces absentes restent absentes. Aucun benchmark ni modèle n'est
lancé. La reconstruction de l'image embarque les nouvelles copies statiques.

## API d'archives séparée

Le service optionnel `api` conserve son fonctionnement : consultation en lecture
seule d'une collection existante dans `analysis_results/`. Il n'est pas démarré
par `frontend_demo`. `CLAIMS_ANALYSIS_DIR` sélectionne son dossier monté ; cette
variable n'affecte pas la démo statique. Son port n'est pas publié par défaut.

Le profil `console` permet toujours de démarrer explicitement tous ses services
avec les bases :

```powershell
docker compose --profile console up --build -d
```

Cette commande requiert les archives pour l'API ; elle ne lance ni migration,
import ni analyse. Les services `postgres-test` et `qdrant-test` restent dans le
profil `test`. Le frontend connecté à une API se construit séparément avec
`npm --prefix frontend run build` ; les paramètres existants de développement
et de lancement Python restent disponibles.

La démo peut aussi être construite sans Docker et hébergée depuis le seul
dossier `frontend/dist/` ; voir le README principal pour les commandes et les
préfixes d'URL.
