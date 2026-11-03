# Démo statique Docker

- Prérequis : Docker Desktop avec moteur Linux.
- Build : Next.js/Webpack et Panda CSS avec Node.js 24 dans Docker. Serveur : Nginx.
- Les quatre cas, PDF, photos et résultats sont embarqués depuis `frontend/public/demo/`.
- Aucun backend ni modèle requis. Python et Node inutiles sur l'hôte.
- L'export Next.js `frontend/out/` est servi à la racine ; les fichiers générés
  sous `/_next/static/` sont conservés avec leurs noms versionnés.

## Démarrer

Depuis la racine du projet :

```powershell
docker compose up --build -d frontend_demo
docker compose logs frontend_demo
docker compose stop frontend_demo
```

- Ouvrir [la démo](http://localhost:8080).
- Autre port : définir `$env:CLAIMS_REVIEW_PORT = '8081'` avant le démarrage.

## Actualiser les cas

Avec [l'environnement Python du projet](../../docs/INSTALLATION.md), exporter une collection synthétique existante :

```powershell
.\.venv\Scripts\python.exe scripts/export_static_demo.py --source-dir analysis_results --output-dir frontend/public/demo --replace
docker compose up --build -d frontend_demo
```

- Archives et résultats calculés préservés. Aucun benchmark lancé.

## API d'archives

```powershell
docker compose --profile console up --build -d
```

- Démarre aussi l'API, PostgreSQL et Qdrant. Archives existantes requises.
- `CLAIMS_ANALYSIS_DIR` sélectionne les archives montées en lecture seule ; défaut : `analysis_results/`.
- Le port de l'API n'est pas publié. La démo statique reste indépendante.
- [Installation et démarrage sans Docker](../../README.md).
