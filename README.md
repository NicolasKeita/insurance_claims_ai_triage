# Insurance claims assessment

AI-assisted claim review: document extraction, ML triage, anomaly detection,
computer vision, RAG and an advisory agent. Recommendations cite their evidence.
Decisions remain human. Review scores are not fraud probabilities.

![Claim Review console](docs/images/step33-guided-results-fr.jpg)

## Toolchain

- **Backend:** Python 3.12, FastAPI, Pydantic and LangGraph. One `.venv`.
- **Data:** PostgreSQL 18, SQLAlchemy, Alembic and Qdrant 1.19.
- **ML / documents:** scikit-learn, MLflow, PyTorch, Ultralytics, PyMuPDF and Tesseract.
- **LLM / retrieval:** Ollama and Sentence Transformers.
- **Frontend:** Next.js 16, React 19, TypeScript 6 and Panda CSS. Node.js 22.13+ and npm.
- **Build / checks:** setuptools, pytest, nk-python-style, Vitest, ESLint and Prettier.
- **Version:** `pyproject.toml` is the application version source.

## Setup

Run commands from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm --prefix frontend ci
```

For vision, CUDA/CPU wheels, optional extras or other platforms, see
[installation](docs/INSTALLATION.md).

## Static demo

Four prepared cases with PDFs, photos and saved results. FR/EN and light/dark themes.

```powershell
npm --prefix frontend run build:demo
npm --prefix frontend run preview:demo
```

- Open [localhost:4173](http://localhost:4173).
- Serve `frontend/out/` over HTTP. The built site needs no backend, Python or models.
- Development: `npm --prefix frontend run dev:demo`.
- Next.js uses Webpack for development and builds; Panda generates the styles.
- Subdirectory hosting: build with `CLAIMS_REVIEW_BASE_PATH=/insurance-demo`,
  then publish the contents of `frontend/out/` at `/insurance-demo/`:

```powershell
$env:CLAIMS_REVIEW_BASE_PATH = '/insurance-demo'
npm --prefix frontend run build:demo
Remove-Item Env:CLAIMS_REVIEW_BASE_PATH
```

- The public demo reads synthetic cases and saved results. Visiting it does not
  run inference or accept new claim documents.

With Docker Desktop, build and serve the same demo without host Python or Node:

```powershell
docker compose up --build -d frontend_demo
```

Open [localhost:8080](http://localhost:8080). See [Docker commands](deploy/docker/README.md).

## Local snapshot demo

Prepare document-only snapshots, then start FastAPI and Next.js together:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_demo.py
.\.venv\Scripts\python.exe scripts/start_review_console.py
```

- Open [127.0.0.1:5173](http://127.0.0.1:5173). Ctrl+C stops both processes.
- No GPU, database or Ollama required. Unavailable components stay explicit.
- Preparation preserves existing output. Use `--output-dir <path>` for another set.
- Read existing snapshots or archives with `start_review_console.py --demo-dir <path>`.
- The console reads saved results. Opening it does not run an assessment.

## Database-backed API

Use a terminal without `CLAIMS_REVIEW_DEMO` or `CLAIMS_REVIEW_DEMO_DIR` set:

```powershell
docker compose up -d --wait postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:CLAIMS_REVIEW_ASSET_ROOT = (Join-Path $PWD 'data')
.\.venv\Scripts\python.exe -m alembic upgrade head
# Initial setup only: import the reference claim.
.\.venv\Scripts\python.exe scripts/import_reference_claim.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- API reference: [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- In a second terminal: `npm --prefix frontend run dev`.
- Open the console at [127.0.0.1:5173](http://127.0.0.1:5173), then enter `CLAIM-2026-00001`.
- Inference needs configured local models and services. See [evaluation setup](docs/END_TO_END_EVALUATION.md).
- PostgreSQL/Qdrant do not start FastAPI. Keep the API terminal open.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -m unit --ignore=tests/e2e
.\.venv\Scripts\nk-python.exe check src tests
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run format:check
npm --prefix frontend run build
```

- End-to-end tests require an explicit request. See [E2E commands](tests/e2e/README.md).
- Integration tests need their services and local models. Pytest starts or downloads nothing.
- Start test databases: `docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test`.
- Run integrations: `.\.venv\Scripts\python.exe -m pytest -m integration --ignore=tests/e2e`.
- RAG uses `data/knowledge` and the cached `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model.
- To populate that cache explicitly:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

## Guides

- [Installation and environments](docs/INSTALLATION.md)
- [Source and test layout](docs/DIRECTORY_LAYOUT.md)
- [Local models, results and generated files](docs/ARTIFACTS.md)
- [Saved analysis exports](analysis_results/README.md)
- [End-to-end evaluation and reports](docs/END_TO_END_EVALUATION.md)
- [Business scenarios](tests/e2e/SCENARIOS.md)
- [Damage model improvement](docs/DAMAGE_MODEL_IMPROVEMENT.md)
