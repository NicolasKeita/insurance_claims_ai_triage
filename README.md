# Insurance claims assessment

This project extracts claim documents, compares the available evidence, builds structured `TriageFeatures`, and serves three separate assessments: operational triage, statistical anomaly detection, and deterministic investigation signals.

## Meaning of the outputs

- **Inconsistency** = disagreement between available evidence, such as two documents reporting different incident dates.
- **Anomaly** = statistical atypicality relative to a reference population. The listed unusual features are descriptive observations, not exact model attributions.
- **Investigation signal** = an explicit reason for additional human review, backed by structured evidence.
- **Fraud determination** = outside the scope of this component.

`review_score` is a deterministic heuristic review-prioritization score defined by a versioned policy. It is not a probability of fraud.

The investigation engine in `src/investigation/` translates a `ClaimConsistencyReport`, an `AnomalyAssessment`, and document completeness fields in `TriageFeatures` into ordered signals. The engine does not consume the triage workflow prediction. Neither an anomaly nor a review recommendation changes the operational triage output.

## Investigation policy

`investigation_policy_v1` is preserved in `src/investigation/policy.py` as `INVESTIGATION_POLICY_V1`. The default is now `INVESTIGATION_POLICY_V2` (`investigation_policy_v2`), which keeps all original weights and thresholds and adds the historical rules described below. Its weights are transparent demonstration choices, not insurance industry standards.

| Signal code | Weight |
| --- | ---: |
| `DOCUMENT_FIELD_MISMATCH` | 10 per mismatched field |
| `ADDITIONAL_QUOTE_DAMAGE` | 18 |
| `UNMAPPED_QUOTE_ITEM` | 8 |
| `ANOMALOUS_PATTERN` | 18 |
| `EXTREME_ANOMALY_PERCENTILE` | 20 |
| `MISSING_DOCUMENTS` | 12 |

The score is the sum of signal weights, capped at 100. Scores below 20 are `NORMAL`; 20–49 are `ELEVATED`; 50–100 are `HIGH`. Human review is recommended at 20 or above. The extreme anomaly percentile signal starts at `0.99`. The completeness signal appears when the ratio is below `0.80` and unavailable entries are observed. An unavailable dossier entry is not asserted to be legally required.

## API

`POST /v1/triage/predict` and `POST /v1/anomaly/assess` continue to accept a `TriageFeatures` JSON object.

`POST /v1/investigation/assess` accepts structured JSON with `features`, `consistency`, and `anomaly` objects. `POST /v1/claims/assess` accepts `features` and `consistency`; it computes triage and anomaly using the loaded candidate models, then passes the resulting anomaly assessment into the investigation engine. Its response has separate `triage`, `anomaly`, and `investigation` objects. The detailed consistency report is required so the engine can retain which checks disagreed and which quote items were involved. Feature counts must agree with that report.

`GET /health` identifies the loaded triage and anomaly models and reports `investigation_policy_version`. The policy is code, not a third MLflow model.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_investigation_engine.py
```

The demonstration reads the real PDFs in `data/CLAIM-2026-00001` and supplies a clearly labelled illustrative anomaly assessment so it runs without MLflow, Ollama, Tesseract, or an external service. To use the registered candidate anomaly model when MLflow tracking is available, set `MLFLOW_TRACKING_URI` and run the script with `--registered-anomaly`.

### PostgreSQL

The `persistence` package uses SQLAlchemy 2, psycopg 3 and Alembic. PostgreSQL
stores customers, policies, structured claims, declared damage, document/image
metadata and immutable-by-convention assessment snapshots. PDF/JPEG bytes,
`claim.json`, extraction assets and MLflow artifacts remain on the filesystem.
Filenames are relative to a claim's existing dossier, not global paths; loading
a persisted `Claim` does not create or locate a `ClaimCase` directory.

PowerShell, from the repository root (Docker Desktop must be running):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
# Optional: copy and customize the development-only Compose settings.
Copy-Item .env.example .env
docker compose up -d --wait postgres
$env:DATABASE_URL = "postgresql+psycopg://claims:claims@localhost:5433/insurance_claims"
python -m alembic upgrade head
python .\scripts\import_reference_claim.py
python .\scripts\check_claim_history.py
python -m pytest
```

The host port defaults to **5433** to coexist with the installed native
PostgreSQL on 5432. Set `POSTGRES_PORT` in `.env` and adjust the URLs to change
it. Compose reads `.env`; Python and Alembic read process environment variables
only. Change the development credentials before using another environment;
do not commit `.env`. The password is never embedded in Python or `alembic.ini`.
Changing Compose credentials does not reset an already initialized volume.

The reference import reports one customer, one policy, four document entries
(three available and one missing), two images and two declared damage types.
Duplicate business `claim_id` values raise `ClaimAlreadyExistsError`; the
script prints a concise message and exits with status 1. Existing records are
never overwritten. Customers/policies are reused through unique business IDs,
including concurrent inserts; a different product for an existing policy is
rejected with `PolicyConflictError`. No exclusive policy/customer relationship
is inferred from the current data.

`ClaimRepository` returns Pydantic `Claim` objects and supports existence,
business-ID lookup and lists by customer or policy. UUIDs stay in the storage
layer. `AssessmentRepository` provides `add_triage`, `add_anomaly`,
`add_investigation`, `get_latest_*` and `list_*_history`. Each add inserts a
new UUID row; there is no repository update/delete API for assessments.
For anomaly assessments, `latest` means maximum `(created_at, insertion_order)`;
the database-generated identity counter reliably breaks timestamp ties by
allocation order, not transaction commit order. Migration `0002` backfills
legacy rows using their original `(created_at, UUID)` order: the actual insertion
chronology of old timestamp ties cannot be recovered. Triage and investigation
retain their existing `(created_at, UUID)` stable ordering. JSONB
preserves probabilities, anomaly `statistical_signals` (as `unusual_features`)
and the original investigation signals/evidence and policy version.
Triage probabilities are checked on writing and reading.

`PersistenceService.import_claim(ClaimCase | Claim)` opens one transaction for
the entire import. `save_assessments(claim_id, triage=..., anomaly=...,
investigation=...)` accepts existing predictor/engine results and atomically
appends whichever results are supplied. Each call creates and closes its own
session. Repositories flush but never commit or roll back the caller's unit
of work. Failures roll back all writes. Existing API contracts stay unchanged;
the persistence service can be called internally without starting FastAPI.

To store an existing combined `/v1/claims/assess` response saved in a JSON
file, run the following twice to demonstrate append-only histories:

```powershell
python .\scripts\check_claim_history.py --assessment-json .\assessment.json
python .\scripts\check_claim_history.py --assessment-json .\assessment.json
```

Amounts use `Decimal` in `RepairEstimate` and PostgreSQL `NUMERIC` without
forced rounding; the filesystem loader parses JSON decimals directly.
Pydantic JSON serialization of that amount now produces a decimal string.
Model feature values remain floats. Dates use `DATE`, timestamps use UTC-aware
`TIMESTAMP WITH TIME ZONE`, and enums use VARCHAR with CHECK constraints.
Created/updated timestamps are supplied centrally by the application; updates
made outside SQLAlchemy must set `updated_at` themselves.

Indexes support customer/date, policy, incident date, collision type and each
assessment's latest ordering (claim/date/counter for anomaly, claim/date/UUID
otherwise). Business IDs and claim/damage pairs
are unique. Foreign keys and CHECK constraints enforce available-document
filenames, enum values, nonnegative amounts, currency format and score bounds.

PostgreSQL integration tests use a separate database, require a name ending
in `_test`, reject the configured development database and create/drop only
their own random schema. They verify an empty-schema migration cycle
`upgrade head -> downgrade base -> upgrade head`, ORM/schema agreement,
round trips, reuse, duplicate rejection, append-only history, latest and
rollback. With no `TEST_DATABASE_URL`, they skip and normal pytest runs
without PostgreSQL. Do not point the test URL at an existing business database.

```powershell
docker compose exec -T postgres createdb -U claims insurance_claims_test
$env:TEST_DATABASE_URL = "postgresql+psycopg://claims:claims@localhost:5433/insurance_claims_test"
python -m pytest -m postgres
Remove-Item Env:TEST_DATABASE_URL
docker compose down
```

If the test database already exists, skip `createdb`. Adapt `-U` if you changed
the Compose user. Normal shutdown keeps the named volume; do not add `-v`.
Append-only is an application convention, not a database permission/trigger
restriction against administrative SQL. No semantic search, embeddings, RAG,
similar-claim lookup or additional model is introduced.

## Historical investigation context (step 25)

Historical context only uses claims of the same customer with an incident date
**strictly before** the current claim's incident date, to **prevent temporal
leakage**. The current claim, later incidents and same-day incidents are excluded;
there is no reliable incident time to order same-day claims. `created_at` is
insertion time and never defines incident chronology. Windows are inclusive at
`D - 30/90/365 days` and exclusive at `D`.

`HistoricalClaimProfile` is a pure Pydantic object. It contains previous claim
count, 30/90/365-day counts, days since the most recent previous incident,
total/average/maximum previous repair estimate amounts, and previous anomalous
claim count. Counts include every currency; monetary aggregates include only
the current claim's currency, identified by `repair_amount_currency` and
`previous_claims_in_amount_currency`. There is no currency conversion. Amounts
are `Decimal` and serialize as JSON strings. With no compatible amounts, total
is zero and average/maximum are null; with no history, counts are zero and days
since the previous claim are null.

`previous_anomalous_claim_count` refers to previous claims whose **latest stored
anomaly assessment is currently flagged anomalous**. It is not a fraud count.
Anomaly history is a model output, not truth. Assessments may have been stored
after the current incident; this is current observed context, not a historical
as-of reconstruction. PostgreSQL uses a window function and a one-row-per-claim
join, so old append-only alerts are never counted separately. Building a profile
takes two SQL queries regardless of history length. Existing investigation
scores, priorities and recommendations are intentionally excluded from inputs
to avoid a policy feedback loop.

Policy v2 adds `HISTORICAL_CONTEXT` signals:

| Code | Demonstration trigger | Weight |
| --- | --- | ---: |
| `RECENT_CLAIM_FREQUENCY` | At least 3 previous incidents in 30 days **or** 5 in 365 days | 15 once |
| `PREVIOUS_ANOMALOUS_CLAIMS` | At least 2 previous claims with latest anomaly flag true | 18 |

Both window triggers share one frequency signal/weight. No short-interval
signal is added to avoid redundant scoring; the interval remains observable.
No cumulative-amount signal is generated. These thresholds demonstrate the
mechanism and do not represent a real insurer's rules. V1 emits no historical
signals even if a profile is supplied; existing saved v1 assessments remain
unchanged. ML features, datasets and training pipelines remain unchanged.

`GET /v1/claims/{claim_id}/history` loads facts from PostgreSQL without inference
or writes. Unknown claims return 404; an unconfigured/unavailable DB returns
503, never an empty profile. Models load lazily, so this route needs no MLflow
server. A session/transaction is created and closed per request; the engine is
disposed when the app lifespan ends.

`POST /v1/claims/{claim_id}/investigation/assess` accepts only:

```json
{"consistency": {"claim_id_match": true, "incident_date_match": true, "location_match": true, "vehicle_match": true, "collision_type_match": true, "injuries_match": true, "declared_damage_match": true, "additional_quote_damage": [], "unmapped_quote_items": []}}
```

Consistency reports are not yet stored in the schema, so the caller supplies
the explicit document comparisons. The service loads the claim, builds its
history, derives the existing features from persisted incident/vehicle/damage/
document metadata and repair estimate plus that comparison, runs the loaded
candidate anomaly model, and appends anomaly and investigation assessments
atomically. Historical counts and feature values cannot be supplied by the
HTTP client. The transaction commits before the response is sent. Signals and
their exact evidence are JSONB snapshots under `investigation_policy_v2`;
later anomaly/history changes do not rewrite earlier explanations. Stateless
routes remain available, using an empty history with the default v2 policy.

After the reference import and migrations, run:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims?connect_timeout=5"
python -m alembic upgrade head
python .\scripts\seed_historical_claims.py
python .\scripts\seed_historical_claims.py  # verifies identical existing fixtures
python .\scripts\check_historical_profile.py
python .\scripts\check_historical_api.py
```

The last script uses FastAPI's HTTP TestClient against the real database,
compares the real reference PDFs, supplies a clearly marked illustrative
current anomaly input, and verifies the saved evidence snapshot. It appends
one current anomaly and investigation per run. Use `--anomaly-model
artifacts/anomaly_v1/anomaly_detector.joblib` to reuse the existing local trained
detector, or `--registered-anomaly` with `MLFLOW_TRACKING_URI` to use the
existing registered candidate. Neither option trains a model. Response JSON
files are written under ignored `artifacts/step25/`.

Fixtures have reserved `DEMO-STEP25-HISTORY-*` IDs, synthetic location/product
labels, six previous incidents at D-2/-15/-30/-90/-365/-400, plus same-day,
later-incident and other-customer controls. The seed is atomic and idempotent;
unexpected data at a reserved claim ID is rejected and preserved. No PDFs or
images are created. For the supplied reference, the profile is 6 previous
claims, windows 3/4/5, interval 2 days, total/average/max EUR 2100/350/600, and
2 latest anomaly flags. Real additional history can change these counts.

Run standard tests without setting `TEST_DATABASE_URL`; run `pytest -m postgres`
with the dedicated test database URL from the PostgreSQL section, replacing
`localhost` with `127.0.0.1` if IPv6 loopback stalls. Integration tests cover
temporal/customer isolation, exact boundaries, currencies, latest-state ties,
two-query behavior, fixture conflicts/idempotence, API atomic rollback and
evidence preservation after history changes. `CODEX_REPORT_STEP_25.md` records
the actual validation outputs and reproduction commands.

## Similar Claims (STEP26)

PostgreSQL is the **source of truth**. Qdrant is a rebuildable **derived vector
index**: it supplies technical UUIDs, ranking and `similarity_score`; displayed
business facts are bulk-reloaded from PostgreSQL. Retrieval never writes SQL
data, creates an index during HTTP requests, or reads previous AI assessments.

The configurable Sentence Transformer defaults to
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (French/English),
with symmetric L2-normalized embeddings and COSINE distance. Model construction
and downloading happen only on explicit indexing or a first valid search;
API import, lifespan startup and standard tests do not load it. One provider
and service are cached per API lifespan, with injectable test services.

`claim_retrieval_v1` uses stable labels for type, collision, location, vehicle,
sorted damages, injuries, repair amount and currency. IDs, incident date,
customer/policy, status, ML outputs and model/policy versions are excluded from
embedding text. Dates are structured filters. Qdrant UUID = PostgreSQL claim
UUID; repeat indexing upserts the same points.

The default collection is `insurance_claims_similar_v1` (override with
`CLAIMS_SIMILAR_COLLECTION`). Collection metadata stores the representation,
model name, dimension, normalization and distance contract. Incompatible or
unmanaged collections raise an explicit error without deletion, including when
two models share a dimension. Prefer a new collection such as
`insurance_claims_similar_v2` when the representation/model changes, reindex it,
then switch the API configuration and restart. The local reproducibility
manifest is `artifacts/similar_claims/index_manifest.json`; it is not authoritative.

Eligibility is strictly `candidate.incident_date < current.incident_date`, and
by default `candidate.claim_type == current.claim_type`. Both filters run in
Qdrant and again after SQL reload. Current, same-day and later incidents are
excluded. Deleted/missing rows and stale dates/types are skipped. Candidate
pages refill the requested top K, with a 1000-candidate safety bound; a very
stale index may return fewer results. Default limit is 5, maximum 20.

PowerShell workflow (from repository root):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
docker compose up -d postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:CLAIMS_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
$env:CLAIMS_SIMILAR_COLLECTION = 'insurance_claims_similar_v1'
$env:CLAIMS_INDEX_BATCH_SIZE = '32'
python -m alembic upgrade head
python .\scripts\import_reference_claim.py # only if not already imported; duplicates are preserved/rejected
python .\scripts\seed_historical_claims.py
python .\scripts\seed_similar_claims.py
python .\scripts\index_similar_claims.py
python .\scripts\index_similar_claims.py # same UUIDs, same point count
python .\scripts\check_similar_claims.py --output .\artifacts\similar_claims\demo_response.json
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# In a second terminal:
Invoke-RestMethod 'http://127.0.0.1:8000/v1/claims/CLAIM-2026-00001/similar?limit=5'
```

`GET /v1/claims/{claim_id}/similar?limit=5` returns `claim_id`,
`representation_version` and `results` with claim ID, incident date, type,
collision, authoritative repair amount/currency and `similarity_score`.
Unknown claim: 404; invalid limit: 422; unavailable/missing/incompatible index
or database: 503. Amounts serialize as decimal strings, matching the existing
domain convention. Qdrant is pinned to `qdrant/qdrant:v1.19.0`, exposes local
REST 6333/gRPC 6334, and checks `/readyz` using the image's built-in bash.
Named volumes survive ordinary `docker compose down`; `down -v` deletes them.

Explicit rebuild of **only** the configured Qdrant index:

```powershell
python .\scripts\index_similar_claims.py --recreate
```

Batch SQL reads use keyset pagination and eager damage loading (two queries
per nonempty batch), with no open SQL transaction during embedding/upsert.
`SimilarClaimsIndexer.upsert_claim(claim_id)` supports later synchronization
after a committed SQL change. This V1 uses manual batch synchronization;
Qdrant is eventually consistent with PostgreSQL. Ordinary upserts do not purge
deleted SQL rows; search ignores them, and explicit rebuild removes them.
Concurrent SQL edits during indexing can require another pass.

Six additional reserved `DEMO-STEP26-SIMILAR-*` fixtures demonstrate frontal,
parking, side and rear collisions, varied vehicles and repair amounts. They
use a separate demo customer/policy, preserving STEP25 historical counts.
The current damage enum only describes frontal components; side/rear/parking
fixtures leave damages empty rather than inventing new enum values.

Tests without infrastructure or model downloads:

```powershell
Remove-Item Env:TEST_DATABASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:RUN_QDRANT_TESTS -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest
# Remove offline flags before a first real model download.
Remove-Item Env:HF_HUB_OFFLINE,Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
# Opt-in real Qdrant; unique test collections are cleaned up.
$env:RUN_QDRANT_TESTS = '1'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
python -m pytest -m qdrant
# For combined PostgreSQL/Qdrant tests, also configure the dedicated *_test DB:
$env:TEST_DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims_test'
python -m pytest -m 'postgres or qdrant'
```

No human similarity ground truth exists: demo sanity checks cannot establish
accuracy, precision@5 or recall@5. A similarity score is not a probability and
is never displayed as a percentage. Dense embeddings are imperfect for precise
numeric similarity (amount, year, counts). A later reranker could combine
semantic, numeric and categorical comparisons; this V1 implements none.

## Policy and procedure RAG (STEP27)

The demo policy and procedure documents are synthetic and are not real insurance terms or legal guidance.
Markdown sources and their strict JSON metadata sidecars in `data/knowledge/`
are authoritative. The two policy products (`AUTO_PREMIUM`, `AUTO_BASIC`) and
the global auto handling procedure exist only to demonstrate the architecture.
Qdrant stores derived text copies and provenance for inspection; search reloads
chunks from local sources before displaying text or resolving citations.

`insurance_policy_chunks_v1` and `insurance_procedure_chunks_v1` are separate
COSINE collections. `knowledge_chunking_v1` follows heading paths and paragraph
boundaries, prefers sentence endings for long paragraphs, and uses a configurable
600-character maximum with no overlap. Oversized single tokens are split as a
last resort. UUID5 chunk IDs depend on source metadata/version, section, index,
text, and chunking configuration; identical input produces identical IDs.

The existing embedding abstraction is reused. `RAG_EMBEDDING_MODEL` is independent
of `CLAIMS_EMBEDDING_MODEL`; both initially default to the cached multilingual
MiniLM model. RAG uses normalized `encode_document()` / `encode_query()` when
available, with a fallback to `encode()`. A model without specialized prompts
may yield equivalent query/document encoding. Similar Claims keeps symmetric
`encode()` and its existing collection contract. Dimensions are discovered.
Changing model, dimension, chunking version or size requires a new collection
version or an explicit rebuild; incompatible/unmanaged indexes are rejected.
Model loading stays lazy; standard tests use fake providers and in-memory Qdrant.

Search applies source type, optional product/language/source ID, and optional `as_of_date`.
Validity uses inclusive start and exclusive end: `effective_from <= date <
effective_to`; null bounds are open. Supplying a product means exact filtering,
with no fallback to another product. A general search without `as_of_date` does
not select a current version automatically. A claim-aware POLICY search reads
the claim from PostgreSQL, forces `claim.policy.product`, and uses its incident
date. The claim request has no client-supplied product/date. PROCEDURE search is
global and has no automatic date filter: no reliable handling date exists yet.
Call the general endpoint with explicit filters when a handling date is known.

`KnowledgeRagService` retrieves first and passes only that context to the existing
structured Ollama provider. The prompt forbids outside knowledge and invented
terms, and requests `insufficient_evidence=true` for unsupported questions.
Unknown cited chunk IDs are rejected (HTTP 502), even in an insufficient answer;
an answer claiming sufficient evidence must cite a chunk. Citation metadata is
resolved from structured citations. If the LLM also repeats references in prose,
those IDs are validated, promoted into the citation list, and their markers are
removed. Unknown inline IDs are rejected too. Metadata is
resolved by the application, never supplied by the LLM. Empty retrieval returns
an insufficient answer without calling the LLM. Citation membership checks do
not prove that every statement is entailed: generation can still make errors.
The RAG component does not change InvestigationPolicy/scoring; the advisory
agent introduced in STEP28 reuses its retriever.

HTTP endpoints (limits: 1–20; invalid payload: 422; unknown claim: 404;
unavailable index/DB/LLM: 503):

- `POST /v1/knowledge/search`: `query`, `source_type`, optional `product`,
  `language` (default `en`), `source_id`, `as_of_date`, `limit` (default 5).
- `POST /v1/knowledge/answer`: `question`, `source_type`, optional `product`,
  `language`, `source_id`, `as_of_date`, `retrieval_limit` (default 5).
- `POST /v1/claims/{claim_id}/knowledge/search`: `query`, `source_type`,
  `language`, `limit`; product/date come from the persisted claim for POLICY.
- `POST /v1/claims/{claim_id}/knowledge/answer`: `question`, `source_type`,
  `language`, `retrieval_limit`; same persisted-claim convention.

Search returns `results` with chunk text, provenance and `similarity_score`.
Answer returns `answer`, `citations` (chunk ID, source ID/version, title, section),
and `insufficient_evidence`. A similarity score is not a probability or percentage.
INFO logs record filters, retrieved IDs/scores and cited IDs. Queries are DEBUG
only; the explicit local smoke script prints demo excerpts. Do not enable this
demo output blindly for sensitive documents.

PowerShell from the repository root (choose an already installed Ollama model):

```powershell
.\.venv\Scripts\Activate.ps1
docker compose up -d postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:RAG_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
$env:RAG_POLICY_COLLECTION = 'insurance_policy_chunks_v1'
$env:RAG_PROCEDURE_COLLECTION = 'insurance_procedure_chunks_v1'
$env:RAG_MAX_CHUNK_CHARS = '600'
$env:RAG_INDEX_BATCH_SIZE = '32'
python -m alembic upgrade head
# Import reference only if absent; the import preserves/rejects duplicates.
python .\scripts\import_reference_claim.py
python .\scripts\index_knowledge_base.py
python .\scripts\index_knowledge_base.py # stable IDs, unchanged point counts
python .\scripts\evaluate_knowledge_retrieval.py # no Ollama required
ollama list
# Set to the installed model you want to use. Start Ollama if it is not running.
$env:CLAIMS_LLM_MODEL = '<installed-model-name>'
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
python .\scripts\check_knowledge_rag.py
python .\scripts\check_knowledge_api.py # real services via HTTP TestClient
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second configured terminal:

```powershell
$search = @{query='collision deductible'; source_type='POLICY'; product='AUTO_PREMIUM'; language='en'; limit=5} | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8000/v1/knowledge/search' -Method Post -ContentType 'application/json' -Body $search
$answer = @{question='What is the collision deductible?'; source_type='POLICY'; language='en'; retrieval_limit=5} | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8000/v1/claims/CLAIM-2026-00001/knowledge/answer' -Method Post -ContentType 'application/json' -Body $answer
```

The index manifest is `artifacts/rag/knowledge_index_manifest.json`. Repeat
upserts do not delete obsolete chunks; search ignores IDs absent from current
sources. A manifest records this run's sources/chunks plus total stored point
counts. Remove old versions/obsolete points with an explicitly controlled rebuild
of only the two owned RAG collections (PostgreSQL/Similar Claims are untouched):

```powershell
python .\scripts\index_knowledge_base.py --recreate
```

The versioned 13-case retrieval benchmark is
`data/evaluation/rag_retrieval_v1.json`: 12 answerable queries, one unsupported
volcanic-ash question. Ground truth is exact source ID + section. Macro Recall@1,
@3, @5 counts distinct relevant sections; MRR uses the first relevant rank within
five results. Unsupported queries are excluded from recall/MRR and separately
report returned count, top score and empty-result rate. With no calibrated
threshold, those are diagnostics, not unsupported-answer detection accuracy.
Generation abstention is tested separately. Results are saved to
`artifacts/rag/retrieval_metrics.json` and `retrieval_results.csv`. This small
synthetic benchmark compares configurations and cannot establish production
quality; pytest imposes no semantic quality threshold.

```powershell
Remove-Item Env:TEST_DATABASE_URL,Env:RUN_QDRANT_TESTS,Env:RUN_RAG_TESTS -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest
# Real retrieval integration (cached model, no Ollama); unique test collections:
$env:RUN_RAG_TESTS = '1'
python -m pytest -m rag
Remove-Item Env:RUN_RAG_TESTS -ErrorAction SilentlyContinue
# Remove offline flags if a chosen model has not yet been downloaded.
Remove-Item Env:HF_HUB_OFFLINE,Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
```

## Advisory claims agent (STEP28)

The versioned `claims_agent_v1` uses an explicit LangGraph loop:
`PLAN -> TOOL -> PLAN`, or `PLAN -> FINALIZE -> VALIDATE -> END`.
The existing `StructuredLlm` abstraction produces a validated Pydantic plan
containing exactly one action. Native model tool calling is not required.
After at most **8 planner/tool iterations**, the graph finalizes with the
available evidence and records the limit as an uncertainty. Exact duplicate
tool invocations are blocked. HTTP callers cannot change the iteration limit,
tool set, or system prompt.

All eight tools read through existing repositories/services:

| Tool | Read source |
| --- | --- |
| `GET_CLAIM` | Persisted claim |
| `GET_HISTORY` | HistoricalClaimProfile |
| `GET_TRIAGE` | Latest stored triage assessment |
| `GET_ANOMALY` | Latest stored anomaly assessment |
| `GET_INVESTIGATION` | Latest stored investigation assessment |
| `FIND_SIMILAR_CLAIMS` | SimilarClaimsService; up to 5 earlier comparable claims |
| `SEARCH_POLICY` | KnowledgeRetriever; current product and incident date, up to 5 chunks |
| `SEARCH_PROCEDURE` | Global procedure KnowledgeRetriever; up to 5 chunks |

The URL claim ID defines the scope; planner arguments cannot supply another
claim ID. Tools perform no business writes, inference, reindexing, SQL generation,
approvals, denials, payment, messaging or fraud determination. Missing stored
assessments return `NOT_AVAILABLE` rather than computing or persisting new ones.
Anomaly means statistical atypicality; similar claims are comparative facts,
not evidence that a previous handling decision applies to this claim.

Collected evidence has stable references and bounded summaries. Recommendation
and rationale evidence IDs must belong to the evidence collected during that run;
invented citations fail validation. Policy/procedure citation metadata is resolved
by application code. Tool output, persisted free text, and retrieved documents
are untrusted evidence and cannot add instructions or capabilities. Citation
membership validates provenance, not full semantic entailment.
The finalizer requires explicit rationale, uncertainties and evidence IDs. Its
aggregate evidence context is capped at 18,000 characters, retaining every ID;
shortened details become an uncertainty. The agent requests an 8192-token Ollama
context through the existing structured LLM adapter to avoid the default local
4096-token context truncating structured responses.

The recommendation always has `human_review_required=true`. Its allowed next
actions are `CONTINUE_STANDARD_REVIEW`, `REQUEST_ADDITIONAL_INFORMATION`,
`REFER_TO_EXPERT`, `REFER_FOR_INVESTIGATION_REVIEW`, and `NO_RECOMMENDATION`.
A human retains the final claim decision. The demo policies and procedures
remain **synthetic** and do not establish real coverage or handling requirements.

Completed and failed runs are appended to PostgreSQL `agent_runs`, with versions,
model identifier, counters, elapsed time, evidence, recommendation, and a bounded
structured action trace. Runs coexist when a claim is reviewed again. No private
model chain of thought is requested or stored. This audit persistence is the
agent's only write; business claim/assessment data stays unchanged.

Endpoints:

- `POST /v1/claims/{claim_id}/agent/review`: optional JSON body with only an
  optional `objective` (1–2000 characters); omitted body uses the default objective.
- `GET /v1/claims/{claim_id}/agent/runs?limit=20`: newest runs first, limit 1–100.
- `GET /v1/agent/runs/{run_id}`: retrieve a completed or failed audit run.

Unknown claims/runs return 404; invalid input returns 422; unavailable services
return sanitized 503 errors; invalid evidence grounding returns 502. Failed review
responses include the persisted run ID when a run was created. Agent services,
embeddings, and retrievers are loaded lazily once per API lifespan, sharing the
existing Similar Claims and knowledge caches. Read-only audit endpoints load no LLM.

PowerShell from the repository root (reuse existing imported/indexed data):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
docker compose up -d postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:CLAIMS_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
$env:CLAIMS_SIMILAR_COLLECTION = 'insurance_claims_similar_v1'
$env:RAG_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
$env:RAG_POLICY_COLLECTION = 'insurance_policy_chunks_v1'
$env:RAG_PROCEDURE_COLLECTION = 'insurance_procedure_chunks_v1'
$env:RAG_MAX_CHUNK_CHARS = '600'
python -m alembic upgrade head
# Only on initial setup: import_reference_claim.py rejects existing duplicates.
# python .\scripts\import_reference_claim.py
python .\scripts\seed_historical_claims.py
python .\scripts\seed_similar_claims.py
python .\scripts\index_similar_claims.py
python .\scripts\index_knowledge_base.py
ollama list
# Choose an installed model; start Ollama if it is not running.
$env:CLAIMS_LLM_MODEL = '<installed-model-name>'
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
python .\scripts\check_claims_agent.py --debug
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

No MLflow server is required for agent reviews: assessments are read from existing
snapshots. An absent assessment stays unavailable. In a second terminal:

```powershell
$body = @{objective='Review this claim and recommend the next human review step.'} | ConvertTo-Json
$run = Invoke-RestMethod 'http://127.0.0.1:8000/v1/claims/CLAIM-2026-00001/agent/review' -Method Post -ContentType 'application/json' -Body $body
$run.recommendation
Invoke-RestMethod 'http://127.0.0.1:8000/v1/claims/CLAIM-2026-00001/agent/runs?limit=2'
Invoke-RestMethod "http://127.0.0.1:8000/v1/agent/runs/$($run.run_id)"
```

The smoke script defaults to `CLAIM-2026-00001`, prints tools, iterations, evidence
IDs, recommendation and uncertainties, and saves the terminal result under
`artifacts/agent/claims_agent_smoke.json`. `--debug` prints the structured action
trace. Standard pytest uses fake planners/finalizers/tools and needs no external
services. PostgreSQL tests remain opt-in through `TEST_DATABASE_URL`.
Small local models may plan imperfectly, and recommendations depend on evidence
quality/availability. This V1 has no long-term conversational memory or
quantitative agent benchmark. `CODEX_REPORT_STEP_28.md` records actual validation.
