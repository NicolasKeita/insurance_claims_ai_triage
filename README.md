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
