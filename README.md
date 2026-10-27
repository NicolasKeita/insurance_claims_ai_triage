# Insurance claims assessment

This project extracts claim documents, compares the available evidence, builds structured `TriageFeatures`, and serves three separate assessments: operational triage, statistical anomaly detection, and deterministic investigation signals.

## Portfolio Demo

**AI-assisted insurance claim review platform combining Document AI, ML triage,
anomaly detection, computer vision, RAG and evidence-grounded agent orchestration.**
Decisions remain human. Every recommendation retains its provenance, with
clickable references to the documents, observations and knowledge excerpts used.
The **Claim Review** console separates workflow triage, statistical anomaly and
investigation priority. It has no approval, denial, payment or misconduct action.

![Claim Review console with recorded coherent claim](docs/images/step33-coherent-claim.jpg)

Additional captures: [investigation signals](docs/images/step33-investigation-signals.jpg),
[visual evidence](docs/images/step33-visual-evidence.jpg),
[agent recommendation](docs/images/step33-agent-recommendation.jpg).

### Minimal demo

Prerequisites: the project's Python 3.12 `.venv` with application dependencies
installed (see [installation](docs/INSTALLATION.md)), and Node.js 22.12+ with npm.
No GPU, PostgreSQL, Ollama, Qdrant or downloaded model is needed for this mode.
Run from the repository root in PowerShell:

```powershell
# Explicit preparation: canonical controlled fixtures, real document parsers,
# consistency comparison and deterministic document/completeness signals only.
.\.venv\Scripts\python.exe scripts/prepare_review_demo.py
$env:CLAIMS_REVIEW_DEMO = '1'
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, from the repository root:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open [Claim Review](http://127.0.0.1:5173). Choose a controlled claim in the
selector: coherent documents, additional quote damage, or missing documents.
The source is `data/evaluation/e2e_v2/scenarios.json`, transformed centrally by
the existing fixture builder; the console maintains no duplicate claim JSON.
Preparation writes snapshots under ignored `artifacts/review_demo/`. Refreshing
the page only reads snapshots. Preparation refuses to overwrite an existing
demo directory; use `--output-dir` and set `CLAIMS_REVIEW_DEMO_DIR` to that path
to prepare a separate demonstration.

Absent ML, vision, history, RAG and agent results show explicit states and
reasons. A partial document assessment is labelled with its limited scope;
its absence of other signals does not imply those components were evaluated.

### Previously generated results

To show richer evidence without running inference, import an **existing real**
benchmark run into a separate demo directory. For example, if this local run
already exists:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_demo.py --output-dir artifacts/review_demo_saved --observations artifacts/evaluation/e2e_v2/serial_damage960_conf035/run_20261027_112250_854978
$env:CLAIMS_REVIEW_DEMO = '1'
$env:CLAIMS_REVIEW_DEMO_DIR = (Join-Path $PWD 'artifacts/review_demo_saved')
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

This reads saved results and their source fixtures, not benchmark expectations.
It never launches the benchmark. Real-run provenance is required; deterministic
fake model outputs are rejected. A visual claim appears only if an eligible
saved result exists. Original images must be available and have provenance;
otherwise their unavailability is explained. Saved results are historical
snapshots, not new measurements. The repository redistributes no CarDD photos.

### Full-stack demo

This mode reads the actual PostgreSQL repositories, including stored assessment
and agent histories. Start the existing services and apply migrations:

```powershell
Remove-Item Env:CLAIMS_REVIEW_DEMO -ErrorAction SilentlyContinue
Remove-Item Env:CLAIMS_REVIEW_DEMO_DIR -ErrorAction SilentlyContinue
docker compose up -d --wait postgres qdrant
$env:DATABASE_URL = 'postgresql+psycopg://claims:claims@127.0.0.1:5433/insurance_claims'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
$env:CLAIMS_LLM_MODEL = 'gemma3:4b'
$env:CLAIMS_REVIEW_ASSET_ROOT = (Join-Path $PWD 'data')
$env:DAMAGE_MODEL_PATH = (Join-Path $PWD 'artifacts/vision/damage_segmentation_v1/train/weights/best.pt')
$env:PART_MODEL_PATH = (Join-Path $PWD 'artifacts/vision/vehicle_parts_v1/train/weights/best.pt')
.\.venv\Scripts\python.exe -m alembic upgrade head
# Run once on a new demonstration database; existing business IDs are preserved.
.\.venv\Scripts\python.exe scripts/import_reference_claim.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use the same frontend command, then enter `CLAIM-2026-00001` (or a stored claim
ID). Configure database credentials/ports to match your local Compose overrides.
Ollama must be served separately with the chosen model already installed when
explicit generation is needed. Knowledge/similar collections and local vision
checkpoints are optional; see their existing setup sections below. No model is
trained, downloaded or run by opening the console. Run expensive assessments
explicitly through existing services/endpoints, serially, one claim/image at a
time. The UI itself issues only reads.

`GET /v1/claims/{claim_id}/review` aggregates saved results in the existing
FastAPI app. `GET /v1/review/demo-claims` lists prepared demo claims (empty in
database mode). Registered images are served from
`CLAIMS_REVIEW_ASSET_ROOT/<claim_id>/images/<filename>` through an allowlisted
read route. Each section reports availability, provenance and timestamps.
Database document records currently retain availability, not extraction or
consistency snapshots; these facts are shown when present in prepared demos.
RAG and similar context can be recovered from stored agent evidence; a standalone
RAG answer or retrieval similarity is left unavailable when it was not saved.

### Console validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests -m unit --ignore=tests/e2e
.\.venv\Scripts\nk-python.exe check src/claims/review src/api/main.py src/api/review.py src/api/version.py tests/api/review scripts/prepare_review_demo.py
Set-Location frontend
npm test
npm run lint
npm run format:check
npm run build
```

The production frontend is emitted to `frontend/dist/`; `npm run preview`
previews that build with the same local API proxy. Investigation priority is
review prioritization, not a fraud probability. Anomaly detection identifies
unusual patterns; it is not a fraud classifier. Not visually confirmed does
not mean declared damage is absent. **HUMAN REVIEW REQUIRED** remains visible
even when no agent recommendation is available.

## Automated test infrastructure

From an activated `.venv`, `pytest` runs **all unit and integration tests**.
Pytest is the test interface; no shell runner or custom `--all` option is needed.
Each normal run prints a short table of available suites once.
Agents must exclude `tests/e2e` unless the user explicitly requests end-to-end
tests, as required by `AGENTS.md`. The `unit` marker alone does not exclude them.

```powershell
# When needed, start the test containers yourself before running integrations:
docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test
pytest --ignore=tests/e2e
pytest -m unit --ignore=tests/e2e
pytest -m postgres
pytest -m qdrant
pytest -m rag
pytest -m vision
pytest --markers
```

`postgres-test` uses PostgreSQL 18, `insurance_claims_test`, dedicated user
`claims_test` (local test password `claims_test`), loopback port **5434**, and
tmpfs storage. `qdrant-test` uses loopback ports **6335/6336** and tmpfs. Both
belong to the Compose `test` profile. The application services and their volumes
are separate. Pytest only inspects existing services; it never starts Docker,
containers, PostgreSQL, Qdrant, MLflow or Ollama, and never installs dependencies.
Missing Docker, a stopped engine/container, an unhealthy container or a refused
connection produces an explicit diagnostic with the appropriate manual command.

Defaults are centralized in `tests/infrastructure/integration_support/_constants.py`. CI may override
`TEST_DATABASE_URL`, `TEST_QDRANT_URL` and `TEST_QDRANT_API_KEY`; application
`QDRANT_URL` does not select the test server. PostgreSQL URLs must identify an
authorized `*_test` database and pass the safety checks before any DDL. Every
run owns a random schema, runs the existing Alembic migrations and cleans only
that schema. Transactions roll back between tests. Qdrant uses reserved UUID
collections and cleans only those collections. An explicitly selected missing
service, invalid connection, missing model or checkpoint fails the suite.

Markers are `unit`, `integration`, `postgres`, `qdrant`, `rag`, `vision` and `ocr`.
Real OCR tests additionally require the explicit `RUN_OCR_TESTS=1` opt-in.
Suites live under their domains (`tests/agent`, `tests/domain/claims`, `tests/intelligence/vision`,
etc.). Shared doubles live in `tests/infrastructure/support`; domain fixtures stay beside the
suites that use them. PostgreSQL setup/migrations and prerequisite probes live
in `tests/infrastructure/integration_support` and `tests/infrastructure/integration_prerequisites`.
The deterministic CI command remains `python -m pytest tests/e2e -m unit`.

The [directory layout](docs/DIRECTORY_LAYOUT.md) groups source packages and test
suites by responsibility, within the eight-subdirectory limit checked by
`nk-python check src tests`.

`pytest -m postgres` and `pytest -m qdrant` include the PostgreSQL/Qdrant/API
integration and therefore need both test services. For PostgreSQL alone use `pytest -m 'postgres and not qdrant'`;
for Qdrant alone use `pytest -m 'qdrant and not postgres'`;
`pytest -m 'postgres or qdrant'` covers all database/service boundary tests.
Without `-m`, all tests are selected. Use
`pytest --collect-only -q -m integration` to inspect the integrations.
Prerequisites are checked after selection: `unit` requires no Docker, and `vision`
requires neither PostgreSQL nor Qdrant. External CI connection overrides are
checked directly without requiring a local Docker installation.

Use **Python 3.12** and one **`.venv`** with `.[dev,vision,cardd-hf]` for the
application, RAG, agents, pytest, YOLO and FiftyOne data preparation.
The default interpreter version is recorded in `.python-version`.
The real vision case executes directly in this interpreter, with no
delegation, child pytest or `VISION_PYTHON` setting. On this NVIDIA workstation
`.venv` uses torch **2.14.1+cu132**, torchvision **0.29.1+cu132** and Ultralytics
**8.4.176**. CPU-only Windows/Linux and macOS remain supported; CUDA is optional.
See [installation instructions](docs/INSTALLATION.md) for explicit CPU/CUDA wheel
selection on a new machine. Pytest never changes the environment.

Pytest does not download models. RAG uses the three
small synthetic documents under `data/knowledge`, the 13-case retrieval benchmark
and the cached `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model.
If needed, populate this cache deliberately before testing:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

Vision uses the existing local damage checkpoint; it needs no dataset and never
downloads weights. `DAMAGE_MODEL_PATH` and `DAMAGE_DEVICE` remain optional overrides.
Use normal pytest reporting, for example `pytest --junitxml=artifacts/pytest.xml`.
The vision case appears once in the same pytest/JUnit result. `--help`,
`--markers` and `--collect-only` do not display the suite
panel or check service prerequisites. See the `Final pytest developer experience`
section of `CODEX_REPORT_TEST_INFRASTRUCTURE.md` for validation and failure probes.

To stop only the test services: `docker compose --profile test stop postgres-test qdrant-test`.

## Computer Vision: CarDD HF mirror damage segmentation

The development source is **CarDD HF mirror (`harpreetsahota/CarDD`)**, a
FiftyOne export whose README reports 2,816 samples. It is a subset/mirror of
CarDD, and our workflow uses a **project split**, not the official CarDD splits.
The detector observes physical damage types and normalized image regions with
YOLO26 instance segmentation. `VisualDamageType` is separate from the claim's
vehicle parts; this step adds no part localization or investigation signal.

The mirror includes `CarDD_license.pdf`. Its hosting on Hugging Face does not
necessarily change the conditions of the original images. Keep that provenance
and the original conditions; the repository does not redistribute photographs.
See the [source repository](https://huggingface.co/datasets/harpreetsahota/CarDD)
and [original CarDD project](https://cardd-ustc.github.io/). Ultralytics uses
AGPL-3.0 or an Enterprise licence depending on use; see its
[licensing page](https://www.ultralytics.com/license).

Dataset loading uses the official
[FiftyOne Hugging Face integration](https://docs.voxel51.com/integrations/huggingface.html),
`fiftyone.utils.huggingface.load_from_hub()`, rather than relying on
`datasets.load_dataset()` to recover labels. Use the same Python 3.12 `.venv`
for data preparation and the application. The optional `cardd-hf` extra installs
FiftyOne and its Hugging Face integration alongside `dev` and `vision`;
a separate `.venv-cardd` is unnecessary. Select the CUDA or CPU torch build
explicitly as described in the installation instructions.

PowerShell from the repository root, after installing the Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vision,cardd-hf]"
$env:CARDD_HF_DATASET = "harpreetsahota/CarDD"
$env:CARDD_HF_REVISION = "56900bde8dddfe00eb7c03114a1d46e9105e3cdb"
$env:CARDD_HF_CACHE_DIR = "C:\datasets\cardd_hf_v1"
$env:CARDD_HF_PREPARED_DIR = "C:\datasets\cardd_hf_v1\prepared_yolo_v2"
.\.venv\Scripts\python.exe .\scripts\inspect_cardd_dataset.py --download
.\.venv\Scripts\python.exe .\scripts\prepare_cardd_yolo.py --download --converter native
```

`--download` explicitly enables loading/downloading through FiftyOne. Subsequent
calls reuse the persistent local dataset and files after validating provenance.
`--dataset-name NAME --cache-dir PATH` inspects/reuses an existing local FiftyOne
dataset with stored source metadata. `--revision SHA`, `--repository`,
`--annotation-field`, `--cache-dir`, `--output-dir`, `--manifest` and
`--split-manifest` provide explicit overrides. The actual source SHA is recorded
and used for every download. Source storage, Hugging Face cache and the local
FiftyOne database can all live outside the repository. Without a storage override,
the cache defaults to the user's local application data directory. The prepared
data fallback `data/external_prepared/cardd_hf_v1/` is ignored by Git. Images,
FiftyOne files, caches, generated runs and model weights are never committed.

Inspection reads the actual field schema and labels, finds the usable instance
segmentation field, counts samples/instances/classes, checks dimensions, and
reports duplicate filenames, filepaths and full-image SHA-256 hashes. Bounding
boxes alone or an unclassified semantic mask are insufficient: the script stops
with an explicit error if usable instance segmentations are absent. Field names
are discovered rather than assumed. The mirror's original `detections` and
`segmentations` names are observations of this source, not a universal contract.
Masks are traced into foreground pixel-cell contours; supplied polylines retain
their vertices. Hole rings and multipart contours use retraced bridges for YOLO's
single-contour representation. Rasterized bridges can add pixels; this limitation
and topology counts are recorded. No mask is synthesized from a bounding box.

A small deterministic multilabel allocation balances image/class representation
while preserving exact sizes, default ratios **70% / 15% / 15%** and **seed 42**.
Images with identical encoded bytes, filename or filepath are grouped into one
split, so exact duplicate photographs cannot leak between splits. Near-duplicate
perceptual similarity is not automatically detected. The frozen manifest is
`artifacts/vision/cardd_hf_split_v2.json`: source SHA, stable sample IDs,
source/content/annotation fingerprints, classes, ratios, seed, split IDs,
distributions and split version. Existing manifests are validated and reused;
source or parameter mismatches fail instead of regenerating the same version.
Rare-class or grouping constraints that prevent balanced coverage are documented.
The initial `cardd_hf_split_v1.json` is retained unchanged: inspecting its class
distribution revealed excess scratch images in TRAIN. Version v2 adds deterministic
group swaps to balance all labels while retaining exact split sizes. This change
was made before any model training or evaluation. Existing v1 manifests can still
be explicitly validated/reused; they are never silently regenerated as v2.

Preparation applies those frozen IDs and writes image/label pairs and
`dataset.yaml`. It validates class indices, normalized polygons, exact inventories,
instance counts, source-to-prepared image hashes, split membership and label hashes.
The portable manifest is `artifacts/vision/cardd_hf_dataset_manifest.json`, with a
copy named `cardd_dataset_manifest.json` in the prepared directory for shared
training validation. `--converter native` requires no Ultralytics installation in
the data environment; `auto` can use the compatible official COCO converter on
staged polygons with `use_segments=True, cls91to80=False`, after strict checks.
Existing output directories are never overwritten. The old local COCO adapter is
available only through explicit `--source coco`; it is not this development source.

Training configuration defaults to
`configs/vision/cardd_hf_yolo26n_seg_v1.toml`: pretrained `yolo26n-seg.pt`, 100
epochs, imgsz 640, batch 8, patience 20, seed 42, workers 0, optimizer `auto`,
CUDA device `0` (RTX 3070 Laptop, 8 GiB baseline). `DAMAGE_BASE_MODEL` overrides the initial weights;
`DAMAGE_DATASET_YAML` points at prepared data outside the repository. Effective
configuration, source/split provenance, actual device and versions are logged.
This is a seeded/reproducible configuration, with no bitwise GPU determinism
promise. Check CUDA in the actual training interpreter, since the global Python
and the application's virtualenv may use different torch builds.

For VAL diagnostics and independent resolution, model-capacity and mask-training
experiments using the same prepared dataset, see
[Améliorer Damage avec le dataset actuel](docs/DAMAGE_MODEL_IMPROVEMENT.md).

```powershell
# Use the same .venv for the application, CUDA vision and pytest.
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$env:DAMAGE_DATASET_YAML = "$env:CARDD_HF_PREPARED_DIR\dataset.yaml"
$env:DAMAGE_CONFIDENCE_THRESHOLD = "0.25"
$env:DAMAGE_DEVICE = "0"
.\.venv\Scripts\python.exe .\scripts\train_damage_segmentation.py
# Resume the same interrupted run (keep the original config, data and threshold):
# .\.venv\Scripts\python.exe .\scripts\train_damage_segmentation.py --resume
.\.venv\Scripts\python.exe .\scripts\evaluate_damage_segmentation.py
.\.venv\Scripts\python.exe .\scripts\check_damage_test_smoke.py --max-images 12
.\.venv\Scripts\python.exe .\scripts\check_claim_vision.py
.\.venv\Scripts\python.exe -m pytest -q
```

Use `--device cpu` explicitly on an interpreter without CUDA. Check the training
interpreter's CUDA build before starting; the single `.venv` may use CPU or CUDA wheels.
TRAIN optimizes, VAL selects the best checkpoint and early stopping, and TEST
remains frozen for final evaluation. TEST never selects the checkpoint,
hyperparameters or threshold. `DAMAGE_CONFIDENCE_THRESHOLD` defaults to the
uncalibrated 0.25 baseline and is recorded before TEST. Test metrics preserve
box/mask precision, recall, mAP50 and mAP50-95 and available per-class metrics;
unavailable values stay null. Weak measured classes and distributions accompany
global scores. **Metrics are evaluated on the project's deterministic held-out 422-image test split of the 2,816-image Hugging Face CarDD mirror, not on the official CarDD test split.** There are no official benchmark scores implied by these outputs.

Generated artifacts live under `artifacts/vision/damage_segmentation_v1/`:
`train/weights/best.pt`, `train/weights/last.pt`, `train/results.csv`,
`train/results.png`, `training_progress.json`, `training_summary.json`,
`test_metrics.json`, `per_class_metrics.csv`, bounded `error_examples/` and
`test_smoke/` with real domain assessments and annotated overlays. Evaluation
verifies the exact VAL-selected checkpoint hash and dataset/split provenance.
Error matching is a class-aware greedy box-IoU approximation, not exact mask
FP/FN analysis or a representative estimate of every failure mode. Correct and
incorrect segmentation examples also use diagnostic polygon raster IoU, separate
from framework mask AP. The TEST smoke follows completed evaluation and validates
checkpoint hash, provenance and the original threshold before inference.

Interrupted runs resume with `train_damage_segmentation.py --resume` using the
same `train/weights/last.pt`, optimizer/early-stopping state and original total
epoch budget. `recovery_training_metadata.json` locks config, threshold, dataset
provenance and checkpoint hashes; it is refreshed after every saved epoch.
Completed or evaluated runs cannot be resumed. `best.pt` stays selected on VAL
across the interruption. Native `results.csv` time restarts on resume;
`training_summary.json` records cumulative compute time and excludes downtime.

`UltralyticsDamageDetector` loads an existing local trained checkpoint lazily,
once per instance. `DAMAGE_MODEL_PATH` and `DAMAGE_DEVICE` configure inference.
Image validation includes decoding, format/size/dimensions and EXIF orientation.
`ClaimVisionService.assess_claim(case)` reuses `ClaimCase`, `ImageAsset` and
`resolve_image_assets()` and preserves per-image boxes, optional polygons and
confidence. Repeated observations are not counted as unique physical damage.
`check_claim_vision.py` saves annotated masks/boxes/classes/confidence and JSON
under `artifacts/vision/reference_claim/` only after successful real inference.
The reference JPEGs currently are empty placeholders. Debug images never replace
the structured output. The claim-aware HTTP endpoint remains deferred because
PostgreSQL filenames do not provide a physical claim storage root.

Empty detections mean **no supported damage detected above the configured
confidence threshold**, not no damage. Confidence is not a probability of claim
validity. Only the training taxonomy is supported; real insurance photos may
differ from this mirror. No OCR, LLM detection, vehicle-part mapping or
image/document consistency is used in this step.

MLflow is enabled by `track_mlflow = true` or an explicit `MLFLOW_TRACKING_URI`
and uses the separate `insurance-claims-vision-damage` experiment. It records
source revision, frozen split, seed/config/device/versions and measured metrics,
with JSON/CSV/TOML artifact allowlisting and optional checkpoint upload.
Raw photographs and error images are excluded; built-in image logging callbacks
are removed. Vision Registry packaging remains deferred, with no production
alias. `pytest -m unit` selects synthetic unit fixtures/stubs and requires no FiftyOne,
MongoDB, dataset, weights, GPU or network. Real model tests are selected with
`-m vision`, require a local checkpoint and never download/train. The durable
execution record and actual provenance are in `CODEX_REPORT_STEP_29.md`.

Run the existing local damage checkpoint integration from the application environment:

```powershell
pytest -rs -m vision
```

This decoding/inference smoke needs no dataset; CPU is also supported. Once selected,
missing Ultralytics or local weights fail the test explicitly.
The `integration` marker groups the real PostgreSQL, Qdrant, RAG and vision
tests. Use `pytest -m 'not integration'`
for the lightweight suite, or `pytest --collect-only -q -m integration` to list
the integrations. `CODEX_TEST_SKIP_AUDIT.md` preserves the earlier skip audit;
the current automatic configuration is described in `CODEX_REPORT_TEST_INFRASTRUCTURE.md`.

The completed local baseline ran 100 epochs on CUDA and selected epoch 90 on VAL.
On the frozen 422-image TEST split, mask mAP50 is 66.03% and mAP50-95 is 52.24%.
Per-class mask mAP50-95 ranges from 9.72% for crack to 93.82% for glass shatter;
the report contains the full box/mask tables and observed failures. Twelve real
TEST inference images produced 21 domain detections with polygons, and all saved
overlays were visually reviewed. Domain polygon overlays use external contours,
so holes may be filled; framework mask AP uses binary masks. MLflow/Registry are
deferred, and reference-claim inference remains unavailable because its JPEGs
are empty. These measurements describe this project's split of the HF mirror.

## Meaning of the outputs

- **Inconsistency** = disagreement between available evidence, such as two documents reporting different incident dates.
- **Anomaly** = statistical atypicality relative to a reference population. The listed unusual features are descriptive observations, not exact model attributions.
- **Investigation signal** = an explicit reason for additional human review, backed by structured evidence.
- **Fraud determination** = outside the scope of this component.

`review_score` is a deterministic heuristic review-prioritization score defined by a versioned policy. It is not a probability of fraud.

The investigation engine in `src/claims/investigation/` translates a `ClaimConsistencyReport`, an `AnomalyAssessment`, and document completeness fields in `TriageFeatures` into ordered signals. The engine does not consume the triage workflow prediction. Neither an anomaly nor a review recommendation changes the operational triage output.

## Investigation policy

`investigation_policy_v1` is preserved in `src/claims/investigation/policy.py` as `INVESTIGATION_POLICY_V1`. The default is now `INVESTIGATION_POLICY_V2` (`investigation_policy_v2`), which keeps all original weights and thresholds and adds the historical rules described below. Its weights are transparent demonstration choices, not insurance industry standards.

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
python -m pytest -m unit
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
rollback. `pytest -m unit` selects unit tests without connecting to PostgreSQL.
Selecting the PostgreSQL marker uses the dedicated service automatically;
connection or configuration failures fail explicitly.

```powershell
docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test
python -m pytest -m postgres
```

The test database is initialized automatically on port 5434 and uses ephemeral
storage. No database is created inside the application PostgreSQL service.
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

Run unit tests with `pytest -m unit`; select `pytest -m postgres` after manually starting
the dedicated test services described above. Integration tests cover
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
API import, lifespan startup and unit tests do not load it. One provider
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
python -m pytest -m unit
# Select real test services; unique test collections are cleaned up.
docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test
python -m pytest -m 'qdrant and not postgres'
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
Model loading stays lazy; unit tests use fake providers and in-memory Qdrant.

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
python -m pytest -m unit
# Real retrieval integration (cached model, no Ollama); unique test collections:
docker compose --profile test up -d --no-deps --wait qdrant-test
python -m pytest -m rag
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
When a collected triage assessment recommends `EXPERT_REVIEW` or `INVESTIGATION`,
an anomaly assessment records `is_anomalous=true`, or an investigation assessment
records `ADDITIONAL_QUOTE_DAMAGE`, `UNMAPPED_QUOTE_ITEM`, `RECENT_CLAIM_FREQUENCY`,
`ANOMALOUS_PATTERN`, or `EXTREME_ANOMALY_PERCENTILE`, the generation schema and
validator require a follow-up action (additional information, expert review, or
investigation review) with a rationale citing a triggering assessment.
`NORMAL` investigation priority or `review_recommended=false` does not cancel
these triggers. Frequency scoring remains 15 points; all ML and investigation
thresholds are unchanged. A follow-up is human workflow guidance and does not
establish fraud.
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
trace. `pytest -m unit` uses fake planners/finalizers/tools and needs no external
services. PostgreSQL integrations are selected with `-m postgres` and use the
dedicated test database automatically.
Small local models may plan imperfectly, and recommendations depend on evidence
quality/availability. This V1 has no long-term conversational memory or
quantitative agent benchmark. `CODEX_REPORT_STEP_28.md` records actual validation.

## Vehicle parts and multimodal perception (STEP30)

The independent Carparts-Seg model starts from COCO `yolo26n-seg.pt`. It shares
the step29 image loader, normalized geometry, Ultralytics extraction helpers and
training/evaluation orchestration. Preparation downloads only the official
archive when explicitly requested, checks every supplied split and writes
`artifacts/vision/carparts_dataset_manifest.json`. Imports and unit tests
never download images or weights.

Run the explicit baseline from a CUDA-capable interpreter. The same interpreter
prints its Python executable, torch/CUDA build, CUDA availability and GPU before
training. A requested GPU with unavailable CUDA fails; there is no CPU fallback.
`best.pt` is selected from VAL; TEST is evaluated only after training, without
changing the baseline from TEST results. Both best and last checkpoints are kept.

```powershell
# From the repository root:
.\.venv\Scripts\python.exe scripts/prepare_carparts_dataset.py
.\.venv\Scripts\python.exe scripts/inspect_carparts_dataset.py
.\.venv\Scripts\python.exe -u scripts/train_vehicle_part_segmentation.py
# Run final TEST once, after completed training:
.\.venv\Scripts\python.exe scripts/evaluate_vehicle_part_segmentation.py
$env:DAMAGE_DEVICE = '0'
$env:PART_DEVICE = '0'
.\.venv\Scripts\python.exe scripts/check_multimodal_vision.py
```

The part experiment is `insurance-claims-vision-parts`, with metadata, VAL/TEST
metrics and the selected checkpoint logged to a separate local SQLite MLflow
store by default. An explicit `MLFLOW_TRACKING_URI` takes precedence. Dataset
images are excluded from logging. Custom vision Registry packaging remains
deferred; damage and part checkpoints have independent paths and SHA-256 versions.

`ClaimMultimodalVisionService` infers each photo once per model and then fuses its
observations. Shapely computes mask intersections in normalized image space;
missing masks use an explicit `BBOX_FALLBACK`. Invalid supplied polygons fail
validation for fusion. The primary measure is damage containment, since a small
scratch may be fully within a large bumper with a very small IoU. The versioned
`vision_fusion_v1` baseline requires damage confidence >=0.5, part confidence
>=0.5 and containment >=0.6; a containment gap <=0.05 between eligible parts
stays ambiguous. These thresholds await calibration on VAL.

Both adapters request Ultralytics native image masks (`retina_masks=True`) and
convert their binary foreground pixel cells into the existing `NormalizedPolygon`
model. Shapely preserves holes and disconnected components; visualization renders
them too. This avoids invalid merged `masks.xyn` contours without silently repairing
polygons or substituting boxes for valid masks. Unexpected mask dimensions fail
explicitly, protecting the shared EXIF-oriented coordinate space.

An inference-only integration smoke reuses the previous local CarDD demonstration
photos. It creates one independent, explicitly synthetic diagnostic claim context
per photo, loads each local checkpoint once, and feeds the same photo to both
models. No training, statistical evaluation, threshold tuning or database writes
occur. Real declarations and quotes are unknown; its consistency/signals must be
read as conditional on its synthetic inputs. Each run gets a new artifact directory.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/check_multimodal_integration_smoke.py --device 0 --max-images 12
```

`object` becomes `VehiclePartType.OTHER` and is excluded from business fusion.
Only front bumper, hood and front left light map to existing business damage
types; generic lights, unknown sides and other parts remain unmapped. Quote
comparison reuses `claims.consistency.map_quote_damage_types`.

`investigation_policy_v3` copies v2 and adds 12 heuristic demo points for each
additional reliable mapped damage category. Each signal preserves all relevant
photo observations and model/fusion snapshots. Positive support and declared
damage without visual confirmation add no points. V1/v2 and the v2 default stay
unchanged. Two photos are separate observations, without physical-damage
deduplication.

Migration `0004` adds append-only `vision_assessments` with normalized polygons
and separate model versions. `VisionAssessmentStore` saves already computed
results. `GET /v1/claims/{claim_id}/vision/latest` and agent
`GET_VISUAL_ASSESSMENT` retrieve a stored snapshot without loading models. An
inference HTTP route is deferred because the current database has image filenames
but no authoritative filesystem claim directory. The filesystem service and
reference script provide the inference path.

The durable `CODEX_REPORT_STEP_30.md` records actual execution status, GPU proof,
counts, metrics, tests and limitations. Reference photos that are empty or invalid
produce a saved `NOT EXECUTED` status instead of invented observations.

## End-to-End Evaluation

For a French explanation of **why each test exists and how the AI should react**,
start with the [business scenarios](tests/e2e/SCENARIOS.md), including repeated
claims as a potential fraud risk requiring investigation. The
[test guide](tests/e2e/README.md) also explains the supporting checks.

The versioned `e2e_claims_v2` benchmark covers ten controlled golden scenarios
from document ingestion and structured extraction through investigation, RAG,
multimodal evidence and advisory agent recommendations. Fast deterministic tests
under `tests/e2e/` run offline in ordinary pytest; real PostgreSQL, Qdrant,
embeddings, Ollama and local model evaluation run through the dedicated script.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e -q
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --preflight-only
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --isolated-demo-stack --database-url 'postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test' --qdrant-url 'http://127.0.0.1:6335' --llm-model gemma3:4b --include-vision --include-agent --include-rag --include-similar-claims --output-dir artifacts/evaluation/e2e_v2/real
```

The isolated mode validates the explicit test database and uses a temporary
migrated PostgreSQL schema plus temporary Qdrant collections. It seeds only
controlled demo data, then cleans up its own schema/collections. Preflight stays
read-only. The command uses the existing installed local model and test-service
ports; no model download is performed.

To run all ten golden cases automatically with a 30-second pause between cases:

```powershell
.\.venv\Scripts\python.exe -u scripts/evaluate_end_to_end_suite.py `
  --mode real `
  --config configs/evaluation/e2e_v2_damage960_val640.toml `
  --llm-model claims-e2e-partiel `
  --include-vision --vision-device 0 --damage-confidence-threshold 0.35 `
  --no-include-diagnostics --isolated-demo-stack `
  --database-url "postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test" `
  --qdrant-url "http://127.0.0.1:6335" `
  --cooldown-seconds 30 `
  --output-dir artifacts/evaluation/e2e_v2/serial_damage960_conf035
```

Cases run in manifest order in separate child processes, with vision enabled only
for the last two. Damage is still unloaded before Part is loaded; the existing
temperature guard remains active. Choose `--cooldown-seconds 60` or `120` for a
longer pause; a fixed delay cannot guarantee GPU temperature. Add `--dry-run` to
preview without execution or output files. Each run retains scenario reports and
an incremental `suite_summary.json` under a unique output subdirectory. The suite
stops at the first failed, skipped or incomplete case by default;
`--no-fail-fast` continues ordinary failures, while a vision safety stop always
stops the suite. See the [suite guide](docs/END_TO_END_EVALUATION.md#run-all-ten-scenarios-with-pauses)
for details.

The manifest is `data/evaluation/e2e_v2/scenarios.json`. Reports distinguish
`PASS`, `FAIL` and `SKIP`; a skip never counts as a pass, and an unavailable
mandatory component leaves the run `PARTIAL`. The two empty reference JPEGs
remain unchanged. Separate local vision diagnostics preserve asset/model
provenance and do not imply golden-label accuracy. JSON/CSV artifacts and
structured traces are saved under `artifacts/evaluation/e2e_v2/`.

In real mode, each golden case's `real_agent_action_policy` requires follow-up
when the observed assessments contain one of the triggers above; otherwise its
explicit alternative actions apply. The unusual-claim case permits standard
review when no trigger is observed, instead of imposing the deterministic anomaly
label on a real model. Missing assessments block the conditional check.
Deterministic expectations and all model thresholds are unchanged. The v1
manifests and configuration remain preserved; v1 and v2 baselines cannot be compared.

The benchmark validates expected behavior on controlled demo scenarios. It is not a measurement of real-world insurance decision accuracy.

See [the evaluation guide](docs/END_TO_END_EVALUATION.md) for configuration,
golden cases, metrics, safety/provenance checks, stable baseline comparison and
limitations. [CODEX_REPORT_STEP_31.md](CODEX_REPORT_STEP_31.md) records actual
deterministic and real-stack execution results.

## Document AI robustness evaluation

`document_ocr_v1` measures document processing independently of the E2E system
benchmark. Native-text PDFs alone do not exercise Tesseract. This CPU-only
benchmark uses the existing `extract_pdf_text_auto` and the real deterministic
claim-form, accident-report and garage-quote extractors. It loads no vision
checkpoint, YOLO, CUDA, Ollama or external service. The optional product LLM
extractors remain a separate path and are not measured here; no LLM retry is
hidden in these results.

Five synthetic dossiers provide 15 source documents and 120 cases. Ground truth
is constructed from controlled structured demo data before PDF generation,
including a coherent dossier, additional HOOD damage and a deliberate date
mismatch. Only fields present in the current product models are scored; those
models do not contain customer identity fields. The implementation lives under
`evaluation/deterministic/document_robustness` to respect the repository's
directory limit, with public entry points in `evaluation.document_robustness`.

`native_clean` retains native text. The image-only variants are `scan_clean`,
`scan_low_resolution`, `scan_jpeg_compressed`, `scan_rotated_small`,
`scan_blurred` and `scan_noisy`. `mixed_native_and_scan` splits the source text
between a native page and an image-only page. Actual ingestion methods are
recorded and checked as `NATIVE`, `OCR` or `HYBRID`. Scan generation uses 180 DPI
(100 for low resolution), JPEG quality 35, rotation 2 degrees, blur radius 0.65
or Gaussian noise with standard deviation 8 as applicable. Noise has a stable
per-document SHA-256-derived seed; every case retains its parameters.

CER/WER use Levenshtein edit distance after whitespace-only normalization.
Digits, dates, amounts, punctuation and case remain unchanged. Aggregate rates
sum edit counts and reference lengths instead of averaging document percentages.
Pure OCR rates exclude native and hybrid documents; hybrid CER/WER describe the
whole mixed document. These text diagnostics are distinct from business accuracy.
Field exact match and required-field recall count correct required values;
required-field presence is reported separately. Amounts use numeric Decimal
equality, damages use sets, and quote items match exact description, numeric
amount and currency with duplicate multiplicity. Critical accuracy separately
tracks claim ID, incident date, repair total, currency and damages.

`consistency_preservation_rate` is the fraction of evaluable degraded dossier
variants retaining every expected fact from the real `ClaimConsistencyReport`.
Evaluation requires all three documents to match their source ground truth.
The eligible count and excluded count matter: this conditional rate is not a
claim that every degraded dossier succeeded. Monetary OCR ambiguities and
conflicting duplicate critical fields fail closed; a valid but wrong date must
still be detected against ground truth or another document.

Run from the repository with the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --preflight-only
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --resume
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --document-type garage_quote --output-dir artifacts/evaluation/document_ocr_v1/quotes
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --variant scan_rotated_small --output-dir artifacts/evaluation/document_ocr_v1/rotated
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --scenario demo_03 --output-dir artifacts/evaluation/document_ocr_v1/one_scenario
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --scenario demo_01 --document-type claim_form --variant scan_clean --output-dir artifacts/evaluation/document_ocr_v1/one_case
```

Selections are repeatable and always run serially. Use a fresh output directory,
or `--resume` to reuse immutable results and append missing cases. Resume rejects
changed corpus, parameters, environment or implementation. An interrupted source
PDF is reused only if its bytes match deterministic regeneration. Recorded
extraction failures are retained; retrying them requires a new output directory.
The English corpus explicitly uses `eng`; preflight prints actual package,
Tesseract and installed-language versions. `--language` and `--tesseract-cmd`
select existing local installations, never install language packs. If OCR is
unavailable the status is `OCR benchmark: NOT EXECUTED` with exact reasons,
unexecuted cases are excluded from aggregate scores, their text and stage
measurements remain null, and the CLI returns 2.

The default output is `artifacts/evaluation/document_ocr_v1/`: manifest,
`ground_truth.json`, `summary.json`, `per_document.csv`, `per_variant.csv`,
`per_document_type.csv`, `critical_fields.csv`, raw text in `ocr_text/`,
per-case JSON in `results/`, generated PDFs in `sources/`, and field/context
diagnostics in `failures/`. These generated artifacts remain ignored by Git.
Latency separates native ingestion, OCR including rendering, and structured
extraction; total case duration also includes synthetic materialization and
text-metric computation. Unavailable stage timings are null.

A complete run can create a new baseline with `--write-baseline PATH`; existing
baselines are never overwritten. The measured reference is
`configs/evaluation/document_ocr_v1_baseline.json`. To compare future complete
runs, add `--baseline configs/evaluation/document_ocr_v1_baseline.json` with a
fresh output directory. Comparisons require matching corpus, configuration,
OCR environment and document/variant population. Declining critical accuracy,
new parse failures or declining consistency preservation produce regression
records and exit code 1. Meaningful changes to scenarios, degradations, ground
truth or metrics require a new benchmark version; pytest imposes no arbitrary
CER/WER threshold.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/evaluation/document_robustness -m unit --ignore=tests/e2e --basetemp=.pytest-tmp-document-unit
$env:RUN_OCR_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests/evaluation/document_robustness/test_real_ocr.py -m ocr --ignore=tests/e2e --basetemp=.pytest-tmp-document-ocr
Remove-Item Env:RUN_OCR_TESTS
```

Real Tesseract tests have the `ocr` marker and skip by default without opt-in.
The ordinary deterministic suite uses doubles for OCR and never needs its
executable. E2E tests remain separate and require explicit authorization.
The corpus uses synthetic layouts and degradations, is Tesseract-specific,
does not evaluate handwriting, and does not represent all insurer layouts or
general production OCR accuracy. Actual results and limitations are recorded in
[CODEX_REPORT_STEP_32.md](CODEX_REPORT_STEP_32.md).
