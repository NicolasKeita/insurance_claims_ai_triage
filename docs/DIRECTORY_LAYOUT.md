# Python package and test layout

Run `.venv\Scripts\nk-python.exe check src tests` from the repository root.
Each directory has at most eight direct subdirectories, excluding the generated
directories ignored by the linter. The default thresholds remain unchanged.

Source imports follow these responsibility groups:

| Previous import | Current import |
| --- | --- |
| `investigation` | `claims.investigation` |
| `knowledge` | `retrieval.knowledge` |
| `ml.anomaly_evaluation`, `ml.anomaly_registry`, `ml.anomaly_tracking` | `ml.anomaly.evaluation`, `ml.anomaly.registry`, `ml.anomaly.tracking` |
| `vision.cardd_hf`, `vision.cardd_hf_preparation`, `vision.cardd_hf_split`, `vision.carparts_dataset` | The same names under `vision.dataset` |
| `vision.detector`, `vision.evaluation`, `vision.error_analysis`, `vision.training`, `vision.tracking` | The same names under `vision.damage` |
| `vision.parts_detector`, `vision.parts_evaluation`, `vision.parts_training`, `vision.parts_tracking` | `vision.parts.detector`, `vision.parts.evaluation`, `vision.parts.training`, `vision.parts.tracking` |

Imports in application code, scripts and tests use these paths. HTTP routes,
model identifiers and the default locations under `data/`, `configs/` and
`artifacts/` remain the same. External Python callers should update the imports
listed above.

Tests are grouped under `tests/domain` (claims, investigation, persistence),
`tests/intelligence` (knowledge, ML, retrieval, vision) and `tests/infrastructure`
(isolation, prerequisites, integration helpers and shared doubles).
`tests/agent`, `tests/api` and `tests/e2e` retain their existing paths.
Invalid claim fixtures are under `tests/data/invalid`.

`pytest -m unit` and `pytest tests/e2e -m unit` retain their existing meanings.
The layout regression test uses the linter's directory discovery and also checks
that the relocated packages resolve their default paths from the repository.
