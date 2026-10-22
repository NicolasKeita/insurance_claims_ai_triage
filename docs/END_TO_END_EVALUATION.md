# End-to-End Evaluation

The benchmark validates expected behavior on controlled demo scenarios. It is not a measurement of real-world insurance decision accuracy.

## Two execution levels

`tests/e2e/` is the fast, offline deterministic suite. It exercises existing
document ingestion/extraction, consistency, feature engineering, investigation,
knowledge grounding, vision fusion and agent services. Model predictions,
embeddings and structured LLM responses use explicit deterministic dependencies;
storage is in memory. It requires no PostgreSQL, Qdrant server, Ollama, MLflow,
GPU, external checkpoints or Internet. These tests run in ordinary pytest and CI.

The dedicated `scripts/evaluate_end_to_end.py --mode real` runner uses real
available application services and local assets. It runs separately from pytest,
performs preflight checks, and never substitutes deterministic detectors or an
LLM fake for an unavailable real component. A component disabled or unavailable
has an explicit `SKIP` reason. A report with unavailable mandatory expectations
remains `PARTIAL`, even when every component that actually ran succeeded.

The real CLI first performs lightweight prerequisite checks for each selected
scenario, before loading embeddings, ML checkpoints or vision weights. These
checks combine the manifest's `infrastructure_requirements` with the enabled
agent, RAG and similar-claims stages. They inspect configuration, service
connectivity, the installed Ollama model, source files, local model/cache files,
and existing stored claims and collections. No inference or model download is
performed during these checks. MLflow remains optional when local ML checkpoints
exist; a registered candidate is an alternative when a local checkpoint is absent.

A missing prerequisite immediately excludes that scenario from execution, with
an explicit `SKIP` reason in the terminal, preflight JSON and scenario report.
`--dependency-unavailable FAIL` retains its explicit failure policy. If all selected
scenarios are blocked by these lightweight checks, no model is loaded and
supplemental diagnostics are skipped.
In a mixed selection, eligible scenarios can still run. Isolated setup requires
explicit safe test endpoints, but does not require pre-existing claims or vector
collections: it creates them after the prerequisite checks pass. Checks confirm
presence and availability; successful loading and inference are validated later.

Terminal output summarizes each blocked scenario with the missing upstream
components and corrective actions. Dependent checks are grouped beneath their
cause: an unreachable Qdrant server is one blocker, rather than three repeated
errors for Qdrant and both collection checks. Available prerequisites are shown
with `[OK]`; raw exception details and all individual checks remain in
`preflight.json` and the scenario artifacts. A blocked scenario is not announced
as running. If the lightweight checks block all scenarios, the terminal explicitly
states that no model was loaded and no scenario executed.
When a database blocks the run and `compose.yaml` exists, the terminal also shows
the command to start both local test databases, their ports and the explicit
isolated setup option. This is guidance only; the runner never starts Docker.
On Windows, the hint explicitly asks the user to open Docker Desktop and wait
for its engine to be ready before running the Compose command.

When RAG and similar-claims retrieval use the same embedding model, preflight
loads its weights once and isolated indexing reuses them. Separate provider views
preserve RAG query/document encoding and symmetric claim encoding. Different
configured model names still use separate weights. The eligibility check after
model loading reads the recorded results; it does not repeat service probes.

While a real scenario runs, the terminal prints each operation's start, result
and elapsed time. Long operations emit a heartbeat every five seconds with step
and total elapsed time. Local NVIDIA telemetry is read through `nvidia-smi` when
available: device utilization, VRAM, temperature and power. These are whole-device
measurements, not proof that this test caused a hardware fault. Missing GPU tools
or a telemetry timeout do not stop evaluation. A matching previous recorded run
provides a duration reference; no percentage or guaranteed ETA is invented.

Each attempt keeps a separate `progress/<scenario_id>/<run_id>/` directory:

- `events.jsonl`: timestamped starts, results, heartbeats and handled interruptions;
- `last_state.json`: the latest event, replaced atomically;
- `checkpoint.json`: the latest saved component observations, replaced atomically.

Events and snapshots are flushed and synchronized to disk before starting the
next operation. After a sudden termination, a stage start without a matching
finish identifies the last known active operation; it does not establish the
underlying cause. Hardware or storage failure can still lose recent writes.
An agent run returned with a failed status is recorded as `[FAIL]` immediately,
with its application-owned error reason in the terminal and progress artifacts.
Retries preserve earlier attempt directories. Monitoring begins at scenario
execution, after preflight and isolated setup. It does not run for blocked
scenarios or `--preflight-only`.

## Frozen golden cases

`E2E_BENCHMARK_VERSION = "e2e_claims_v1"` is declared in `src/evaluation/`.
The versioned manifest is `data/evaluation/e2e_v1/scenarios.json`. An important
change to scenarios or their expected behavior requires a new benchmark version.
The loader rejects unknown versions, duplicate IDs and unknown scenario fields.

| Scenario | Required behavior |
| --- | --- |
| `simple_coherent_claim` | Structured extraction and matching document fields; no invented discrepancy. |
| `additional_quote_damage` | HOOD appears beyond declared FRONT_BUMPER, with quote evidence. |
| `missing_documents` | One unavailable document, completeness 3/4, policy-driven factual follow-up. |
| `recent_claim_history` | Inclusive 30/90/365 counts; same-day, future and other-customer claims excluded. |
| `statistically_unusual_claim` | Explicit fake anomaly path; no fraud conclusion. Real predictions remain observations. |
| `policy_answerable` | Expected demo policy and Deductible section retrieved; grounded valid citation. |
| `policy_insufficient_evidence` | Unsupported volcanic-ash question produces insufficient evidence without invented clauses. |
| `agent_missing_tool_data` | Unavailable history tool produces uncertainty and mandatory human review. |
| `visual_additional_damage` | SCRATCH morphology associates with HOOD geometry; policy v3 emits visual evidence. |
| `visual_non_detection` | Unconfirmed LEFT_HEADLIGHT remains informational; non-detection alone adds no risk or review points. |

Every entry declares its source, controlled inputs, files, external assets,
structured expected facts, required/forbidden signals, acceptable agent actions,
knowledge sources, uncertainty and infrastructure requirements. Facts use paths
such as `features.missing_document_count` and
`vision.consistency.additional_visual_damage`. Expected lists, sets and counts
are explicit. Exact generated LLM sentences and an exact ML workflow are not
ground truth. Entries can mark fake-only expectations through
`deterministic_only_facts` and `deterministic_only_signals` in their inputs;
the real runner records the model result without imposing those fake labels.

The controlled documents are generated from manifest inputs in an isolated
temporary directory. Their provenance identifies the synthetic benchmark;
the original reference claim is preserved. Geometric visual golden cases use
explicit synthetic observations rather than third-party photographs. Real vision
cannot establish those particular golden labels without appropriate controlled
image assets, even when model checkpoints are loadable.

`data/evaluation/e2e_v1/diagnostics.json` separately versions two real execution
diagnostics: the original `CLAIM-2026-00001` documents and prior locally available
CarDD photographs with documented source IDs. Diagnostics do not enter the ten
golden cases' pass-rate denominator. The reference claim's two JPEGs are empty;
it is a document-centric case with vision skipped. Local CarDD diagnostics use
synthetic claim context and make no claim about actual declaration accuracy or
insurance ground truth. External images are referenced locally and are not
redistributed by the benchmark.

## Commands

From the repository root in PowerShell:

```powershell
# Deterministic E2E tests only; no service or checkpoint initialization.
.\.venv\Scripts\python.exe -m pytest tests/e2e -q

# The complete standard suite, including existing integrations when available.
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -rs

# Persist a deterministic benchmark run.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode deterministic --output-dir artifacts/evaluation/e2e_v1/deterministic

# Lightweight prerequisite checks only; no model loading or scenario execution.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --preflight-only

# Request the available real components; unavailable components remain explicit.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --include-vision --include-agent --include-rag --include-similar-claims --output-dir artifacts/evaluation/e2e_v1/real

# Run controlled claims against this project's local, isolated test services.
# This uses the already installed model; it never pulls an Ollama model.
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
.\.venv\Scripts\python.exe -u scripts/evaluate_end_to_end.py --mode real --isolated-demo-stack --database-url 'postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test' --qdrant-url 'http://127.0.0.1:6335' --llm-model gemma3:4b --include-vision --include-agent --include-rag --include-similar-claims --save-baseline --output-dir artifacts/evaluation/e2e_v1/isolated_real

# Select a case (repeat --scenario to select multiple).
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode deterministic --scenario visual_non_detection --output-dir artifacts/evaluation/e2e_v1/non_detection

# A successful complete real-stack run may create the stable baseline.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --save-baseline --output-dir artifacts/evaluation/e2e_v1

# Compare a later real run against that baseline.
.\.venv\Scripts\python.exe scripts/evaluate_end_to_end.py --mode real --baseline artifacts/evaluation/e2e_v1/baseline_summary.json --output-dir artifacts/evaluation/e2e_v1/current
```

`--isolated-demo-stack` validates an explicitly selected test database, creates
a random schema migrated to the current Alembic head, and seeds only the
controlled demo claims/history. It creates three randomly named Qdrant
collections through the existing real embedding and indexing services. Actual
computed assessments are persisted for the real agent's existing tools; no
fake prediction is inserted. On completion it drops only the exact schema and
collections owned by that run. The setup audit is saved with the artifacts.
`--preflight-only` remains read-only, including when this flag is supplied.
The endpoints above are this project's local test-service ports; they are
explicit rather than inferred from the production application database.

Without isolated setup, the runner requires matching already prepared claims
in its configured database for history, similar claims and agent tools.
Unseeded controlled cases are skipped before any model loading. The installed
`gemma3:4b` model and IPv4 Ollama host above reproduce the local execution;
other environments can select another already installed model explicitly.

`--fail-fast` is opt-in; the default tries every compatible scenario.
`--no-include-vision`, `--no-include-agent`, `--no-include-rag` and
`--no-include-similar-claims` disable individual components explicitly.
`--no-include-diagnostics` omits the supplemental real diagnostics. Configuration
defaults and local checkpoint/source paths are in
`configs/evaluation/e2e_v1.toml`. The existing environment configuration controls
PostgreSQL, Qdrant, embedding and Ollama services. MLflow is optional; the report
records actual local model/checkpoint versions and loading sources.

The production agent first reads `GET_CLAIM` and `GET_INVESTIGATION`, then lets
the LLM plan further tool calls. Both reads count toward the existing eight-step
limit and require no LLM generation. An unavailable stored investigation remains
explicit uncertainty; the agent does not compute one or retry the initial read.
Investigation evidence includes bounded signal codes and concrete facts (such as
the extra quoted vehicle part), preserved in the finalizer context. The
`claims_agent_prompt_v5` instructions distinguish queue priority from unresolved
document discrepancies and ask for clarification or repair-expert review when
quote items are unexplained. The benchmark's acceptable actions remain unchanged.
For collected `ADDITIONAL_QUOTE_DAMAGE` or `UNMAPPED_QUOTE_ITEM` signals, the
application restricts the Ollama response schema to clarification, expert review,
or investigation review. The final rationale must cite the relevant assessment.
An ordinary queue priority does not remove this follow-up requirement. The same
checks run before accepting the recommendation, including with a custom LLM
adapter; incompatible output is rejected rather than rewritten. Without a
collected repair discrepancy, the original action choices remain available.

The generation schema also restricts source IDs to collected evidence and fixes
agent/prompt versions to their current values. Grounding failures retain a
specific application-owned code, such as `UNKNOWN_EVIDENCE_ID` or
`REPAIR_ASSESSMENT_NOT_CITED`, in the saved run and evaluation output. Raw rejected
drafts and provider exception details are not included in these messages.
For a correctable grounding failure, the finalizer requests one new response
using the same evidence and safe validation feedback. This adds at most one LLM
generation; a valid first response adds none. Transport errors on the initial
call and invalid source metadata are not retried. A second invalid response
still fails all normal checks; no decision or source is rewritten to pass.

When a restricted Windows sandbox cannot create pytest's default temporary
directory, use a writable workspace `--basetemp` directory after creating its
parent. This is an execution-environment workaround, not a reason to skip tests.

## Assertions, metrics and artifacts

`PASS` means every mandatory expectation was checked and satisfied. `FAIL`
means an executed expectation or component failed. `SKIP` means required
execution could not be completed; it never counts as a pass. The scenario pass
rate is `PASS / (PASS + FAIL)`; an all-skipped run has no rate. Coverage and
component skipped counts are reported alongside that denominator. Technical
component completion is not model accuracy.

Assertions validate the application's schemas first. Agent/investigation
decision values must never be `FRAUD_CONFIRMED`, `APPROVE_CLAIM`, `DENY_CLAIM` or
`PAYMENT_AUTHORIZED`. Human review stays mandatory. Agent rationale IDs must
resolve to collected `EvidenceItem`s; policy citations must resolve to retrieved
chunks with matching source/version/section. Document conclusions retain source
filename/type/page. Vision preserves image reference, damage/part model versions,
geometry and fusion policy. `VisualDamageType`, `VehiclePartType` and `DamageType`
remain separate typed domains. Anomaly and non-detection never establish fraud.

The report includes controlled fact recall, unexpected list members, required
signal recall, forbidden signal count, expected-source retrieval, citation
validity, agent schema/action/evidence validity, tool/blocked-duplicate counts,
and component durations. It reuses existing RAG source/section evaluation rather
than defining a competing retrieval accuracy benchmark. With fewer than twenty
timing samples, it reports individual durations and means; median/p95 require
sufficient samples. These timings identify bottlenecks, not hardware performance.

The output directory contains `summary.json`, `scenario_results.json`,
`scenario_results.csv`, `timings.csv`, structured `traces/` and `failures/`.
Failures retain expected, actual, component and source/trace information. Traces
contain structured inputs, outputs, tool calls, IDs and timings, with no hidden
chain-of-thought. Existing visualizations are referenced rather than duplicated.

Baseline creation is withheld for a partial run or a deterministic run. A valid
real baseline retains benchmark/model/policy versions, scenario statuses and
stable structured expectations. Comparison detects scenario regressions, missing
required or newly forbidden signals, retrieval failures and invalid citations.
Changing a saved expectation under the same benchmark version is also reported
as `benchmark_expectation_changed`.
Exact elapsed times, LLM prose and floating-point confidence values are excluded
from regression gates. A version or evaluation-mode mismatch is an error.

## Interpretation and limitations

The cases, policies and procedures are synthetic demonstrations. Some ML
components use synthetic training data. Vision evidence inherits the CarDD HF
mirror's custom test split and possible CV domain shift. Local LLMs may abstain,
fail structured output, or vary despite temperature zero; wording is not tested
for equality. A small controlled benchmark does not prove production insurance
accuracy, calibrated fraud probabilities or generalization to real claims.

Actual results, preflight versions, remaining pytest skips, observed failures,
limitations and exact reproduction commands are recorded in
`CODEX_REPORT_STEP_31.md`. Existing skip history remains in
`CODEX_TEST_SKIP_AUDIT.md`; new evaluation tests do not introduce skip gates.
