"""Run identity, durable case records and non-destructive resumption."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.deterministic.document_robustness import DOCUMENT_OCR_BENCHMARK_VERSION


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def corpus_payload(scenarios):
    return [
        {
            "scenario_id": scenario.scenario_id,
            "expected_consistency": scenario.expected_consistency,
            "documents": [
                {
                    "document_id": doc.document_id,
                    "document_type": doc.document_type,
                    "reference_text": doc.text,
                    "expected": doc.expected.model_dump(mode="json"),
                }
                for doc in scenario.documents
            ],
        }
        for scenario in scenarios
    ]


def run_identity(scenarios, preflight):
    from evaluation.deterministic.document_robustness.transformations import VARIANTS, variant_config

    payload = corpus_payload(scenarios)
    return {
        "benchmark_version": DOCUMENT_OCR_BENCHMARK_VERSION,
        "extraction_mode": "deterministic",
        "ocr_render_dpi": 300,
        "ocr_timeout_seconds": 30,
        "preflight": preflight,
        "degradations": {variant: variant_config(variant, "run-identity") for variant in VARIANTS},
        "corpus_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    }


def prepare_run(output, identity, resume):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume or a fresh --output-dir")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Resume identity differs (corpus, degradation, OCR configuration or environment)")
        if manifest.get("implementation_sha256") != implementation_fingerprint():
            raise ValueError("Resume implementation differs; use a fresh --output-dir after code changes")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output directory has no manifest; choose a fresh --output-dir")
    manifest = {
        "identity": identity,
        "implementation_sha256": implementation_fingerprint(),
        "created_at": timestamp(),
        "cases": {},
    }
    write_json(manifest_path, manifest, exclusive=True)
    return manifest


def load_records(output, manifest):
    records = []
    for key in sorted(manifest["cases"]):
        path = output / "results" / f"{key}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["case_key"] != key:
            raise ValueError(f"Case record identity differs: {path}")
        records.append(record)
    return records


def implementation_fingerprint():
    source = Path(__file__).resolve().parents[4]
    paths = list(Path(__file__).resolve().parents[1].rglob("*.py"))
    for name in ("document_ingestion", "document_extraction", "document_models", "ocr"):
        paths.extend((source / "claims" / name).rglob("*.py"))
    paths.append(source / "claims" / "consistency.py")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
