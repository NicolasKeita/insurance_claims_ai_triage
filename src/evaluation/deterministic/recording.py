"""Record deterministic component results into an observation trace."""

from hashlib import sha256
from time import perf_counter

from claims.case import ClaimCase
from evaluation.deterministic.fixtures import SYNTHETIC_PROVENANCE
from evaluation.models import ComponentResult, TraceEvent


class TraceRecorder:
    """Append pass/fail component results, durations and sources to a trace."""

    def __init__(self, output):
        self.output = output

    def record(self, name, call, *, inputs=None, sources=None):
        started = perf_counter()
        try:
            result = call()
        except Exception as error:
            self._failed(name, started, error, inputs)
            raise
        snapshot = self._snapshot(name, result)
        duration = perf_counter() - started
        self.output.components[name] = ComponentResult(
            status="PASS", duration_seconds=duration, metadata={"backend": "deterministic"}
        )
        self.output.trace.append(
            TraceEvent(
                component=name,
                status="PASS",
                inputs=inputs or {},
                outputs=snapshot,
                duration_seconds=duration,
                source_references=sources or [],
            )
        )
        return result

    def _failed(self, name, started, error, inputs):
        duration = perf_counter() - started
        message = f"{type(error).__name__}: {error}"
        self.output.components[name] = ComponentResult(status="FAIL", duration_seconds=duration, reason=message)
        self.output.errors.append(f"{name}: {message}")
        self.output.trace.append(
            TraceEvent(
                component=name,
                status="FAIL",
                duration_seconds=duration,
                inputs=inputs or {},
                outputs={"error_type": type(error).__name__},
            )
        )

    def _snapshot(self, name, result):
        if isinstance(result, ClaimCase):
            return {"claim": result.claim.model_dump(mode="json"), "fixture_provenance": SYNTHETIC_PROVENANCE}
        if name == "document":
            return self._document_snapshot(result)
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return result if isinstance(result, dict) else {"completed": True}

    @staticmethod
    def _document_snapshot(result):
        extracted_models, pdf_texts = result
        return {
            "structured_extractions": {
                kind.value: model.model_dump(mode="json") for kind, model in extracted_models.items()
            },
            "documents": [
                {
                    "filename": filename,
                    "document_type": text.asset.type.value,
                    "page_count": text.page_count,
                    "method": text.method.value,
                    "pages": list(text.pages),
                    "text_sha256": sha256(text.text.encode("utf-8")).hexdigest(),
                    "asset_sha256": sha256(text.asset.path.read_bytes()).hexdigest(),
                }
                for filename, text in pdf_texts.items()
            ],
        }
