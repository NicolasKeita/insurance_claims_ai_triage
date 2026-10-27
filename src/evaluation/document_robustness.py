"""Public entry points for the independent CPU-only document OCR benchmark."""

from evaluation.deterministic.document_robustness import DOCUMENT_OCR_BENCHMARK_VERSION
from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.runtime.preflight import ocr_preflight
from evaluation.deterministic.document_robustness.runtime.session import BenchmarkConfig, run_benchmark

__all__ = ["DOCUMENT_OCR_BENCHMARK_VERSION", "BenchmarkConfig", "build_corpus", "ocr_preflight", "run_benchmark"]
