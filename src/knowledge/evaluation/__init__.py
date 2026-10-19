"""Section-level retrieval metrics, separate from generation and abstention."""

from knowledge.evaluation.metrics import section_metrics
from knowledge.evaluation.models import RetrievalBenchmark, RetrievalEvaluationCase, load_benchmark
from knowledge.evaluation.reporting import write_evaluation
from knowledge.evaluation.runner import evaluate_retrieval

__all__ = [
    "RetrievalBenchmark",
    "RetrievalEvaluationCase",
    "evaluate_retrieval",
    "load_benchmark",
    "section_metrics",
    "write_evaluation",
]
