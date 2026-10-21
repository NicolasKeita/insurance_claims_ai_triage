"""Section-level retrieval metrics, separate from generation and abstention."""

from retrieval.knowledge.evaluation.metrics import section_metrics
from retrieval.knowledge.evaluation.models import RetrievalBenchmark, RetrievalEvaluationCase, load_benchmark
from retrieval.knowledge.evaluation.reporting import write_evaluation
from retrieval.knowledge.evaluation.runner import evaluate_retrieval

__all__ = [
    "RetrievalBenchmark",
    "RetrievalEvaluationCase",
    "evaluate_retrieval",
    "load_benchmark",
    "section_metrics",
    "write_evaluation",
]
