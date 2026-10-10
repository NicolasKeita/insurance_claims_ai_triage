from types import SimpleNamespace

import pytest

from knowledge.config import ROOT
from knowledge.evaluation import evaluate_retrieval, load_benchmark, section_metrics, write_evaluation


def test_metrics_known_ranks_and_duplicate_sections():
    results = [SimpleNamespace(source_id=s, section=t) for s, t in [("wrong", "A"), ("correct", "A"), ("correct", "A"), ("correct", "B")]]
    metrics = section_metrics(results, "correct", ["A", "B"])
    assert metrics == {"recall_at_1": 0, "recall_at_3": 0.5, "recall_at_5": 1, "mrr": 0.5}
    assert section_metrics([], "correct", ["A"])["mrr"] == 0
    with pytest.raises(ValueError, match="ground truth"):
        section_metrics([], None, [])


def test_complete_benchmark_and_unsupported_separate(tmp_path):
    class EmptyRetriever:
        calls = []

        def search(self, *args, **kwargs):
            self.calls.append(args)
            return []

    benchmark = load_benchmark(ROOT / "data/evaluation/rag_retrieval_v1.json")
    retriever = EmptyRetriever()
    report = evaluate_retrieval(retriever, benchmark)
    assert len(retriever.calls) == report["evaluated_cases"] == 13
    assert report["answerable_cases"] == 12 and report["unsupported_cases"] == 1
    assert report["unsupported"]["empty_result_rate"] == 1
    assert all(0 <= report[k] <= 1 for k in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"))
    assert "metrics" not in report["cases"][-1]
    write_evaluation(report, tmp_path)
    assert (tmp_path / "retrieval_metrics.json").is_file()
    assert (tmp_path / "retrieval_results.csv").is_file()
