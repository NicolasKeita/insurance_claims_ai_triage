"""Frozen diagnostic manifest loading and provenance context."""

import json

from evaluation import E2E_BENCHMARK_VERSION
from evaluation.models import EvaluationScenario
from evaluation.preflight import ROOT


def real_diagnostic_scenarios() -> list[EvaluationScenario]:
    """Diagnostics supplement the frozen golden manifest without invented truth."""
    path = ROOT / "data/evaluation/e2e_v2/diagnostics.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["benchmark_version"] != E2E_BENCHMARK_VERSION:
        raise ValueError("Unsupported diagnostic manifest version")
    return [EvaluationScenario.model_validate(item) for item in manifest["diagnostics"]]


def diagnostic_context(declared_damage):
    """Explicit provenance for the synthetic external-image diagnostic."""
    return {
        "declaration_is_synthetic": True,
        "quoted_damage_is_synthetic": True,
        "declared_damage": list(declared_damage),
        "ground_truth_available": False,
        "detection_truth_labels_loaded": False,
        "meaning": "Conditional integration diagnostic; does not establish facts about an insurance claim",
        "existing_visualizations": "artifacts/environment_merge/gpu-smoke/run_20261015T135812_022547Z/summary.json",
    }
