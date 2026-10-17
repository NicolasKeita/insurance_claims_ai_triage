"""Real application adapters behind the opt-in evaluation pipeline."""

from evaluation.isolated_stack import isolated_demo_stack
from evaluation.real.diagnostics import real_diagnostic_scenarios
from evaluation.real.pipeline import RealClaimsEvaluationPipeline

__all__ = ["RealClaimsEvaluationPipeline", "isolated_demo_stack", "real_diagnostic_scenarios"]
