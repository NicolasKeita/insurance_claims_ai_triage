"""Read-only dependency probes for the explicit, opt-in real evaluation.

Heavy adapters are loaded once and retained for the run. No downloaded models,
service startup, writes to the database/vector stores, or fake replacements.
"""

from evaluation.preflight.components import ComponentProbes
from evaluation.preflight.config import DEFAULT_CONFIG, ROOT, RealEvaluationConfig
from evaluation.preflight.models import ModelProbes
from evaluation.preflight.report import PreflightReport, checkpoint_version, print_preflight_row
from evaluation.preflight.session import PreflightSession

__all__ = [
    "DEFAULT_CONFIG",
    "ROOT",
    "PreflightReport",
    "PreflightSession",
    "RealEvaluationConfig",
    "checkpoint_version",
    "print_preflight_row",
    "run_preflight",
]


def run_preflight(config: RealEvaluationConfig, *, progress=None) -> PreflightReport:
    """Probe every configured dependency and return the retained resources."""
    return PreflightSession(config, progress).run()
