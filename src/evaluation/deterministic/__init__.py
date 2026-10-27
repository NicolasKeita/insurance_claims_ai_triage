"""Offline deterministic evaluation backend for the controlled benchmark."""

from importlib import import_module

_EXPORTS = {
    "analysis": ("investigate", "review_agent", "score_models"),
    "documents": ("extract_case_documents",),
    "fixtures": ("ROOT", "SYNTHETIC_PROVENANCE", "materialize_scenario_case"),
    "recording": ("TraceRecorder",),
    "retrieval": ("retrieve_knowledge", "retrieve_similar"),
    "stages": (
        "DeterministicRun",
        "assess_history",
        "build_features",
        "compare_documents",
        "extract_documents",
        "load_claim",
    ),
    "vision": ("assess_vision", "fake_vision_service"),
}
_MODULES = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = [
    "ROOT",
    "SYNTHETIC_PROVENANCE",
    "DeterministicRun",
    "TraceRecorder",
    "assess_history",
    "assess_vision",
    "build_features",
    "compare_documents",
    "extract_case_documents",
    "extract_documents",
    "fake_vision_service",
    "investigate",
    "load_claim",
    "materialize_scenario_case",
    "retrieve_knowledge",
    "retrieve_similar",
    "review_agent",
    "score_models",
]


def __getattr__(name):
    """Load only the requested evaluation stage, keeping document imports CPU-only."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
