"""Lazy storage adapters keep domain errors usable without model dependencies."""

from importlib import import_module

from persistence.repositories.errors import ClaimAlreadyExistsError, ClaimNotFoundError, PolicyConflictError

__all__ = [
    "AssessmentRepository",
    "ClaimAlreadyExistsError",
    "ClaimImportResult",
    "ClaimNotFoundError",
    "ClaimRepository",
    "PolicyConflictError",
]

_EXPORTS = {
    "AssessmentRepository": "persistence.repositories.assessments",
    "ClaimImportResult": "persistence.repositories.claims",
    "ClaimRepository": "persistence.repositories.claims",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
