"""Sanitized failure payloads shared by the graph nodes."""

from agent.models import RunStatus


def failure(kind: str) -> dict:
    """
    Never persist raw exceptions, prompts, credentials or provider responses.
    """
    return {
        "status": RunStatus.FAILED.value,
        "recommendation": None,
        "error": f"Agent {kind.lower()} failed.",
        "failure_kind": kind,
    }
