from agent.diagnostics import grounding_error_text
from agent.models import RunStatus


def failure(kind: str, *, code=None) -> dict:
    """
    Never persist raw exceptions, prompts, credentials or provider responses.
    """
    return {
        "status": RunStatus.FAILED.value,
        "recommendation": None,
        "error": grounding_error_text(code) if kind == "GROUNDING" else f"Agent {kind.lower()} failed.",
        "failure_kind": kind,
    }
