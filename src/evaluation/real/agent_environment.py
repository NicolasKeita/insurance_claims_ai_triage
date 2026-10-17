"""Temporary environment override for the real agent's configured local LLM."""

import os
from contextlib import contextmanager


@contextmanager
def llm_environment(model):
    """Expose the configured model name to the application during the agent run."""
    previous = os.environ.get("CLAIMS_LLM_MODEL")
    os.environ["CLAIMS_LLM_MODEL"] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CLAIMS_LLM_MODEL", None)
        else:
            os.environ["CLAIMS_LLM_MODEL"] = previous
