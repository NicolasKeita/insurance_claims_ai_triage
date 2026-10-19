"""Structured-output LLM access through a local Ollama server."""

from typing import Protocol, TypeVar

from httpx import RequestError
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredLlm(Protocol):
    def generate(self, *, prompt: str, response_model: type[T]) -> T: ...


class LlmError(Exception):
    pass


class OllamaStructuredLlm:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        system_prompt: str | None = None,
        timeout: float = 120,
        context_window: int | None = None,
    ):
        self.model = model
        if context_window is not None and (type(context_window) is not int or context_window < 1024):
            raise ValueError("context_window must be at least 1024 tokens")
        self.context_window = context_window
        self.system_prompt = system_prompt or (
            "You extract structured data from insurance documents. "
            "Use only information explicitly present in the document. "
            "Never invent missing information."
        )
        self.client = Client(host=host, timeout=timeout)

    def generate(self, *, prompt: str, response_model: type[T]) -> T:
        try:
            response = self.client.chat(**self._chat_arguments(prompt, response_model))
        except ResponseError as error:
            raise LlmError(f"Ollama request failed: {error}") from error
        except (ConnectionError, RequestError) as error:
            raise LlmError("Ollama connection failed or timed out") from error
        try:
            return response_model.model_validate_json(response.message.content)
        except ValidationError as error:
            raise LlmError(_invalid_response_message(response, error)) from error

    def _chat_arguments(self, prompt, response_model):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": 0,
                "seed": 42,
                **({"num_ctx": self.context_window} if self.context_window is not None else {}),
            },
            "stream": False,
        }


def _invalid_response_message(response, error):
    raw_response = response.message.content
    return f"LLM returned invalid structured data.\n\nRAW LLM RESPONSE:\n{raw_response}\n\nPYDANTIC ERROR:\n{error}"
