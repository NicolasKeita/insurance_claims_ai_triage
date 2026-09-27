from typing import Protocol, TypeVar

from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError


T = TypeVar(
    "T",
    bound=BaseModel,
)


class StructuredLlm(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        response_model: type[T],
    ) -> T:
        ...


class LlmError(Exception):
    pass


class OllamaStructuredLlm:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(
            host=host
        )

    def generate(
        self,
        *,
        prompt: str,
        response_model: type[T],
    ) -> T:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured data "
                            "from insurance documents. "
                            "Use only information explicitly "
                            "present in the document. "
                            "Never invent missing information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format=response_model.model_json_schema(),
                options={
                    "temperature": 0,
                },
                stream=False,
            )

        except ResponseError as error:
            raise LlmError(
                f"Ollama request failed: {error}"
            ) from error

        try:
            return response_model.model_validate_json(
                response.message.content
            )

        except ValidationError as error:
            raise LlmError(
                "LLM returned invalid structured data"
            ) from error