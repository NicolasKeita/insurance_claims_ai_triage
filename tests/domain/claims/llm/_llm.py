"""Llm."""

from tests.domain.claims.llm._constants import T


class FakeStructuredLlm:
    def __init__(
        self,
        payloads: dict | list[dict],
    ):
        if isinstance(payloads, dict):
            payloads = [payloads]
        self.payloads = payloads
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        prompt: str,
        response_model: type[T],
    ) -> T:
        index = min(
            self.call_count,
            len(self.payloads) - 1,
        )
        self.call_count += 1
        self.prompts.append(prompt)
        return response_model.model_validate(self.payloads[index])
