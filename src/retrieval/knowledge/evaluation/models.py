"""Section-level retrieval metrics, separate from generation and abstention."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval.knowledge.models import KnowledgeSourceType


class RetrievalEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    query: str
    source_type: KnowledgeSourceType
    product: str | None
    language: str = "en"
    as_of_date: date | None = None
    expected_source_id: str | None
    expected_sections: list[str]
    answerable: bool

    @model_validator(mode="after")
    def truth(self):
        has_truth = bool(self.expected_source_id and self.expected_sections)
        if self.answerable != has_truth:
            raise ValueError("Answerable cases need source/sections; unsupported cases have no ground truth")
        if not self.answerable and (self.expected_source_id is not None or self.expected_sections):
            raise ValueError("Unsupported cases must have empty ground truth")
        return self


class RetrievalBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    cases: list[RetrievalEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Benchmark IDs must be unique")
        return self


def load_benchmark(path):
    return RetrievalBenchmark.model_validate_json(path.read_text(encoding="utf-8"))
