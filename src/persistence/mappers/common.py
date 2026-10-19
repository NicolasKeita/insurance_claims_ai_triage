"""Shared generic record wrapper for stored assessments."""

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

Result = TypeVar("Result", bound=BaseModel)


@dataclass(frozen=True)
class AssessmentRecord(Generic[Result]):
    id: UUID
    created_at: datetime
    result: Result
