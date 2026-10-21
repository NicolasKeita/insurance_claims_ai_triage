"""Shared configuration and fixed dataset contracts."""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar(
    "T",
    bound=BaseModel,
)
