"""Shared base model, unit field type and absence wording for vision snapshots."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NO_DETECTION_SUMMARY = "No supported damage detected above the configured confidence threshold."


class VisionDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)