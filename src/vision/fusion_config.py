"""Versioned demo thresholds, fixed before TEST and awaiting calibration on VAL."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VisionFusionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["vision_fusion_v1"] = "vision_fusion_v1"
    min_damage_confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    min_part_confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    min_damage_containment: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False)
    ambiguity_margin: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)

    @classmethod
    def from_toml(cls, path: Path):
        with Path(path).open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))


DEFAULT_FUSION_POLICY = VisionFusionPolicy()
