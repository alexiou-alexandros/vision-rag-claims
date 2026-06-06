from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    damage_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask_area_pixels: int
    bbox_area_pixels: int
    mask: np.ndarray | None = None


class DetectionResult(BaseModel):
    detections: list[Detection]
    image_height: int
    image_width: int


class PolicyChunk(BaseModel):
    content: str
    source_document: str
    section: str
    metadata: dict = Field(default_factory=dict)


class CoverageDecision(BaseModel):
    damage_type: str
    is_covered: bool
    deductible: str = Field(
        description=(
            "Short dollar amount or 'N/A'. Examples: '$500', '$1,000', 'N/A'. "
            "Do NOT write a sentence — only the value."
        )
    )
    citations: list[str]
    reasoning: str


class SeverityLevel(str, Enum):
    minor = "minor"
    moderate = "moderate"
    severe = "severe"


class ClaimReport(BaseModel):
    summary: str
    damages: list[CoverageDecision]
    severity: SeverityLevel
    estimated_total: str
    recommended_action: str
    next_steps: list[str]
